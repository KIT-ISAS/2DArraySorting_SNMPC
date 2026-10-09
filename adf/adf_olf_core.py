"""Controller-free ADF NumPy OLF prediction (shared by mixin and MC validation).

Does not import ``controller`` / ``simulator`` / TensorFlow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adf.adf_contact_model import adf_contact_moment_matching


def stage_olf_cost(
    m_ej: np.ndarray,
    is_acc: np.ndarray,
    is_rej: np.ndarray,
    accept_cost_weight: float,
    reject_cost_weight: float,
) -> np.ndarray:
    """Expected stage OLF cost ``E[g_n] = Σ_i (c_acc 1_acc − c_rej 1_rej) m_ej^i``.

    :param m_ej: A np.array with trailing particle axis, expected ejection mass.
    :param is_acc: A Boolean np.array of shape ``[num_particles]``, accept mask.
    :param is_rej: A Boolean np.array of shape ``[num_particles]``, reject mask.
    :param accept_cost_weight: A float, weight ``c_acc`` for accept ejections.
    :param reject_cost_weight: A float, weight ``c_rej`` for reject survivors /
        non-ejections in the signed stage form.

    :returns: A np.array of stage costs (batch axes of ``m_ej`` preserved).
    """
    return (
        accept_cost_weight * np.sum(m_ej * is_acc, axis=-1)
        - reject_cost_weight * np.sum(m_ej * is_rej, axis=-1)
    )


def terminal_olf_cost(
    pi_N: np.ndarray,
    particle_class: np.ndarray,
    accept_cost_weight: float,
    reject_cost_weight: float,
) -> np.ndarray:
    """Terminal OLF cost ``J = Σ_i c_acc 1(acc) π̃^{(0)} + Σ_i c_rej 1(rej) π̃^{(1)}``.

    :param pi_N: A np.array of shape ``[..., num_particles, 2]``, existence at the
        horizon end.
    :param particle_class: An integer np.array of shape ``[num_particles]``.
    :param accept_cost_weight: A float, weight ``c_acc``.
    :param reject_cost_weight: A float, weight ``c_rej``.

    :returns: A np.array of terminal costs (leading batch axes preserved).
    """
    is_acc = particle_class == 0
    is_rej = particle_class == 1
    cost_acc = accept_cost_weight * np.sum(pi_N[..., 0] * is_acc, axis=-1)
    cost_rej = reject_cost_weight * np.sum(pi_N[..., 1] * is_rej, axis=-1)
    return cost_acc + cost_rej


def parse_forward_inputs(
    u: np.ndarray,
    estimated_motion_state_mean: np.ndarray,
    estimated_motion_state_cov: np.ndarray,
    est_existence_probs: np.ndarray,
    t_act_k_at_T: np.ndarray,
):
    """Validates and normalizes ADF/MC forward inputs.

    :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
    :param estimated_motion_state_mean: A np.array of shape
        ``[num_particles, state_dim]``.
    :param estimated_motion_state_cov: A np.array of shape
        ``[num_particles, state_dim, state_dim]``.
    :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
        ``[batch_size, num_particles, 2]``.
    :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
        ``[batch_size, num_actors]``.

    :returns: A tuple
        ``(u, mu0, P0, pi_shared, t_act_shared, batch_size, num_actors,
        num_time_steps, num_particles, state_dim)`` with normalized arrays and
        shared (first-batch) ``π`` / ``t_act``.

    :raises ValueError: If shapes are inconsistent.
    """
    u = np.asarray(u, dtype=float)
    t_act0 = np.asarray(t_act_k_at_T, dtype=float)
    pi_in = np.asarray(est_existence_probs, dtype=float)
    mu0 = np.asarray(estimated_motion_state_mean, dtype=float)
    P0 = np.asarray(estimated_motion_state_cov, dtype=float)

    if u.ndim != 3:
        raise ValueError("u must have shape [batch_size, num_actors, num_time_steps]")
    batch_size, num_actors, num_time_steps = u.shape
    num_particles, state_dim = mu0.shape

    if pi_in.ndim == 3:
        if pi_in.shape != (batch_size, num_particles, 2):
            raise ValueError(
                f"est_existence_probs shape {pi_in.shape} != "
                f"({batch_size}, {num_particles}, 2)"
            )
        pi_shared = pi_in[0]
    elif pi_in.ndim == 2:
        if pi_in.shape != (num_particles, 2):
            raise ValueError(
                f"est_existence_probs shape {pi_in.shape} != ({num_particles}, 2)"
            )
        pi_shared = pi_in
    else:
        raise ValueError("est_existence_probs must be (N, 2) or (B, N, 2)")

    if t_act0.ndim == 2:
        if t_act0.shape != (batch_size, num_actors):
            raise ValueError(
                f"t_act_k_at_T shape {t_act0.shape} != ({batch_size}, {num_actors})"
            )
        t_act_shared = t_act0[0]
    elif t_act0.ndim == 1:
        if t_act0.shape != (num_actors,):
            raise ValueError(f"t_act_k_at_T shape {t_act0.shape} != ({num_actors},)")
        t_act_shared = t_act0
    else:
        raise ValueError("t_act_k_at_T must be (Na,) or (B, Na)")

    return (
        u,
        mu0,
        P0,
        pi_shared,
        t_act_shared,
        batch_size,
        num_actors,
        num_time_steps,
        num_particles,
        state_dim,
    )


def forward_chain_adf_numpy(
    *,
    particle_model: Any,
    contact_model: Any,
    actor_model: Any,
    u: np.ndarray,
    mu0: np.ndarray,
    P0: np.ndarray,
    pi_shared: np.ndarray,
    t_act_shared: np.ndarray,
    accept_cost_weight: float,
    reject_cost_weight: float,
    use_terminal: bool,
    collect_states: bool = False,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Staged NumPy ADF horizon using pre-normalized inputs.

    :param particle_model: Particle model with ``particle_class``, ``S_w``,
        ``predict_motion``.
    :param contact_model: Contact model with ``precompute_horizon_phase_geometry``
        and ``build_adf_gt_maps`` / moment matching helpers.
    :param actor_model: Actor model providing ``t_lead``.
    :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``,
        already validated.
    :param mu0: A np.array of shape ``[num_particles, state_dim]``.
    :param P0: A np.array of shape ``[num_particles, state_dim, state_dim]``.
    :param pi_shared: A np.array of shape ``[num_particles, 2]``, shared across the
        batch.
    :param t_act_shared: A np.array of shape ``[num_actors]``.
    :param accept_cost_weight: A float, ``c_acc``.
    :param reject_cost_weight: A float, ``c_rej``.
    :param use_terminal: A Boolean, terminal vs signed step costs.
    :param collect_states: A Boolean, if ``True`` also return per-stage
        ``π, μ, P``.

    :returns: A tuple ``(cost, states_or_None)`` where ``cost`` has shape
        ``[batch_size]`` and ``states`` (if requested) is a dict with
        ``pi_stages``, ``mu_stages``, ``P_stages``.
    """
    particle_class = np.asarray(particle_model.particle_class, dtype=int)
    u_lead = np.add(u, actor_model.t_lead, dtype=float)
    (
        _t_act_seq,
        lower_seq,
        upper_seq,
        phase_active_seq,
        p_phase,
        actuator_active_seq,
        actor_x,
    ) = contact_model.precompute_horizon_phase_geometry(
        t_act_k_at_T=t_act_shared,
        u_act_with_lead=u_lead,
        particle_class=particle_class,
    )

    batch_size, _num_actors, num_time_steps = u.shape
    num_particles, state_dim = mu0.shape

    pi = np.broadcast_to(pi_shared, (batch_size, num_particles, 2)).copy()
    mu = np.broadcast_to(mu0, (batch_size, num_particles, state_dim))
    P = np.broadcast_to(P0, (batch_size, num_particles, state_dim, state_dim))
    mu = np.stack([mu, mu], axis=2)
    P = np.stack([P, P], axis=2)

    step_cost = np.zeros(batch_size, dtype=float)
    is_acc = particle_class == 0
    is_rej = particle_class == 1

    pi_stages: List[np.ndarray] = []
    mu_stages: List[np.ndarray] = []
    P_stages: List[np.ndarray] = []
    if collect_states:
        pi_stages.append(pi.copy())
        mu_stages.append(mu.copy())
        P_stages.append(P.copy())

    for n in range(num_time_steps):
        H, d, R = contact_model.build_adf_gt_maps(
            motion_state_mean=mu[:, :, 1, :],
            S_w=particle_model.S_w,
            actuator_active=actuator_active_seq[n],
            actor_x=actor_x,
        )
        pi_before = pi
        adf = adf_contact_moment_matching(
            pi0=pi[..., 0],
            mu0=mu[:, :, 0, :],
            P0=P[:, :, 0, :, :],
            pi1=pi[..., 1],
            mu1=mu[:, :, 1, :],
            P1=P[:, :, 1, :, :],
            H=H,
            d=d,
            lower=lower_seq[n],
            upper=upper_seq[n],
            R=R,
            phase_active=phase_active_seq[n],
            p_phase=p_phase,
        )
        pi = adf.pi
        if not use_terminal:
            m_ej = np.maximum(pi[..., 0] - pi_before[..., 0], 0.0)
            step_cost = step_cost + stage_olf_cost(
                m_ej, is_acc, is_rej, accept_cost_weight, reject_cost_weight
            )
        mu_flat = adf.mu.reshape(batch_size * num_particles * 2, state_dim)
        P_flat = adf.P.reshape(batch_size * num_particles * 2, state_dim, state_dim)
        mu_pred, P_pred = particle_model.predict_motion(mu_flat, P_flat)
        mu = mu_pred.reshape(batch_size, num_particles, 2, state_dim)
        P = P_pred.reshape(batch_size, num_particles, 2, state_dim, state_dim)
        if collect_states:
            pi_stages.append(pi.copy())
            mu_stages.append(mu.copy())
            P_stages.append(P.copy())

    if use_terminal:
        cost = terminal_olf_cost(
            pi, particle_class, accept_cost_weight, reject_cost_weight
        )
    else:
        cost = step_cost

    if not collect_states:
        return cost, None
    return cost, {
        "pi_stages": pi_stages,
        "mu_stages": mu_stages,
        "P_stages": P_stages,
    }
