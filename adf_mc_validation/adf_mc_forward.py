"""Monte-Carlo open-loop forward prediction parallel to ADF ``_forward_chain``.

Uses fine-``Δt`` geometric contact (no Gauß–Taylor) and the same OLF cost
functionals as ``ADFOLFCostMixin``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from adf_mc_validation.adf_mc_contact import (
    sample_stage_gt_box_and_motion,
    sample_stage_motion_and_ejection,
)


def _sample_initial_states(
    rng: np.random.Generator,
    mu0: np.ndarray,
    P0: np.ndarray,
    pi: np.ndarray,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Samples initial motion and one-hot existence for the MC forward chain.

    :param rng: A ``numpy.random.Generator``.
    :param mu0: A np.array of shape ``[num_particles, state_dim]``, initial means.
    :param P0: A np.array of shape ``[num_particles, state_dim, state_dim]``, initial
        covariances.
    :param pi: A np.array of shape ``[num_particles, 2]``, existence probabilities
        ``(π^{(0)}, π^{(1)})``.
    :param num_samples: An integer, number of MC samples ``S``.

    :returns: A tuple ``(ms, ex)`` with ``ms`` of shape
        ``[num_samples, num_particles, state_dim]`` and ``ex`` of shape
        ``[num_samples, num_particles, 2]`` (one-hot).
    """
    num_particles, state_dim = mu0.shape
    ms = np.empty((num_samples, num_particles, state_dim), dtype=float)
    for i in range(num_particles):
        ms[:, i, :] = rng.multivariate_normal(mu0[i], P0[i], size=num_samples)
    alive = rng.binomial(1, pi[:, 1], size=(num_samples, num_particles))
    ex = np.stack([1 - alive, alive], axis=-1).astype(int)
    return ms, ex


def _empirical_pi(ex: np.ndarray) -> np.ndarray:
    """Converts one-hot existence samples to empirical frequencies.

    :param ex: A np.array of shape ``[num_samples, num_particles, 2]``, one-hot
        existence.

    :returns: A np.array of shape ``[num_particles, 2]``, mean frequencies over
        samples.
    """
    return ex.mean(axis=0).astype(float)


def forward_chain_mc(
    controller: Any,
    u: np.ndarray,
    estimated_motion_state_mean: np.ndarray,
    estimated_motion_state_cov: np.ndarray,
    est_existence_probs: np.ndarray,
    t_act_k_at_T: np.ndarray,
    num_samples: int = 2000,
    num_micro_steps: int = 100,
    return_diagnostics: bool = False,
    rng: Optional[np.random.Generator] = None,
    initial_time_step: int = 0,
    contact_mode: str = "geometric",
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Monte-Carlo analogue of :meth:`ADFOLFCostMixin._forward_chain`.

    Parameter shapes match ``_forward_chain`` plus MC controls. Only the first
    batch element of shared ``π`` / ``t_act`` is used when those arrays are batched
    (same convention as ADF).

    :param controller: An ADF-capable controller providing particle / actor /
        contact models and OLF cost weights.
    :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``,
        open-loop activation sequences.
    :param estimated_motion_state_mean: A np.array of shape
        ``[num_particles, state_dim]``.
    :param estimated_motion_state_cov: A np.array of shape
        ``[num_particles, state_dim, state_dim]``.
    :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
        ``[batch_size, num_particles, 2]``.
    :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
        ``[batch_size, num_actors]``, actors' internal time.
    :param num_samples: An integer, number of MC samples.
    :param num_micro_steps: An integer, fine-``Δt`` steps per stage (geometric mode).
    :param return_diagnostics: A Boolean, if ``True`` also return per-stage samples.
    :param rng: None or a ``numpy.random.Generator``; seeded from the controller if
        ``None``.
    :param initial_time_step: An integer, unused (kept for signature parity with ADF).
    :param contact_mode: A string, ``"geometric"`` (fine-``Δt`` first-passage, no
        Gauß–Taylor) or ``"gt_box"`` (ADF soft-Bernoulli boxes on GT maps).

    :returns: A np.array of shape ``[batch_size]`` with OLF costs, or a tuple
        ``(cost, diagnostics)`` when ``return_diagnostics`` is ``True``. Diagnostics
        include per-stage empirical ``pi``, motion samples, existence samples, and
        ``first_contact_actors_stages`` in geometric mode.

    :raises ValueError: If ``contact_mode`` is unknown or input shapes are invalid.
    """
    del initial_time_step
    if contact_mode not in ("geometric", "gt_box"):
        raise ValueError(f"Unknown contact_mode={contact_mode!r}")
    if rng is None:
        seed = getattr(controller, "_seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed) + 17)

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
        pi_shared = pi_in[0]
    elif pi_in.ndim == 2:
        pi_shared = pi_in
    else:
        raise ValueError("est_existence_probs must be (N, 2) or (B, N, 2)")
    if pi_shared.shape != (num_particles, 2):
        raise ValueError(f"est_existence_probs particle axis mismatch: {pi_shared.shape}")

    if t_act0.ndim == 2:
        t_act_shared = t_act0[0]
    elif t_act0.ndim == 1:
        t_act_shared = t_act0
    else:
        raise ValueError("t_act_k_at_T must be (Na,) or (B, Na)")
    if t_act_shared.shape != (num_actors,):
        raise ValueError(f"t_act_k_at_T shape {t_act_shared.shape} != ({num_actors},)")

    particle_class = np.asarray(controller._particle_model.particle_class, dtype=int)
    u_lead = np.add(u, controller._actor_model.t_lead, dtype=float)
    use_terminal = bool(controller._use_only_terminal_costs)
    stage_T = float(controller._T)  # we have to use the controller's internal T controller._T (for the prediction
    # horizon), which differs from the external T (controller.T) that describes the frequency in which the controller
    # is called
    S_w = controller._particle_model.S_w
    boundaries = getattr(controller._particle_model, "_boundaries", None)
    is_acc = particle_class == 0
    is_rej = particle_class == 1
    A_stage = np.asarray(controller._particle_model._A, dtype=float)
    C_w_stage = np.asarray(controller._particle_model._C_w, dtype=float)

    (
        _t_act_seq,
        lower_seq,
        upper_seq,
        phase_active_seq,
        p_phase,
        actuator_active_seq,
        actor_x,
    ) = controller._contact_model.precompute_horizon_phase_geometry(
        t_act_k_at_T=t_act_shared,
        u_act_with_lead=u_lead,
        particle_class=particle_class,
    )

    costs = np.zeros(batch_size, dtype=float)
    cost_var_acc = np.zeros(batch_size, dtype=float)  # for stderr of sample costs

    diag_pi: List[np.ndarray] = []
    diag_ms: List[np.ndarray] = []
    diag_ex: List[np.ndarray] = []
    diag_first_contact: List[np.ndarray] = []

    for b in range(batch_size):
        ms, ex = _sample_initial_states(rng, mu0, P0, pi_shared, num_samples)
        step_cost_samples = np.zeros(num_samples, dtype=float)

        if return_diagnostics and b == 0:
            diag_pi.append(_empirical_pi(ex))
            diag_ms.append(ms.copy())
            diag_ex.append(ex.copy())
            diag_first_contact.append(
                np.full((num_samples, num_particles), -1, dtype=np.int32)
            )

        for n in range(num_time_steps):
            living = ex[..., 1].astype(bool)
            if contact_mode == "geometric":
                result = sample_stage_motion_and_ejection(
                    ms=ms,
                    living=living,
                    lower=lower_seq[n, b],
                    upper=upper_seq[n, b],
                    phase_active=phase_active_seq[n, b],
                    actor_x=actor_x,
                    p_phase=p_phase,
                    S_w=S_w,
                    stage_T=stage_T,
                    num_micro_steps=num_micro_steps,
                    rng=rng,
                    boundaries=boundaries,
                )
            else:
                # Linearize GT maps at empirical living mean (per particle).
                mu_live = np.zeros((1, num_particles, state_dim), dtype=float)
                for p in range(num_particles):
                    mask_p = living[:, p]
                    if np.any(mask_p):
                        mu_live[0, p] = ms[mask_p, p].mean(axis=0)
                    else:
                        mu_live[0, p] = ms[:, p].mean(axis=0)
                H_b, d_b, R_b = controller._contact_model.build_adf_gt_maps(
                    motion_state_mean=mu_live,
                    S_w=S_w,
                    actuator_active=actuator_active_seq[n, b : b + 1],
                    actor_x=actor_x,
                )
                result = sample_stage_gt_box_and_motion(
                    ms=ms,
                    living=living,
                    H=H_b[0],
                    d=d_b[0],
                    R=R_b[0],
                    lower=lower_seq[n, b],
                    upper=upper_seq[n, b],
                    phase_active=phase_active_seq[n, b],
                    p_phase=p_phase,
                    A=A_stage,
                    C_w=C_w_stage,
                    rng=rng,
                )
            ej = result.ejection
            # Existence transition: dead absorbs; living → dead if ejected
            ex_next = ex.copy()
            became_dead = living & (ej == 1)
            ex_next[became_dead, 0] = 1
            ex_next[became_dead, 1] = 0
            # living & ~ej stay living; already dead stay dead

            if not use_terminal:
                # Stage cost sample: Σ_i (c_acc 1_acc − c_rej 1_rej) 1_{ej ∧ living}
                m_ej = (living & (ej == 1)).astype(float)  # (S, N)
                stage = (
                    controller._accept_cost_weight * (m_ej * is_acc).sum(axis=-1)
                    - controller._reject_cost_weight * (m_ej * is_rej).sum(axis=-1)
                )
                step_cost_samples = step_cost_samples + stage

            ms = result.ms_end
            ex = ex_next

            if return_diagnostics and b == 0:
                diag_pi.append(_empirical_pi(ex))
                diag_ms.append(ms.copy())
                diag_ex.append(ex.copy())
                diag_first_contact.append(np.asarray(result.first_contact_actor))

        if use_terminal:
            # Per-sample terminal cost, then average
            pi_hat = ex.astype(float)  # (S, N, 2) one-hot → indicators
            sample_costs = (
                controller._accept_cost_weight * (pi_hat[..., 0] * is_acc).sum(axis=-1)
                + controller._reject_cost_weight * (pi_hat[..., 1] * is_rej).sum(axis=-1)
            )
        else:
            sample_costs = step_cost_samples

        costs[b] = float(sample_costs.mean())
        if sample_costs.size > 1:
            cost_var_acc[b] = float(sample_costs.var(ddof=1))

    if not return_diagnostics:
        return costs

    diagnostics = {
        "pi_stages": diag_pi,  # list length N+1 of (num_particles, 2)
        "ms_stages": diag_ms,  # list length N+1 of (S, N, dx)
        "ex_stages": diag_ex,  # list length N+1 of (S, N, 2)
        # length N+1: stage 0 all -1; stage k>0 = first contact during transition k-1→k
        "first_contact_actors_stages": diag_first_contact,
        "cost_stderr": np.sqrt(cost_var_acc / max(num_samples, 1)),
        "num_samples": num_samples,
        "num_micro_steps": num_micro_steps,
        "contact_mode": contact_mode,
    }
    return costs, diagnostics
