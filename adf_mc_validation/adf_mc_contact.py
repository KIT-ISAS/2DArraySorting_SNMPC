"""Fine-``Δt`` geometric contact sampling without Gauß–Taylor.

Simulates an LGSSM path over one controller stage ``[0, T]`` with micro-steps
and detects first passages at actuator ``x``-positions.  Phase membership uses
the same HIT/UP/DOWN time–lateral rectangles as the ADF geometry
(``lower`` / ``upper`` / ``phase_active`` / ``p_phase``).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple, Union

import numpy as np

from adf.adf_gauss_taylor import lateral_index, lgssm_transition_matrices


class FineDtContactResult(NamedTuple):
    """Outputs of one-stage fine-``Δt`` contact + motion.

    Attributes:
        ms_end: A np.array of shape ``[num_samples, num_particles, state_dim]``,
            motion state at the end of the stage (path endpoint).
        ejection: A np.array of shape ``[num_samples, num_particles]``, Bernoulli
            samples ``ej ∈ {0, 1}`` (0 for non-living inputs).
        first_contact_actor: A np.array of shape ``[num_samples, num_particles]``,
            index of the first actuator whose phase box was entered on a first
            passage in this stage; ``-1`` if none.
    """

    ms_end: np.ndarray
    ejection: np.ndarray
    first_contact_actor: np.ndarray


def _x_index(state_dim: int) -> int:
    """Returns the streamwise position index in the motion state (always 0).

    :param state_dim: An integer, the motion-state dimension (unused; kept for
        call-site symmetry with lateral helpers).

    :returns: An integer, the ``x`` component index.
    """
    del state_dim
    return 0


def _sample_gaussian_noise(
    rng: np.random.Generator,
    cov: np.ndarray,
    size: Tuple[int, ...],
) -> np.ndarray:
    """Samples ``N(0, cov)`` with shape ``size + (dx,)``.

    :param rng: A ``numpy.random.Generator``.
    :param cov: A np.array of shape ``[dx, dx]``, the covariance.
    :param size: A tuple of integers, leading sample dimensions.

    :returns: A np.array of shape ``size + (dx,)``.
    """
    dx = cov.shape[0]
    # Diagonal shortcut (common for tiny CV blocks after eigendecomp failure)
    try:
        return rng.multivariate_normal(
            mean=np.zeros(dx),
            cov=cov,
            size=size,
            check_valid="ignore",
        )
    except np.linalg.LinAlgError:
        # Fallback: eigendecomposition with clipped eigenvalues
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        z = rng.standard_normal(size + (dx,))
        return (z * np.sqrt(w)) @ v.T


def sample_stage_motion_and_ejection(
    ms: np.ndarray,
    living: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    phase_active: np.ndarray,
    actor_x: np.ndarray,
    p_phase: np.ndarray,
    S_w: Union[float, Tuple[float, float]],
    stage_T: float,
    num_micro_steps: int,
    rng: np.random.Generator,
    boundaries: Optional[Tuple[float, float]] = None,
) -> FineDtContactResult:
    """Propagates samples over one stage and samples geometric ejections.

    :param ms: A np.array of shape ``[num_samples, num_particles, state_dim]``,
        motion state at the beginning of the stage.
    :param living: A Boolean np.array of shape ``[num_samples, num_particles]``,
        ``True`` where existence is living (only living particles can be ejected).
    :param lower: A np.array of shape ``[num_actors, 3, 2]``, phase-rectangle lower
        bounds (``[..., 0]`` = time, ``[..., 1]`` = lateral ``y``).
    :param upper: A np.array of shape ``[num_actors, 3, 2]``, phase-rectangle upper
        bounds (same layout as ``lower``).
    :param phase_active: A Boolean np.array of shape ``[num_actors, 3]``, which
        HIT/UP/DOWN phases occur in this stage.
    :param actor_x: A np.array of shape ``[num_actors]``, actuator streamwise
        positions.
    :param p_phase: A np.array of shape ``[num_particles, 3]``, HIT/UP/DOWN ejection
        probabilities per particle class.
    :param S_w: A float or length-2 sequence, process-noise PSD (same convention as
        ``SNMPCParticleModel``).
    :param stage_T: A float, controller stage duration ``T``.
    :param num_micro_steps: An integer, number of micro-steps ``M``
        (``Δt = T / M``).
    :param rng: A ``numpy.random.Generator``.
    :param boundaries: None or a tuple ``(y_lo, y_hi)`` for wall reflection in 2D
        models.

    :returns: A :class:`FineDtContactResult` with end-of-stage motion, ejection
        indicators, and ``first_contact_actor``.
    """
    ms = np.asarray(ms, dtype=float)
    living = np.asarray(living, dtype=bool)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    phase_active = np.asarray(phase_active, dtype=bool)
    actor_x = np.asarray(actor_x, dtype=float)
    p_phase = np.asarray(p_phase, dtype=float)

    if ms.ndim != 3:
        raise ValueError("ms must have shape (num_samples, num_particles, state_dim)")
    num_samples, num_particles, state_dim = ms.shape
    if living.shape != (num_samples, num_particles):
        raise ValueError(f"living shape {living.shape} != {(num_samples, num_particles)}")
    if num_micro_steps < 1:
        raise ValueError("num_micro_steps must be >= 1")
    if stage_T <= 0:
        raise ValueError("stage_T must be positive")

    num_actors = int(actor_x.shape[0])
    if lower.shape[:2] != (num_actors, 3) or upper.shape[:2] != (num_actors, 3):
        raise ValueError("lower/upper must have shape (Na, 3, 2)")
    if phase_active.shape != (num_actors, 3):
        raise ValueError("phase_active must have shape (Na, 3)")
    if p_phase.shape != (num_particles, 3):
        raise ValueError(f"p_phase shape {p_phase.shape} != ({num_particles}, 3)")

    dt = float(stage_T) / float(num_micro_steps)
    A, C_w = lgssm_transition_matrices(np.asarray(dt), S_w, state_dim)
    A = np.asarray(A, dtype=float)
    C_w = np.asarray(C_w, dtype=float)

    x_ind = _x_index(state_dim)
    y_ind = lateral_index(state_dim)

    # Survival probability over actuators: start at 1, multiply (1 - p_hit_j)
    survive_prob = np.ones((num_samples, num_particles), dtype=float)
    # First-passage mask per actuator (not yet crossed)
    not_crossed = np.ones((num_samples, num_particles, num_actors), dtype=bool)
    first_contact_actor = np.full((num_samples, num_particles), -1, dtype=np.int32)

    state = ms.copy()
    for step in range(num_micro_steps):
        t0 = step * dt
        state_prev = state
        noise = _sample_gaussian_noise(rng, C_w, (num_samples, num_particles))
        state = np.einsum("ij,...j->...i", A, state_prev) + noise

        if boundaries is not None and y_ind is not None:
            y_lo_b, y_hi_b = float(boundaries[0]), float(boundaries[1])
            y = state[..., y_ind]
            below = y < y_lo_b
            above = y > y_hi_b
            state[..., y_ind] = np.where(below, 2.0 * y_lo_b - y, y)
            state[..., y_ind] = np.where(
                above, 2.0 * y_hi_b - state[..., y_ind], state[..., y_ind]
            )
            v_ind = y_ind + 1
            if v_ind < state_dim:
                state[..., v_ind] = np.where(
                    below | above, -state[..., v_ind], state[..., v_ind]
                )

        x0 = state_prev[..., x_ind]
        x1 = state[..., x_ind]
        dx = x1 - x0

        for j in range(num_actors):
            ax = float(actor_x[j])
            crossed = not_crossed[..., j] & living & (dx != 0.0) & (
                ((x0 < ax) & (x1 >= ax)) | ((x0 > ax) & (x1 <= ax))
            )
            if not np.any(crossed):
                continue

            alpha = np.zeros((num_samples, num_particles), dtype=float)
            alpha[crossed] = (ax - x0[crossed]) / dx[crossed]
            alpha = np.clip(alpha, 0.0, 1.0)
            tau = t0 + alpha * dt

            if y_ind is None:
                y_tau = np.zeros((num_samples, num_particles), dtype=float)
            else:
                y0 = state_prev[..., y_ind]
                y1 = state[..., y_ind]
                y_tau = y0 + alpha * (y1 - y0)

            p_hit_j = np.zeros((num_samples, num_particles), dtype=float)
            any_in_box = np.zeros((num_samples, num_particles), dtype=bool)
            for m in range(3):
                if not phase_active[j, m]:
                    continue
                t_lo = lower[j, m, 0]
                t_hi = upper[j, m, 0]
                y_lo = lower[j, m, 1]
                y_hi = upper[j, m, 1]
                in_box = (
                    crossed
                    & np.isfinite(t_lo)
                    & np.isfinite(t_hi)
                    & (tau >= t_lo)
                    & (tau < t_hi)
                    & (y_tau >= y_lo)
                    & (y_tau <= y_hi)
                )
                if not np.any(in_box):
                    continue
                any_in_box |= in_box
                p_m = p_phase[:, m]
                p_hit_j = 1.0 - (1.0 - p_hit_j) * (
                    1.0 - np.where(in_box, p_m[None, :], 0.0)
                )

            # First actuator whose phase box is entered on a first passage.
            assign = any_in_box & (first_contact_actor < 0)
            first_contact_actor = np.where(assign, j, first_contact_actor).astype(
                np.int32
            )

            survive_prob = survive_prob * (1.0 - p_hit_j)
            not_crossed[..., j] = not_crossed[..., j] & ~crossed

    p_eject = np.clip(1.0 - survive_prob, 0.0, 1.0)
    ejection = np.zeros((num_samples, num_particles), dtype=int)
    mask = living & (p_eject > 0.0)
    if np.any(mask):
        ejection[mask] = rng.binomial(1, p_eject[mask])
    return FineDtContactResult(
        ms_end=state,
        ejection=ejection,
        first_contact_actor=first_contact_actor,
    )


def sample_stage_gt_box_and_motion(
    ms: np.ndarray,
    living: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    R: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    phase_active: np.ndarray,
    p_phase: np.ndarray,
    A: np.ndarray,
    C_w: np.ndarray,
    rng: np.random.Generator,
) -> FineDtContactResult:
    """One-stage MC using the ADF soft-Bernoulli GT-box contact model.

    Samples ``ξ ~ N(H x + d, R)`` per actuator, applies the same phase-box
    Bernoulli union as ADF, leaves continuous states unchanged at contact, then
    applies a single LGSSM step ``(A, C_w)`` (stage length ``T``). This matches
    the ADF generative event (Gauß–Taylor + soft boxes), not geometric
    first-passage — use :func:`sample_stage_motion_and_ejection` for the latter.

    :param ms: A np.array of shape ``[num_samples, num_particles, state_dim]``,
        motion state at the beginning of the stage.
    :param living: A Boolean np.array of shape ``[num_samples, num_particles]``.
    :param H: A np.array of shape ``[num_particles, num_actors, 2, state_dim]``,
        Gauß–Taylor measurement maps.
    :param d: A np.array of shape ``[num_particles, num_actors, 2]``, affine offsets.
    :param R: A np.array of shape ``[num_particles, num_actors, 2, 2]``, measurement
        covariances.
    :param lower: A np.array of shape ``[num_actors, 3, 2]``, phase-rectangle lower
        bounds.
    :param upper: A np.array of shape ``[num_actors, 3, 2]``, phase-rectangle upper
        bounds.
    :param phase_active: A Boolean np.array of shape ``[num_actors, 3]``.
    :param p_phase: A np.array of shape ``[num_particles, 3]``, phase ejection
        probabilities.
    :param A: A np.array of shape ``[state_dim, state_dim]``, LGSSM transition.
    :param C_w: A np.array of shape ``[state_dim, state_dim]``, process-noise
        covariance for one stage step.
    :param rng: A ``numpy.random.Generator``.

    :returns: A :class:`FineDtContactResult`. For GT-box contact,
        ``first_contact_actor`` is ``-1`` (no geometric first-passage attribution).
    """
    ms = np.asarray(ms, dtype=float)
    living = np.asarray(living, dtype=bool)
    H = np.asarray(H, dtype=float)
    d = np.asarray(d, dtype=float)
    R = np.asarray(R, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    phase_active = np.asarray(phase_active, dtype=bool)
    p_phase = np.asarray(p_phase, dtype=float)
    A = np.asarray(A, dtype=float)
    C_w = np.asarray(C_w, dtype=float)

    if ms.ndim != 3:
        raise ValueError("ms must have shape (num_samples, num_particles, state_dim)")
    num_samples, num_particles, state_dim = ms.shape
    if living.shape != (num_samples, num_particles):
        raise ValueError(f"living shape {living.shape} != {(num_samples, num_particles)}")
    if H.ndim != 4 or H.shape[0] != num_particles or H.shape[-1] != state_dim:
        raise ValueError(f"H shape {H.shape} incompatible with ms {ms.shape}")
    num_actors = int(H.shape[1])
    if H.shape[2] != 2:
        raise ValueError(f"H must have shape (N, Na, 2, dx), got {H.shape}")
    if d.shape != (num_particles, num_actors, 2):
        raise ValueError(f"d shape {d.shape} != {(num_particles, num_actors, 2)}")
    if R.shape != (num_particles, num_actors, 2, 2):
        raise ValueError(f"R shape {R.shape} != {(num_particles, num_actors, 2, 2)}")
    if lower.shape[:2] != (num_actors, 3) or upper.shape[:2] != (num_actors, 3):
        raise ValueError("lower/upper must have shape (Na, 3, 2)")
    if phase_active.shape != (num_actors, 3):
        raise ValueError("phase_active must have shape (Na, 3)")
    if p_phase.shape != (num_particles, 3):
        raise ValueError(f"p_phase shape {p_phase.shape} != ({num_particles}, 3)")

    survive_prob = np.ones((num_samples, num_particles), dtype=float)
    for j in range(num_actors):
        p_hit_j = np.zeros((num_samples, num_particles), dtype=float)
        for m in range(3):
            if not phase_active[j, m]:
                continue
            t_lo, y_lo = lower[j, m]
            t_hi, y_hi = upper[j, m]
            if not (np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo):
                continue
            for p in range(num_particles):
                mask = living[:, p]
                if not np.any(mask):
                    continue
                n_m = int(np.count_nonzero(mask))
                xi = (
                    ms[mask, p] @ H[p, j].T
                    + d[p, j]
                    + _sample_gaussian_noise(rng, R[p, j], (n_m,))
                )
                in_box = (
                    (xi[:, 0] >= t_lo)
                    & (xi[:, 0] < t_hi)
                    & (xi[:, 1] >= y_lo)
                    & (xi[:, 1] <= y_hi)
                )
                pe = float(p_phase[p, m])
                if pe <= 0.0 or not np.any(in_box):
                    continue
                p_hit_j[mask, p] = 1.0 - (1.0 - p_hit_j[mask, p]) * (
                    1.0 - np.where(in_box, pe, 0.0)
                )
        survive_prob = survive_prob * (1.0 - p_hit_j)

    p_eject = np.clip(1.0 - survive_prob, 0.0, 1.0)
    ejection = np.zeros((num_samples, num_particles), dtype=int)
    mask = living & (p_eject > 0.0)
    if np.any(mask):
        ejection[mask] = rng.binomial(1, p_eject[mask])

    noise = _sample_gaussian_noise(rng, C_w, (num_samples, num_particles))
    ms_end = np.einsum("ij,...j->...i", A, ms) + noise
    first_contact_actor = np.full((num_samples, num_particles), -1, dtype=np.int32)
    return FineDtContactResult(
        ms_end=ms_end,
        ejection=ejection,
        first_contact_actor=first_contact_actor,
    )
