"""Gauß–Taylor linearization of arrival time and lateral arrival position.

User / paper form (placeholders)::

    τ = τ̂ + ∇_x(E{x(τ̂)})ᵀ (x(τ̂) − E{x(τ̂)})
    y = E{y(τ̂)} + D(τ̂, E{x(τ̂)}) (τ − τ̂)
        + ∇_y(τ̂, E{x(τ̂)})ᵀ (x(τ̂) − E{x(τ̂)})

with ``τ̂ = τ̂(E{ms})`` and full-state vectors ``x(τ̂)``.  With the linear
Gaussian motion model ``x(τ̂) = A(τ̂) ms + w``, ``w ∼ N(0, C_w(τ̂))``,

    ξ = [τ, y]ᵀ = H ms + d + G w ,

so ``ξ | ms`` is Gaussian and, with Gaussian ``ms``, jointly Gaussian.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Placeholders — gradients / coeffs w.r.t. the full motion state
# ---------------------------------------------------------------------------

def hat_tau(mu: np.ndarray, x_pred_to: np.ndarray, state_dim: int, t_L: float = 0.0) -> np.ndarray:
    """
    Point prediction ``τ̂(E{ms})`` of the arrival time at the actuator boundary.

    :param mu: A np.array of shape ``[..., state_dim]``, the expected motion state
        ``E{ms}`` at the stage start.
    :param x_pred_to: A np.array of shape ``[...]`` (broadcastable to ``mu``), the
        streamwise actuator boundary position.
    :param state_dim: An integer in ``{2, 3, 4, 6}``, the motion-state length
        (CV/CA, 1D/2D).
    :param t_L: A float, the time of the last state (usually ``0`` within a stage).

    :returns: A np.array of shape ``[...]``, the predicted arrival times ``τ̂``.
    """
    if state_dim == 2 or state_dim == 4:
        # for CV (1D & 2D)
        return (x_pred_to - mu[..., 0]) / mu[..., 1] + t_L
    elif state_dim == 3 or state_dim == 6:
        # for CA (1D & 2D)
        delta_x = np.subtract(x_pred_to, mu[..., 0], dtype=float)
        sqrt_val = mu[..., 1] ** 2 + 2 * mu[..., 2] * delta_x
        pos_sqrt_val_mask = sqrt_val >= 0
        # sqrt with fallback sqrt = v if negative sqrt_val, so that dt = 2(x - x_0) / ( v + v ) = (x - x_0) / v
        sqrt = np.sqrt(sqrt_val, where=pos_sqrt_val_mask, out=np.array(mu[..., 1], copy=True))
        return np.sign(mu[..., 1]) * np.divide(
            2.0 * delta_x, np.abs(mu[..., 1]) + sqrt, dtype=float
        ) + t_L
    else:
        raise NotImplementedError(f"Invalid state dimension: {state_dim}")


def nabla_x(ex_at_hat_tau: np.ndarray, state_dim: int) -> np.ndarray:
    """
    Full-state gradient ``∇_x(E{x(τ̂)})`` of the arrival-time map.

    :param ex_at_hat_tau: A np.array of shape ``[..., state_dim]``, the expected
        full motion state ``E{x(τ̂)}`` at the predicted arrival time.
    :param state_dim: An integer in ``{2, 3, 4, 6}``, the motion-state length
        (kept for API symmetry; the CV/CA formula does not branch on it).

    :returns: A np.array of shape ``[..., state_dim]``, the gradient of arrival
        time w.r.t. the full motion state (only the streamwise-position entry is
        non-zero: ``∂τ/∂x = −1/v(τ̂)``).
    """
    # for CV, CA (all dimensions): ∂τ/∂x = −1/v(τ̂)
    grad = np.zeros_like(ex_at_hat_tau)
    grad[..., 0] = -1.0 / ex_at_hat_tau[..., 1]
    return grad


def D(tau_hat: np.ndarray, ex_at_hat_tau: np.ndarray, state_dim: int) -> np.ndarray:
    """
    Coupling coefficient ``D(τ̂, E{x(τ̂)})`` of lateral arrival to the time residual.

    :param tau_hat: A np.array of shape ``[...]``, the predicted arrival times
        (used for the output shape in 1D models).
    :param ex_at_hat_tau: A np.array of shape ``[..., state_dim]``, the expected
        full motion state at ``τ̂``.
    :param state_dim: An integer in ``{2, 3, 4, 6}``, the motion-state length.

    :returns: A np.array of shape ``[...]``, the scalar coupling ``D`` (zeros for
        1D models; lateral velocity at ``τ̂`` for 2D CV/CA).
    """
    if state_dim in (2, 3):
        # 1D models: no lateral channel
        return np.zeros(np.shape(tau_hat), dtype=float)
    if state_dim == 4:
        # for CV
        return ex_at_hat_tau[..., 3]
    if state_dim == 6:
        # for CA
        return ex_at_hat_tau[..., 4]
    raise NotImplementedError(f"Invalid state dimension: {state_dim}")


def nabla_y(tau_hat: np.ndarray, ex_at_hat_tau: np.ndarray, state_dim: int) -> np.ndarray:
    """
    Full-state gradient ``∇_y(τ̂, E{x(τ̂)})`` of the lateral arrival map.

    :param tau_hat: A np.array of shape ``[...]``, the predicted arrival times
        (unused; kept for API symmetry with :func:`D`).
    :param ex_at_hat_tau: A np.array of shape ``[..., state_dim]``, the expected
        full motion state at ``τ̂`` (determines the output shape).
    :param state_dim: An integer in ``{2, 3, 4, 6}``, the motion-state length.

    :returns: A np.array of shape ``[..., state_dim]``, the gradient of lateral
        arrival w.r.t. the full motion state (zeros in 1D; unit entry on the
        lateral position for 2D CV/CA).
    """
    if state_dim in (2, 3):
        return np.zeros_like(ex_at_hat_tau)
    if state_dim == 4:
        # for CV
        grad = np.zeros_like(ex_at_hat_tau)
        grad[..., 2] = 1.0
        return grad
    if state_dim == 6:
        # for CA
        grad = np.zeros_like(ex_at_hat_tau)
        grad[..., 3] = 1.0
        return grad
    raise NotImplementedError(f"Invalid state dimension: {state_dim}")


# ---------------------------------------------------------------------------
# LGSSM transition A(dt), C_w(dt)  (same construction as SNMPCParticleModel)
# ---------------------------------------------------------------------------

def lgssm_transition_matrices(
    dt: np.ndarray,
    S_w: Union[float, Tuple[float, float]],
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transition ``A(dt)`` and process-noise covariance ``C_w(dt)`` for CV/CA models.

    Matches ``SNMPCParticleModel`` (scalar ``S_w`` → 1D; length-2 → 2D).
    State dims: CV ``{2, 4}``, CA ``{3, 6}``.

    :param dt: A np.array of shape ``[...]``, the time increment(s); must be
        non-negative where finite (non-finite / negative values are treated as 0).
    :param S_w: A float or a length-2 sequence of floats, the process-noise power
        spectral density (scalar for 1D, per-axis for 2D).
    :param state_dim: An integer in ``{2, 3, 4, 6}``, the motion-state length.

    :returns: A tuple ``(A, C_w)`` of np.arrays of shape
        ``[..., state_dim, state_dim]``, the state-transition matrix and the
        discrete process-noise covariance for duration ``dt``.
    """
    dt = np.asarray(dt, dtype=float)
    leading = dt.shape
    S_w_arr = np.asarray(S_w, dtype=float)

    A = np.zeros(leading + (state_dim, state_dim), dtype=float)
    C = np.zeros(leading + (state_dim, state_dim), dtype=float)

    # Safe powers for possibly non-finite / negative dt (inactive actuators)
    dt_safe = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
    dt2 = dt_safe ** 2
    dt3 = dt_safe ** 3
    dt4 = dt_safe ** 4
    dt5 = dt_safe ** 5

    if state_dim == 2:
        # 1D CV
        sw = float(S_w_arr) if S_w_arr.ndim == 0 else float(S_w_arr[..., 0])
        A[..., 0, 0] = 1.0
        A[..., 0, 1] = dt_safe
        A[..., 1, 1] = 1.0
        C[..., 0, 0] = sw * dt3 / 3.0
        C[..., 0, 1] = sw * dt2 / 2.0
        C[..., 1, 0] = sw * dt2 / 2.0
        C[..., 1, 1] = sw * dt_safe
    elif state_dim == 3:
        # 1D CA
        sw = float(S_w_arr) if S_w_arr.ndim == 0 else float(S_w_arr[..., 0])
        A[..., 0, 0] = 1.0
        A[..., 0, 1] = dt_safe
        A[..., 0, 2] = dt2 / 2.0
        A[..., 1, 1] = 1.0
        A[..., 1, 2] = dt_safe
        A[..., 2, 2] = 1.0
        C[..., 0, 0] = sw * dt5 / 20.0
        C[..., 0, 1] = sw * dt4 / 8.0
        C[..., 0, 2] = sw * dt3 / 6.0
        C[..., 1, 0] = sw * dt4 / 8.0
        C[..., 1, 1] = sw * dt3 / 3.0
        C[..., 1, 2] = sw * dt2 / 2.0
        C[..., 2, 0] = sw * dt3 / 6.0
        C[..., 2, 1] = sw * dt2 / 2.0
        C[..., 2, 2] = sw * dt_safe
    elif state_dim == 4:
        # 2D CV
        swx = float(S_w_arr[0]) if S_w_arr.ndim else float(S_w_arr)
        swy = float(S_w_arr[1]) if S_w_arr.ndim else float(S_w_arr)
        A[..., 0, 0] = 1.0
        A[..., 0, 1] = dt_safe
        A[..., 1, 1] = 1.0
        A[..., 2, 2] = 1.0
        A[..., 2, 3] = dt_safe
        A[..., 3, 3] = 1.0
        C[..., 0, 0] = swx * dt3 / 3.0
        C[..., 0, 1] = swx * dt2 / 2.0
        C[..., 1, 0] = swx * dt2 / 2.0
        C[..., 1, 1] = swx * dt_safe
        C[..., 2, 2] = swy * dt3 / 3.0
        C[..., 2, 3] = swy * dt2 / 2.0
        C[..., 3, 2] = swy * dt2 / 2.0
        C[..., 3, 3] = swy * dt_safe
    elif state_dim == 6:
        # 2D CA
        swx = float(S_w_arr[0]) if S_w_arr.ndim else float(S_w_arr)
        swy = float(S_w_arr[1]) if S_w_arr.ndim else float(S_w_arr)
        for base, sw in ((0, swx), (3, swy)):
            A[..., base, base] = 1.0
            A[..., base, base + 1] = dt_safe
            A[..., base, base + 2] = dt2 / 2.0
            A[..., base + 1, base + 1] = 1.0
            A[..., base + 1, base + 2] = dt_safe
            A[..., base + 2, base + 2] = 1.0
            C[..., base, base] = sw * dt5 / 20.0
            C[..., base, base + 1] = sw * dt4 / 8.0
            C[..., base, base + 2] = sw * dt3 / 6.0
            C[..., base + 1, base] = sw * dt4 / 8.0
            C[..., base + 1, base + 1] = sw * dt3 / 3.0
            C[..., base + 1, base + 2] = sw * dt2 / 2.0
            C[..., base + 2, base] = sw * dt3 / 6.0
            C[..., base + 2, base + 1] = sw * dt2 / 2.0
            C[..., base + 2, base + 2] = sw * dt_safe
    else:
        raise ValueError(f"Unsupported state_dim={state_dim} (expected 2, 3, 4, or 6).")

    return A, C


def predict_motion_dt(
    motion_state_mean: np.ndarray,
    motion_state_cov: np.ndarray,
    dt: np.ndarray,
    S_w: Union[float, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts the motion state over duration ``dt`` (no wall collisions).

    Same prediction as ``SNMPCParticleModel.predict_motion`` for a free flight
    of length ``dt``: ``x' = A(dt) x``, ``P' = A P Aᵀ + C_w(dt)``.

    :param motion_state_mean: A np.array of shape ``[..., state_dim]``, the
        current motion-state mean.
    :param motion_state_cov: A np.array of shape ``[..., state_dim, state_dim]``,
        the current motion-state covariance.
    :param dt: A np.array broadcastable to the batch of ``motion_state_mean``,
        the prediction duration(s).
    :param S_w: A float or a length-2 sequence of floats, the process-noise PSD.

    :returns: A tuple ``(mean, cov)`` of np.arrays with the same leading shapes
        as the inputs, the predicted motion-state mean and covariance.
    """
    mu = np.asarray(motion_state_mean, dtype=float)
    P = np.asarray(motion_state_cov, dtype=float)
    state_dim = mu.shape[-1]
    A, C_w = lgssm_transition_matrices(dt, S_w, state_dim)
    mu_n = np.einsum("...ij,...j->...i", A, mu)
    P_n = np.einsum("...ij,...jk,...lk->...il", A, P, A) + C_w
    return mu_n, P_n


def lateral_index(state_dim: int) -> Optional[int]:
    """
    Returns the index of the lateral position in the motion state.

    :param state_dim: An integer in ``{2, 3, 4, 6}``, the motion-state length.

    :returns: An integer index of the lateral position coordinate, or ``None``
        for 1D models (``state_dim`` in ``{2, 3}``).
    """
    if state_dim in (2, 3):
        return None
    if state_dim == 4:
        return 2
    if state_dim == 6:
        return 3
    raise ValueError(f"Unsupported state_dim={state_dim}.")


# ---------------------------------------------------------------------------
# Assemble H, d, R  →  ξ = H ms + d + noise,  Cov(noise) = R
# ---------------------------------------------------------------------------

def gauss_taylor_linear_map(
    mu: np.ndarray,
    x_pred_to: np.ndarray,
    S_w: Union[float, Tuple[float, float]],
    t_L: float = 0.0,
    P: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds the affine map ``[τ, y]ᵀ = H ms + d + η`` with ``Cov(η) = R``.

    Uses ``A(τ̂ − t_L)``, ``C_w(τ̂ − t_L)`` as in the particle model, and

        E{x(τ̂)} = A μ ,
        ∇ terms from the placeholders on the full state,
        H = G A ,   R = G C_w Gᵀ ,
        G = [[∇_xᵀ], [(D ∇_x + ∇_y)ᵀ]].

    :param mu: A np.array of shape ``[..., state_dim]``, the motion-state mean
        at the stage start.
    :param x_pred_to: A np.array of shape ``[...]`` (broadcastable to ``mu``),
        the streamwise actuator boundary position.
    :param S_w: A float or a length-2 sequence of floats, the process-noise PSD.
    :param t_L: A float, the time of the last state (usually ``0`` within a stage).
    :param P: None or a np.array of shape ``[..., state_dim, state_dim]``.
        Unused for ``(H, d, R)``; kept for call-site symmetry.

    :returns: A tuple ``(H, d, R)`` where ``H`` is a np.array of shape
        ``[..., 2, state_dim]`` (Jacobian), ``d`` is a np.array of shape
        ``[..., 2]`` (offset), and ``R`` is a np.array of shape ``[..., 2, 2]``
        (additive process-noise covariance in ``ξ``-space).
    """
    del P
    mu = np.asarray(mu, dtype=float)
    x_pred_to = np.asarray(x_pred_to, dtype=float)
    state_dim = mu.shape[-1]
    leading = mu.shape[:-1]
    flat = int(np.prod(leading)) if leading else 1

    # Fast path: fused Numba kernels (no dense A/C_w materialization).
    try:
        from adf.adf_numba_kernels import (
            gauss_taylor_linear_map_ca3_numba,
            gauss_taylor_linear_map_ca6_numba,
            gauss_taylor_linear_map_cv2_numba,
            gauss_taylor_linear_map_cv4_numba,
            numba_enabled,
        )
    except ImportError:
        numba_enabled = lambda: False  # noqa: E731

    if numba_enabled() and flat >= 1:
        mu_f = np.ascontiguousarray(mu.reshape(flat, state_dim), dtype=np.float64)
        x_f = np.ascontiguousarray(
            np.broadcast_to(x_pred_to, leading).reshape(flat), dtype=np.float64
        )
        S_w_arr = np.asarray(S_w, dtype=float)
        if state_dim == 2:
            sw = float(S_w_arr) if S_w_arr.ndim == 0 else float(S_w_arr[..., 0])
            H, d, R = gauss_taylor_linear_map_cv2_numba(mu_f, x_f, sw, float(t_L))
        elif state_dim == 3:
            sw = float(S_w_arr) if S_w_arr.ndim == 0 else float(S_w_arr[..., 0])
            H, d, R = gauss_taylor_linear_map_ca3_numba(mu_f, x_f, sw, float(t_L))
        elif state_dim == 4:
            swx = float(S_w_arr[0]) if S_w_arr.ndim else float(S_w_arr)
            swy = float(S_w_arr[1]) if S_w_arr.ndim else float(S_w_arr)
            H, d, R = gauss_taylor_linear_map_cv4_numba(
                mu_f, x_f, swx, swy, float(t_L)
            )
        elif state_dim == 6:
            swx = float(S_w_arr[0]) if S_w_arr.ndim else float(S_w_arr)
            swy = float(S_w_arr[1]) if S_w_arr.ndim else float(S_w_arr)
            H, d, R = gauss_taylor_linear_map_ca6_numba(
                mu_f, x_f, swx, swy, float(t_L)
            )
        else:
            H = d = R = None
        if H is not None:
            return (
                H.reshape(leading + (2, state_dim)),
                d.reshape(leading + (2,)),
                R.reshape(leading + (2, 2)),
            )

    tau_hat = hat_tau(mu, x_pred_to, state_dim, t_L=t_L)
    dt = tau_hat - t_L
    A, C_w = lgssm_transition_matrices(dt, S_w, state_dim)

    # E{x(τ̂)} = A μ   (same mean update as predict_motion for duration dt)
    ex = np.einsum("...ij,...j->...i", A, mu)

    g_x = nabla_x(ex, state_dim)  # (..., state_dim)
    g_y = nabla_y(tau_hat, ex, state_dim)
    d_ty = D(tau_hat, ex, state_dim)  # (...)

    # G maps state residual / process noise at τ̂ into ξ
    # G_tau = g_x,  G_y = D g_x + g_y
    G_y = d_ty[..., None] * g_x + g_y
    G = np.stack([g_x, G_y], axis=-2)  # (..., 2, state_dim)

    H = np.einsum("...ai,...ij->...aj", G, A)  # (..., 2, state_dim)

    y_idx = lateral_index(state_dim)
    ey = ex[..., y_idx] if y_idx is not None else np.zeros(ex.shape[:-1], dtype=float)

    d = np.stack(
        [
            tau_hat - np.einsum("...i,...i->...", H[..., 0, :], mu),
            ey - np.einsum("...i,...i->...", H[..., 1, :], mu),
        ],
        axis=-1,
    )

    # R = G C_w Gᵀ
    R = np.einsum("...ai,...ij,...bj->...ab", G, C_w, G)
    R = 0.5 * (R + np.swapaxes(R, -1, -2))
    return H, d, R
