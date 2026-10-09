"""Numba kernels for ADF truncation / projection hot paths.

Import is optional: if ``numba`` is missing, :data:`NUMBA_AVAILABLE` is ``False``
and callers keep using the NumPy implementations.

Toggle at runtime with :func:`set_numba_enabled` (PTCR config ``adf.use_numba``).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrap(fn):
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return wrap

    def prange(*args, **kwargs):  # type: ignore[misc]
        return range(*args)


_NUMBA_ENABLED = True


def set_numba_enabled(enabled: bool) -> None:
    """
    Enables or disables ADF Numba kernels at runtime.

    :param enabled: A Boolean. If False, callers use NumPy implementations only.
    """
    global _NUMBA_ENABLED
    _NUMBA_ENABLED = bool(enabled)


def numba_enabled() -> bool:
    """
    Whether ADF Numba kernels should be used.

    :returns: A Boolean, True if Numba is importable and enabled via
        :func:`set_numba_enabled`.
    """
    if not NUMBA_AVAILABLE:
        return False
    return _NUMBA_ENABLED


def warmup_adf_numba(state_dim: int = 4) -> None:
    """
    Triggers JIT compilation of ADF kernels (no-op if ``numba_enabled()`` is false).

    :param state_dim: An integer in ``{2, 3, 4, 6}``, the motion-state length used to
        select which Gauß–Taylor / horizon specialisations are compiled.
    """
    if not numba_enabled():
        return
    # Particles in single-stage projection / truncation kernels.
    k = 8
    n = int(state_dim)
    mu = np.zeros((k, n), dtype=np.float64)
    P = np.broadcast_to(np.eye(n, dtype=np.float64), (k, n, n)).copy()
    H = np.zeros((k, 2, n), dtype=np.float64)
    H[:, 0, 0] = 1.0
    H[:, 1, min(2, n - 1)] = 1.0
    d = np.zeros((k, 2), dtype=np.float64)
    R = np.broadcast_to(0.01 * np.eye(2, dtype=np.float64), (k, 2, 2)).copy()
    mu_xi, P_xi, Kg = forward_project_numba(mu, P, H, d, R)
    backward_project_numba(mu, P, mu_xi, P_xi, Kg, mu_xi, P_xi)
    mean = np.zeros((k, 2), dtype=np.float64)
    cov = np.broadcast_to(np.eye(2, dtype=np.float64), (k, 2, 2)).copy()
    lo = np.full((k, 2), -1.0)
    hi = np.full((k, 2), 1.0)
    moments_2d_uncorrelated_numba(mean, cov, lo, hi)
    cov_c = cov.copy()
    cov_c[:, 0, 1] = cov_c[:, 1, 0] = 0.2
    moments_2d_correlated_numba(mean, cov_c, lo, hi)
    bivariate_rect_cdf_numba(mean, cov_c, lo, hi)
    # Gauß–Taylor: compile the specialisation for this motion model only.
    mu[:, 0] = 0.0
    mu[:, 1] = 1.0
    if n >= 4:
        mu[:, 2] = 0.0
        mu[:, 3] = 0.0
    x_pred = np.full(k, 1.0, dtype=np.float64)
    if n == 2:
        gauss_taylor_linear_map_cv2_numba(mu, x_pred, 1.0, 0.0)
    elif n == 3:
        mu[:, 2] = 0.0
        gauss_taylor_linear_map_ca3_numba(mu, x_pred, 1.0, 0.0)
    elif n == 4:
        gauss_taylor_linear_map_cv4_numba(mu, x_pred, 1.0, 1.0, 0.0)
    elif n == 6:
        mu[:, 2] = 0.0
        mu[:, 4] = 0.0
        mu[:, 5] = 0.0
        gauss_taylor_linear_map_ca6_numba(mu, x_pred, 1.0, 1.0, 0.0)
    A = np.eye(n, dtype=np.float64)
    if n >= 2:
        A[0, 1] = 0.1
    if n >= 4:
        A[2, 3] = 0.1
    C_w = 0.01 * np.eye(n, dtype=np.float64)
    predict_motion_numba(mu, P, A, C_w, False, 0.0, 1.0, 0)
    # Actor model (Variant B): one particle row per actor slot.
    t_act = np.zeros(k, dtype=np.float64)
    u_act = np.full(k, 0.05, dtype=np.float64)
    predict_actor_state_numba(
        t_act, u_act, 0.1, 0.1, 0.25, 0.02, 0.02, 0.04
    )
    actor_phase_intervals_numba(
        t_act, u_act, 0.1, 0.25, 0.02, 0.02, 0.04, 0.02
    )
    # Fused phased contact: minimal B × N × Na grid (see docstring).
    B, N, Na, n = 2, 4, 3, 4
    pi0 = np.full((B, N), 0.1)
    pi1 = np.full((B, N), 0.9)
    mu0 = np.zeros((B, N, n))
    mu1 = np.zeros((B, N, n))
    mu1[..., 1] = 1.0
    P0 = np.broadcast_to(np.eye(n), (B, N, n, n)).copy()
    P1 = P0.copy()
    H = np.zeros((B, N, Na, 2, n))
    H[..., 0, 0] = 1.0
    H[..., 1, 2] = 1.0
    d = np.zeros((B, N, Na, 2))
    R = np.broadcast_to(0.01 * np.eye(2), (B, N, Na, 2, 2)).copy()
    lower = np.zeros((B, Na, 3, 2))
    upper = np.ones((B, Na, 3, 2))
    phase_active = np.zeros((B, Na, 3), dtype=np.bool_)
    phase_active[:, 0, 0] = True
    p_phase = np.full((N, 3), 0.5)
    adf_phased_contact_match_numba(
        pi0, mu0, P0, pi1, mu1, P1, H, d, R, lower, upper, phase_active, p_phase, 1e-15
    )
    Tsteps = 2
    lower_h = np.zeros((Tsteps, B, Na, 3, 2), dtype=np.float64)
    upper_h = np.ones((Tsteps, B, Na, 3, 2), dtype=np.float64)
    ph_h = np.zeros((Tsteps, B, Na, 3), dtype=np.bool_)
    ph_h[:, :, 0, 0] = True
    actor_x_w = np.arange(Na, dtype=np.float64) + 1.0
    Aw = np.eye(n, dtype=np.float64)
    Cw = 0.01 * np.eye(n, dtype=np.float64)
    is_acc = np.zeros(N, dtype=np.bool_)
    is_rej = np.ones(N, dtype=np.bool_)
    mu_init = np.zeros((N, n), dtype=np.float64)
    mu_init[:, 1] = 1.0
    if n >= 4:
        mu_init[:, 3] = 0.0
    P_init = np.broadcast_to(np.eye(n, dtype=np.float64), (N, n, n)).copy()
    pi_shared = np.stack(
        [np.full(N, 0.1), np.full(N, 0.9)], axis=-1
    ).astype(np.float64)
    # Terminal-cost horizon path (use_terminal_cost=True).
    adf_olf_horizon_numba(
        pi_shared,
        mu_init,
        P_init,
        lower_h,
        upper_h,
        ph_h,
        p_phase,
        actor_x_w,
        Aw,
        Cw,
        1.0,
        1.0,
        False,
        0.0,
        1.0,
        2 if n >= 4 else 0,
        is_acc,
        is_rej,
        0.5,
        0.5,
        True,
        1e-15,
    )
    # Step-cost horizon path (use_terminal_cost=False).
    adf_olf_horizon_numba(
        pi_shared,
        mu_init,
        P_init,
        lower_h,
        upper_h,
        ph_h,
        p_phase,
        actor_x_w,
        Aw,
        Cw,
        1.0,
        1.0,
        False,
        0.0,
        1.0,
        2 if n >= 4 else 0,
        is_acc,
        is_rej,
        0.5,
        0.5,
        False,
        1e-15,
    )
    # Sequential horizon phase-geometry precompute
    t0 = np.zeros(Na, dtype=np.float64)
    u_h = np.full((B, Na, Tsteps), 0.05, dtype=np.float64)
    y_lo_w = np.full(Na, -1.0, dtype=np.float64)
    y_hi_w = np.full(Na, 1.0, dtype=np.float64)
    precompute_horizon_phase_geometry_numba(
        t0, u_h, y_lo_w, y_hi_w, 0.1, 0.25, 0.02, 0.02, 0.04, 0.02
    )

    # CA motion models: compile n=3 and n=6 even when state_dim is CV.
    for n_ca, y_ind_ca in ((3, 0), (6, 3)):
        Bc, Nc, Nac = 2, 3, 2
        pi_c = np.stack(
            [np.full(Nc, 0.1), np.full(Nc, 0.9)], axis=-1
        ).astype(np.float64)
        mu_c = np.zeros((Nc, n_ca), dtype=np.float64)
        mu_c[:, 1] = 1.0
        P_c = np.broadcast_to(np.eye(n_ca, dtype=np.float64), (Nc, n_ca, n_ca)).copy()
        lower_c = np.zeros((2, Bc, Nac, 3, 2), dtype=np.float64)
        upper_c = np.ones((2, Bc, Nac, 3, 2), dtype=np.float64)
        ph_c = np.zeros((2, Bc, Nac, 3), dtype=np.bool_)
        ph_c[:, :, 0, 0] = True
        p_c = np.full((Nc, 3), 0.5)
        ax_c = np.arange(Nac, dtype=np.float64) + 1.0
        Ac = np.eye(n_ca, dtype=np.float64)
        Cc = 0.01 * np.eye(n_ca, dtype=np.float64)
        is_ac = np.zeros(Nc, dtype=np.bool_)
        is_rj = np.ones(Nc, dtype=np.bool_)
        adf_olf_horizon_numba(
            pi_c, mu_c, P_c, lower_c, upper_c, ph_c, p_c,
            ax_c, Ac, Cc, 1.0, 1.0, False, 0.0, 1.0, y_ind_ca,
            is_ac, is_rj, 0.5, 0.5, True, 1e-15,
        )
    # Fused phased contact (n=4 typical)
    B, N, Na, n = 2, 4, 3, 4
    pi0 = np.full((B, N), 0.1)
    pi1 = np.full((B, N), 0.9)
    mu0 = np.zeros((B, N, n))
    mu1 = np.zeros((B, N, n))
    mu1[..., 1] = 1.0
    P0 = np.broadcast_to(np.eye(n), (B, N, n, n)).copy()
    P1 = P0.copy()
    H = np.zeros((B, N, Na, 2, n))
    H[..., 0, 0] = 1.0
    H[..., 1, 2] = 1.0
    d = np.zeros((B, N, Na, 2))
    R = np.broadcast_to(0.01 * np.eye(2), (B, N, Na, 2, 2)).copy()
    lower = np.zeros((B, Na, 3, 2))
    upper = np.ones((B, Na, 3, 2))
    phase_active = np.zeros((B, Na, 3), dtype=np.bool_)
    phase_active[:, 0, 0] = True
    p_phase = np.full((N, 3), 0.5)
    adf_phased_contact_match_numba(
        pi0, mu0, P0, pi1, mu1, P1, H, d, R, lower, upper, phase_active, p_phase, 1e-15
    )


# ---------------------------------------------------------------------------
# Scalar specials
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _ndtr(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))


@njit(cache=True, fastmath=True)
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) * 0.3989422804014327  # 1/sqrt(2π)


@njit(cache=True, fastmath=True)
def _trunc1d(
    mu: float, var: float, a: float, b: float, eps: float
) -> tuple[float, float, float]:
    v = var if var > eps else eps
    sigma = math.sqrt(v)
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    if not math.isfinite(alpha):
        phi_a = 0.0
        Phi_a = 0.0 if alpha < 0.0 else 1.0
        alpha_s = 0.0
    else:
        phi_a = _norm_pdf(alpha)
        Phi_a = _ndtr(alpha)
        alpha_s = alpha
    if not math.isfinite(beta):
        phi_b = 0.0
        Phi_b = 0.0 if beta < 0.0 else 1.0
        beta_s = 0.0
    else:
        phi_b = _norm_pdf(beta)
        Phi_b = _ndtr(beta)
        beta_s = beta

    m = Phi_b - Phi_a
    if m < 0.0:
        m = 0.0
    elif m > 1.0:
        m = 1.0
    if m > eps:
        dens = (phi_a - phi_b) / m
        mean = mu + sigma * dens
        edge = (alpha_s * phi_a - beta_s * phi_b) / m
        vs = 1.0 + edge - dens * dens
        if vs < 0.0:
            vs = 0.0
        return m, mean, vs * v
    return 0.0, mu, v


# ---------------------------------------------------------------------------
# 1D / uncorrelated 2D
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True, fastmath=True)
def moments_1d_truncated_gaussian_numba(
    mu: np.ndarray,
    var: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba kernel for 1D truncated-Gaussian mass, mean and variance.

    :param mu: A 1-D np.array of means.
    :param var: A 1-D np.array of variances.
    :param a: A 1-D np.array of lower bounds.
    :param b: A 1-D np.array of upper bounds.
    :param eps: A float, the numerical floor.

    :returns: A tuple ``(mass, mean, var)`` of 1-D np.arrays.
    """
    k = mu.shape[0]
    mass = np.empty(k, dtype=np.float64)
    mean = np.empty(k, dtype=np.float64)
    var_t = np.empty(k, dtype=np.float64)
    for i in prange(k):
        mass[i], mean[i], var_t[i] = _trunc1d(mu[i], var[i], a[i], b[i], eps)
    return mass, mean, var_t


@njit(cache=True, parallel=True, fastmath=True)
def moments_2d_uncorrelated_numba(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba kernel for uncorrelated 2D rectangular truncated-Gaussian moments.

    :param mean: A np.array of shape ``[K, 2]``.
    :param cov: A np.array of shape ``[K, 2, 2]`` (off-diagonals ignored).
    :param lower: A np.array of shape ``[K, 2]``.
    :param upper: A np.array of shape ``[K, 2]``.
    :param eps: A float, the numerical floor.

    :returns: A tuple ``(mass, mean_trunc, cov_trunc)``.
    """
    k = mean.shape[0]
    mass = np.empty(k, dtype=np.float64)
    mean_t = np.empty((k, 2), dtype=np.float64)
    cov_t = np.zeros((k, 2, 2), dtype=np.float64)
    for i in prange(k):
        m0, mu0, v0 = _trunc1d(
            mean[i, 0], cov[i, 0, 0], lower[i, 0], upper[i, 0], eps
        )
        m1, mu1, v1 = _trunc1d(
            mean[i, 1], cov[i, 1, 1], lower[i, 1], upper[i, 1], eps
        )
        mass[i] = m0 * m1
        mean_t[i, 0] = mu0
        mean_t[i, 1] = mu1
        cov_t[i, 0, 0] = v0
        cov_t[i, 1, 1] = v1
    return mass, mean_t, cov_t


# ---------------------------------------------------------------------------
# Forward / backward (m = 2)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True, fastmath=True)
def forward_project_numba(
    mu: np.ndarray,
    P: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba kernel for forward projection ``μ_ξ, P_ξ, K``.

    :param mu: A np.array of shape ``[K, n]``.
    :param P: A np.array of shape ``[K, n, n]``.
    :param H: A np.array of shape ``[K, m, n]``.
    :param d: A np.array of shape ``[K, m]``.
    :param R: A np.array of shape ``[K, m, m]``.

    :returns: A tuple ``(mu_xi, P_xi, K)``.
    """
    k = H.shape[0]
    n = H.shape[2]
    mu_xi = np.empty((k, 2), dtype=np.float64)
    P_xi = np.empty((k, 2, 2), dtype=np.float64)
    Kgain = np.empty((k, n, 2), dtype=np.float64)

    for i in prange(k):
        for r in range(2):
            s = d[i, r]
            for c in range(n):
                s += H[i, r, c] * mu[i, c]
            mu_xi[i, r] = s

        tmp0 = np.empty(n, dtype=np.float64)
        tmp1 = np.empty(n, dtype=np.float64)
        for c in range(n):
            s0 = 0.0
            s1 = 0.0
            for t in range(n):
                s0 += H[i, 0, t] * P[i, t, c]
                s1 += H[i, 1, t] * P[i, t, c]
            tmp0[c] = s0
            tmp1[c] = s1

        p00 = R[i, 0, 0]
        p01 = R[i, 0, 1]
        p10 = R[i, 1, 0]
        p11 = R[i, 1, 1]
        for t in range(n):
            p00 += tmp0[t] * H[i, 0, t]
            p01 += tmp0[t] * H[i, 1, t]
            p10 += tmp1[t] * H[i, 0, t]
            p11 += tmp1[t] * H[i, 1, t]
        P_xi[i, 0, 0] = p00
        P_xi[i, 0, 1] = p01
        P_xi[i, 1, 0] = p10
        P_xi[i, 1, 1] = p11

        det = p00 * p11 - p01 * p10
        if abs(det) < 1e-30:
            det = 1e-30 if det >= 0.0 else -1e-30
        # NumPy path: K = (P_xi^{-1} (H P))ᵀ = (P Hᵀ) (P_ξ^{-1})ᵀ
        invT00 = p11 / det
        invT01 = -p10 / det
        invT10 = -p01 / det
        invT11 = p00 / det

        for r in range(n):
            ht0 = 0.0
            ht1 = 0.0
            for t in range(n):
                ht0 += P[i, r, t] * H[i, 0, t]
                ht1 += P[i, r, t] * H[i, 1, t]
            Kgain[i, r, 0] = ht0 * invT00 + ht1 * invT10
            Kgain[i, r, 1] = ht0 * invT01 + ht1 * invT11

    return mu_xi, P_xi, Kgain


@njit(cache=True, parallel=True, fastmath=True)
def backward_project_numba(
    mu: np.ndarray,
    P: np.ndarray,
    mu_xi: np.ndarray,
    P_xi: np.ndarray,
    K: np.ndarray,
    mu_y: np.ndarray,
    P_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba kernel for backward projection of truncated measurement moments.

    :param mu: A np.array of shape ``[K, n]``.
    :param P: A np.array of shape ``[K, n, n]``.
    :param mu_xi: A np.array of shape ``[K, m]``.
    :param P_xi: A np.array of shape ``[K, m, m]``.
    :param K: A np.array of shape ``[K, n, m]``.
    :param mu_y_trunc: A np.array of shape ``[K, m]``.
    :param P_y_trunc: A np.array of shape ``[K, m, m]``.

    :returns: A tuple ``(mu_post, P_post)``.
    """
    k = K.shape[0]
    n = K.shape[1]
    mu_x = np.empty((k, n), dtype=np.float64)
    P_x = np.empty((k, n, n), dtype=np.float64)

    for i in prange(k):
        d0 = mu_y[i, 0] - mu_xi[i, 0]
        d1 = mu_y[i, 1] - mu_xi[i, 1]
        for r in range(n):
            mu_x[i, r] = mu[i, r] + K[i, r, 0] * d0 + K[i, r, 1] * d1

        dp00 = P_y[i, 0, 0] - P_xi[i, 0, 0]
        dp11 = P_y[i, 1, 1] - P_xi[i, 1, 1]
        dp01 = P_y[i, 0, 1] - P_xi[i, 0, 1]
        dp10 = P_y[i, 1, 0] - P_xi[i, 1, 0]

        for r in range(n):
            t0 = K[i, r, 0] * dp00 + K[i, r, 1] * dp10
            t1 = K[i, r, 0] * dp01 + K[i, r, 1] * dp11
            for c in range(n):
                P_x[i, r, c] = P[i, r, c] + t0 * K[i, c, 0] + t1 * K[i, c, 1]

        for r in range(n):
            for c in range(r + 1, n):
                s = 0.5 * (P_x[i, r, c] + P_x[i, c, r])
                P_x[i, r, c] = s
                P_x[i, c, r] = s

    return mu_x, P_x


# ---------------------------------------------------------------------------
# Genz BVN rectangle CDF
# ---------------------------------------------------------------------------

_GL_W6 = np.array(
    [0.1713244923791705, 0.3607615730481384, 0.4679139345726904] * 2,
    dtype=np.float64,
)
_GL_X6 = np.array(
    [
        1.0 - 0.9324695142031522,
        1.0 - 0.6612093864662647,
        1.0 - 0.2386191860831970,
        1.0 + 0.9324695142031522,
        1.0 + 0.6612093864662647,
        1.0 + 0.2386191860831970,
    ],
    dtype=np.float64,
)
_GL_W12 = np.array(
    [
        0.04717533638651177,
        0.1069393259953183,
        0.1600783285433464,
        0.2031674267230659,
        0.2334925365383547,
        0.2491470458134029,
    ]
    * 2,
    dtype=np.float64,
)
_GL_X12 = np.array(
    [
        1.0 - 0.9815606342467191,
        1.0 - 0.9041172563704750,
        1.0 - 0.7699026741943050,
        1.0 - 0.5873179542866171,
        1.0 - 0.3678314989981802,
        1.0 - 0.1252334085114692,
        1.0 + 0.9815606342467191,
        1.0 + 0.9041172563704750,
        1.0 + 0.7699026741943050,
        1.0 + 0.5873179542866171,
        1.0 + 0.3678314989981802,
        1.0 + 0.1252334085114692,
    ],
    dtype=np.float64,
)
_GL_W20 = np.array(
    [
        0.01761400713915212,
        0.04060142980038694,
        0.06267204833410906,
        0.08327674157670475,
        0.1019301198172404,
        0.1181945319615184,
        0.1316886384491766,
        0.1420961093183821,
        0.1491729864726037,
        0.1527533871307259,
    ]
    * 2,
    dtype=np.float64,
)
_GL_X20 = np.array(
    [
        1.0 - 0.9931285991850949,
        1.0 - 0.9639719272779138,
        1.0 - 0.9122344282513259,
        1.0 - 0.8391169718222188,
        1.0 - 0.7463319064601508,
        1.0 - 0.6360536807265150,
        1.0 - 0.5108670019508271,
        1.0 - 0.3737060887154196,
        1.0 - 0.2277858511416451,
        1.0 - 0.07652652113349733,
        1.0 + 0.9931285991850949,
        1.0 + 0.9639719272779138,
        1.0 + 0.9122344282513259,
        1.0 + 0.8391169718222188,
        1.0 + 0.7463319064601508,
        1.0 + 0.6360536807265150,
        1.0 + 0.5108670019508271,
        1.0 + 0.3737060887154196,
        1.0 + 0.2277858511416451,
        1.0 + 0.07652652113349733,
    ],
    dtype=np.float64,
)


@njit(cache=True, fastmath=True)
def _genz_tail_one(hh: float, kk: float, rr: float) -> float:
    if rr > 1.0:
        rr = 1.0
    elif rr < -1.0:
        rr = -1.0
    out = _ndtr(-hh) * _ndtr(-kk)
    if rr == 0.0:
        return out
    hs = 0.5 * (hh * hh + kk * kk)
    hk = hh * kk
    asr = 0.5 * math.asin(rr)
    abs_r = abs(rr)
    if abs_r < 0.3:
        w = _GL_W6
        x = _GL_X6
    elif abs_r < 0.75:
        w = _GL_W12
        x = _GL_X12
    else:
        w = _GL_W20
        x = _GL_X20
    integral = 0.0
    for j in range(w.shape[0]):
        sn = math.sin(asr * x[j])
        den = 1.0 - sn * sn
        if abs(den) < 1e-30:
            continue
        integral += math.exp((hk * sn - hs) / den) * w[j]
    out = out + integral * (asr / (2.0 * math.pi))
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


@njit(cache=True, fastmath=True)
def _bvn_tail(h: float, k: float, r: float) -> float:
    if h == math.inf or k == math.inf:
        return 0.0
    if h == -math.inf and k == -math.inf:
        return 1.0
    if h == -math.inf:
        return _ndtr(-k)
    if k == -math.inf:
        return _ndtr(-h)
    if not (math.isfinite(h) and math.isfinite(k) and math.isfinite(r)):
        return 0.0
    return _genz_tail_one(h, k, r)


@njit(cache=True, parallel=True, fastmath=True)
def bivariate_rect_cdf_numba(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Numba kernel for batched bivariate rectangular CDF (Genz path).

    :param mean: A np.array of shape ``[K, 2]``.
    :param cov: A np.array of shape ``[K, 2, 2]``.
    :param lower: A np.array of shape ``[K, 2]``.
    :param upper: A np.array of shape ``[K, 2]``.

    :returns: A np.array of shape ``[K,]``, the rectangle probabilities.
    """
    k = mean.shape[0]
    out = np.empty(k, dtype=np.float64)
    for i in prange(k):
        lo0 = lower[i, 0]
        lo1 = lower[i, 1]
        up0 = upper[i, 0]
        up1 = upper[i, 1]
        if (lo0 >= up0) or (lo1 >= up1):
            out[i] = 0.0
            continue
        s00 = cov[i, 0, 0]
        s11 = cov[i, 1, 1]
        s01 = cov[i, 0, 1]
        if s00 < 1e-30:
            s00 = 1e-30
        if s11 < 1e-30:
            s11 = 1e-30
        sig1 = math.sqrt(s00)
        sig2 = math.sqrt(s11)
        rho = s01 / (sig1 * sig2)
        if rho > 1.0:
            rho = 1.0
        elif rho < -1.0:
            rho = -1.0
        m0 = mean[i, 0]
        m1 = mean[i, 1]
        xl = -math.inf if not math.isfinite(lo0) else (lo0 - m0) / sig1
        xu = math.inf if not math.isfinite(up0) else (up0 - m0) / sig1
        yl = -math.inf if not math.isfinite(lo1) else (lo1 - m1) / sig2
        yu = math.inf if not math.isfinite(up1) else (up1 - m1) / sig2
        p = (
            _bvn_tail(xl, yl, rho)
            - _bvn_tail(xu, yl, rho)
            - _bvn_tail(xl, yu, rho)
            + _bvn_tail(xu, yu, rho)
        )
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        out[i] = p
    return out


@njit(cache=True, fastmath=True)
def _face_dens(
    x: float,
    a_o: float,
    b_o: float,
    cnn: float,
    ccross: float,
    coo: float,
    mass: float,
    eps: float,
) -> float:
    if (not math.isfinite(x)) or mass <= eps:
        return 0.0
    if cnn < eps:
        cnn = eps
    csd_n = math.sqrt(cnn)
    cmean = x * ccross / cnn
    cvar = coo - ccross * ccross / cnn
    if cvar < 0.0:
        cvar = 0.0
    csd = math.sqrt(cvar if cvar > eps else eps)
    cmass = _ndtr((b_o - cmean) / csd) - _ndtr((a_o - cmean) / csd)
    return (_norm_pdf(x / csd_n) / csd_n) * cmass / mass


@njit(cache=True, fastmath=True)
def _corner_dens(
    xq: float,
    xr: float,
    a0: float,
    a1: float,
    b0: float,
    b1: float,
    inv00: float,
    inv01: float,
    inv11: float,
    twopi_sdet: float,
    mass: float,
) -> float:
    if not (math.isfinite(xq) and math.isfinite(xr)):
        return 0.0
    if xq < a0 or xq > b0 or xr < a1 or xr > b1:
        return 0.0
    quad = inv00 * xq * xq + 2.0 * inv01 * xq * xr + inv11 * xr * xr
    return math.exp(-0.5 * quad) / twopi_sdet / mass


@njit(cache=True, parallel=True, fastmath=True)
def moments_2d_correlated_numba(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba kernel for correlated 2D rectangular truncated-Gaussian moments.

    :param mean: A np.array of shape ``[K, 2]``.
    :param cov: A np.array of shape ``[K, 2, 2]``.
    :param lower: A np.array of shape ``[K, 2]``.
    :param upper: A np.array of shape ``[K, 2]``.
    :param eps: A float, the numerical floor.

    :returns: A tuple ``(mass, mean_trunc, cov_trunc)``.
    """
    k = mean.shape[0]
    a = np.empty((k, 2), dtype=np.float64)
    b = np.empty((k, 2), dtype=np.float64)
    zero = np.zeros((k, 2), dtype=np.float64)
    for i in range(k):
        a[i, 0] = lower[i, 0] - mean[i, 0]
        a[i, 1] = lower[i, 1] - mean[i, 1]
        b[i, 0] = upper[i, 0] - mean[i, 0]
        b[i, 1] = upper[i, 1] - mean[i, 1]

    mass = bivariate_rect_cdf_numba(zero, cov, a, b)
    mean_t = np.empty((k, 2), dtype=np.float64)
    cov_t = np.empty((k, 2, 2), dtype=np.float64)

    for i in prange(k):
        m = mass[i]
        s00 = cov[i, 0, 0]
        s11 = cov[i, 1, 1]
        s01 = cov[i, 0, 1]
        a0 = a[i, 0]
        a1 = a[i, 1]
        b0 = b[i, 0]
        b1 = b[i, 1]

        if m <= eps:
            mean_t[i, 0] = mean[i, 0]
            mean_t[i, 1] = mean[i, 1]
            cov_t[i, 0, 0] = s00
            cov_t[i, 0, 1] = s01
            cov_t[i, 1, 0] = s01
            cov_t[i, 1, 1] = s11
            mass[i] = 0.0
            continue

        Fa0 = (
            _face_dens(a0, a1, b1, s00, s01, s11, m, eps)
            if math.isfinite(a0) and a0 <= b0
            else 0.0
        )
        Fb0 = (
            _face_dens(b0, a1, b1, s00, s01, s11, m, eps)
            if math.isfinite(b0) and b0 >= a0
            else 0.0
        )
        Fa1 = (
            _face_dens(a1, a0, b0, s11, s01, s00, m, eps)
            if math.isfinite(a1) and a1 <= b1
            else 0.0
        )
        Fb1 = (
            _face_dens(b1, a0, b0, s11, s01, s00, m, eps)
            if math.isfinite(b1) and b1 >= a1
            else 0.0
        )

        f0 = Fa0 - Fb0
        f1 = Fa1 - Fb1
        tm0 = s00 * f0 + s01 * f1
        tm1 = s01 * f0 + s11 * f1

        det = s00 * s11 - s01 * s01
        if det < eps:
            det = eps
        inv00 = s11 / det
        inv11 = s00 / det
        inv01 = -s01 / det
        twopi_sdet = 2.0 * math.pi * math.sqrt(det)

        d_aa = _corner_dens(a0, a1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, m)
        d_ba = _corner_dens(b0, a1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, m)
        d_ab = _corner_dens(a0, b1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, m)
        d_bb = _corner_dens(b0, b1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, m)
        F2_01 = (d_aa - d_ba) - (d_ab - d_bb)
        F2_10 = (d_aa - d_ab) - (d_ba - d_bb)

        e0 = (a0 * Fa0 if math.isfinite(a0) else 0.0) - (
            b0 * Fb0 if math.isfinite(b0) else 0.0
        )
        e1 = (a1 * Fa1 if math.isfinite(a1) else 0.0) - (
            b1 * Fb1 if math.isfinite(b1) else 0.0
        )
        inv_d0 = 1.0 / (s00 if s00 > eps else eps)
        inv_d1 = 1.0 / (s11 if s11 > eps else eps)

        t100 = s00 * s00 * inv_d0 * e0 + s01 * s01 * inv_d1 * e1
        t111 = s01 * s01 * inv_d0 * e0 + s11 * s11 * inv_d1 * e1
        t101 = s00 * s01 * inv_d0 * e0 + s01 * s11 * inv_d1 * e1

        # q=0, j=1: F2[0,0]=0, F2[0,1]=F2_01
        tt_10 = s01 - s00 * s01 * inv_d0
        tt_11 = s11 - s01 * s01 * inv_d0
        sum_s = tt_10 * 0.0 + tt_11 * F2_01
        t2_01 = s00 * sum_s
        t2_11 = s01 * sum_s

        # q=1, j=0: F2[1,0]=F2_10, F2[1,1]=0
        tt_00 = s00 - s01 * s01 * inv_d1
        tt_01 = s01 - s11 * s01 * inv_d1
        sum_s = tt_00 * F2_10 + tt_01 * 0.0
        t2_00 = s01 * sum_s
        t2_10 = s11 * sum_s

        exx00 = s00 + t100 + t2_00
        exx11 = s11 + t111 + t2_11
        exx01 = s01 + t101 + t2_01
        exx10 = s01 + t101 + t2_10

        mean_t[i, 0] = tm0 + mean[i, 0]
        mean_t[i, 1] = tm1 + mean[i, 1]
        cov_t[i, 0, 0] = exx00 - tm0 * tm0
        cov_t[i, 1, 1] = exx11 - tm1 * tm1
        c01 = 0.5 * ((exx01 - tm0 * tm1) + (exx10 - tm0 * tm1))
        cov_t[i, 0, 1] = c01
        cov_t[i, 1, 0] = c01

    return mass, mean_t, cov_t


# ---------------------------------------------------------------------------
# Gauß–Taylor linear map (fused, structured A / C_w — no dense n×n allocs)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _hat_tau_cv(x: float, v: float, x_pred: float, t_L: float) -> float:
    if abs(v) < 1e-15:
        v = 1e-15 if v >= 0.0 else -1e-15
    return (x_pred - x) / v + t_L


@njit(cache=True, fastmath=True)
def _hat_tau_ca(x: float, v: float, a: float, x_pred: float, t_L: float) -> float:
    delta_x = x_pred - x
    sqrt_val = v * v + 2.0 * a * delta_x
    if sqrt_val >= 0.0:
        sq = math.sqrt(sqrt_val)
    else:
        sq = v
    denom = abs(v) + sq
    if abs(denom) < 1e-15:
        denom = 1e-15
    sgn = 1.0 if v >= 0.0 else -1.0
    return sgn * (2.0 * delta_x) / denom + t_L


@njit(cache=True, parallel=True, fastmath=True)
def gauss_taylor_linear_map_cv2_numba(
    mu: np.ndarray,
    x_pred_to: np.ndarray,
    sw: float,
    t_L: float,
) -> tuple:
    """
    Numba Gauß–Taylor linear map for 1D CV (state_dim=2).

    :param mu: A np.array of shape ``[K, 2]``, the motion-state means.
    :param x_pred_to: A np.array of shape ``[K,]``, the streamwise actuator positions.
    :param sw: A float, the process-noise PSD.
    :param t_L: A float, the time of the last state.

    :returns: A tuple ``(H, d, R)`` of np.arrays of shapes ``[K, 2, 2]``, ``[K, 2]``,
        and ``[K, 2, 2]``.
    """
    k = mu.shape[0]
    H = np.zeros((k, 2, 2), dtype=np.float64)
    d = np.zeros((k, 2), dtype=np.float64)
    R = np.zeros((k, 2, 2), dtype=np.float64)
    for i in prange(k):
        tau = _hat_tau_cv(mu[i, 0], mu[i, 1], x_pred_to[i], t_L)
        dt = tau - t_L
        if (not math.isfinite(dt)) or dt < 0.0:
            dt = 0.0
        dt2 = dt * dt
        dt3 = dt2 * dt
        ex1 = mu[i, 1]
        if abs(ex1) < 1e-15:
            ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
        a = -1.0 / ex1
        H[i, 0, 0] = a
        H[i, 0, 1] = a * dt
        d[i, 0] = tau - (H[i, 0, 0] * mu[i, 0] + H[i, 0, 1] * mu[i, 1])
        c00 = sw * dt3 / 3.0
        R[i, 0, 0] = a * a * c00
    return H, d, R


@njit(cache=True, parallel=True, fastmath=True)
def gauss_taylor_linear_map_cv4_numba(
    mu: np.ndarray,
    x_pred_to: np.ndarray,
    swx: float,
    swy: float,
    t_L: float,
) -> tuple:
    """
    Numba Gauß–Taylor linear map for 2D CV (state_dim=4).

    :param mu: A np.array of shape ``[K, 4]``, the motion-state means.
    :param x_pred_to: A np.array of shape ``[K,]``, the streamwise actuator positions.
    :param swx: A float, the streamwise process-noise PSD.
    :param swy: A float, the lateral process-noise PSD.
    :param t_L: A float, the time of the last state.

    :returns: A tuple ``(H, d, R)`` of np.arrays of shapes ``[K, 2, 4]``, ``[K, 2]``,
        and ``[K, 2, 2]``.
    """
    k = mu.shape[0]
    H = np.empty((k, 2, 4), dtype=np.float64)
    d = np.empty((k, 2), dtype=np.float64)
    R = np.empty((k, 2, 2), dtype=np.float64)
    for i in prange(k):
        tau = _hat_tau_cv(mu[i, 0], mu[i, 1], x_pred_to[i], t_L)
        dt = tau - t_L
        if (not math.isfinite(dt)) or dt < 0.0:
            dt = 0.0
        dt2 = dt * dt
        dt3 = dt2 * dt
        ex1 = mu[i, 1]
        ex2 = mu[i, 2] + dt * mu[i, 3]
        ex3 = mu[i, 3]
        if abs(ex1) < 1e-15:
            ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
        gx0 = -1.0 / ex1
        H[i, 0, 0] = gx0
        H[i, 0, 1] = gx0 * dt
        H[i, 0, 2] = 0.0
        H[i, 0, 3] = 0.0
        gy0 = ex3 * gx0
        H[i, 1, 0] = gy0
        H[i, 1, 1] = gy0 * dt
        H[i, 1, 2] = 1.0
        H[i, 1, 3] = dt
        d[i, 0] = tau - (
            H[i, 0, 0] * mu[i, 0]
            + H[i, 0, 1] * mu[i, 1]
            + H[i, 0, 2] * mu[i, 2]
            + H[i, 0, 3] * mu[i, 3]
        )
        d[i, 1] = ex2 - (
            H[i, 1, 0] * mu[i, 0]
            + H[i, 1, 1] * mu[i, 1]
            + H[i, 1, 2] * mu[i, 2]
            + H[i, 1, 3] * mu[i, 3]
        )
        c00 = swx * dt3 / 3.0
        c22 = swy * dt3 / 3.0
        r00 = gx0 * gx0 * c00
        r01 = gx0 * gy0 * c00
        r11 = gy0 * gy0 * c00 + c22
        R[i, 0, 0] = r00
        R[i, 0, 1] = r01
        R[i, 1, 0] = r01
        R[i, 1, 1] = r11
    return H, d, R


@njit(cache=True, parallel=True, fastmath=True)
def gauss_taylor_linear_map_ca3_numba(
    mu: np.ndarray,
    x_pred_to: np.ndarray,
    sw: float,
    t_L: float,
) -> tuple:
    """
    Numba Gauß–Taylor linear map for 1D CA (state_dim=3).

    :param mu: A np.array of shape ``[K, 3]``, the motion-state means.
    :param x_pred_to: A np.array of shape ``[K,]``, the streamwise actuator positions.
    :param sw: A float, the process-noise PSD.
    :param t_L: A float, the time of the last state.

    :returns: A tuple ``(H, d, R)`` of np.arrays of shapes ``[K, 2, 3]``, ``[K, 2]``,
        and ``[K, 2, 2]``.
    """
    k = mu.shape[0]
    H = np.zeros((k, 2, 3), dtype=np.float64)
    d = np.zeros((k, 2), dtype=np.float64)
    R = np.zeros((k, 2, 2), dtype=np.float64)
    for i in prange(k):
        tau = _hat_tau_ca(mu[i, 0], mu[i, 1], mu[i, 2], x_pred_to[i], t_L)
        dt = tau - t_L
        if (not math.isfinite(dt)) or dt < 0.0:
            dt = 0.0
        dt2 = dt * dt
        dt5 = dt2 * dt2 * dt
        ex1 = mu[i, 1] + dt * mu[i, 2]
        if abs(ex1) < 1e-15:
            ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
        gx0 = -1.0 / ex1
        H[i, 0, 0] = gx0
        H[i, 0, 1] = gx0 * dt
        H[i, 0, 2] = gx0 * (0.5 * dt2)
        d[i, 0] = tau - (
            H[i, 0, 0] * mu[i, 0] + H[i, 0, 1] * mu[i, 1] + H[i, 0, 2] * mu[i, 2]
        )
        c00 = sw * dt5 / 20.0
        R[i, 0, 0] = gx0 * gx0 * c00
    return H, d, R


@njit(cache=True, parallel=True, fastmath=True)
def gauss_taylor_linear_map_ca6_numba(
    mu: np.ndarray,
    x_pred_to: np.ndarray,
    swx: float,
    swy: float,
    t_L: float,
) -> tuple:
    """
    Numba Gauß–Taylor linear map for 2D CA (state_dim=6).

    :param mu: A np.array of shape ``[K, 6]``, the motion-state means.
    :param x_pred_to: A np.array of shape ``[K,]``, the streamwise actuator positions.
    :param swx: A float, the streamwise process-noise PSD.
    :param swy: A float, the lateral process-noise PSD.
    :param t_L: A float, the time of the last state.

    :returns: A tuple ``(H, d, R)`` of np.arrays of shapes ``[K, 2, 6]``, ``[K, 2]``,
        and ``[K, 2, 2]``.
    """
    k = mu.shape[0]
    H = np.empty((k, 2, 6), dtype=np.float64)
    d = np.empty((k, 2), dtype=np.float64)
    R = np.empty((k, 2, 2), dtype=np.float64)
    for i in prange(k):
        tau = _hat_tau_ca(mu[i, 0], mu[i, 1], mu[i, 2], x_pred_to[i], t_L)
        dt = tau - t_L
        if (not math.isfinite(dt)) or dt < 0.0:
            dt = 0.0
        dt2 = dt * dt
        dt5 = dt2 * dt2 * dt
        ex1 = mu[i, 1] + dt * mu[i, 2]
        ex3 = mu[i, 3] + dt * mu[i, 4] + 0.5 * dt2 * mu[i, 5]
        ex4 = mu[i, 4] + dt * mu[i, 5]
        if abs(ex1) < 1e-15:
            ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
        gx0 = -1.0 / ex1
        gy0 = ex4 * gx0
        H[i, 0, 0] = gx0
        H[i, 0, 1] = gx0 * dt
        H[i, 0, 2] = gx0 * (0.5 * dt2)
        H[i, 0, 3] = 0.0
        H[i, 0, 4] = 0.0
        H[i, 0, 5] = 0.0
        H[i, 1, 0] = gy0
        H[i, 1, 1] = gy0 * dt
        H[i, 1, 2] = gy0 * (0.5 * dt2)
        H[i, 1, 3] = 1.0
        H[i, 1, 4] = dt
        H[i, 1, 5] = 0.5 * dt2
        hxmu = 0.0
        hymu = 0.0
        for j in range(6):
            hxmu += H[i, 0, j] * mu[i, j]
            hymu += H[i, 1, j] * mu[i, j]
        d[i, 0] = tau - hxmu
        d[i, 1] = ex3 - hymu
        c00 = swx * dt5 / 20.0
        c33 = swy * dt5 / 20.0
        r00 = gx0 * gx0 * c00
        r01 = gx0 * gy0 * c00
        r11 = gy0 * gy0 * c00 + c33
        R[i, 0, 0] = r00
        R[i, 0, 1] = r01
        R[i, 1, 0] = r01
        R[i, 1, 1] = r11
    return H, d, R


# ---------------------------------------------------------------------------
# Discrete-stage LGSSM predict_motion (fixed A, C_w)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True, fastmath=True)
def predict_motion_numba(
    mu: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    C_w: np.ndarray,
    apply_walls: bool,
    y_lo: float,
    y_hi: float,
    y_ind: int,
) -> tuple:
    """
    Batched LGSSM motion prediction with optional y-wall reflection.

    :param mu: A np.array of shape ``[K, n]``, the motion-state means.
    :param P: A np.array of shape ``[K, n, n]``, the motion-state covariances.
    :param A: A np.array of shape ``[n, n]``, the shared state-transition matrix.
    :param C_w: A np.array of shape ``[n, n]``, the shared process-noise covariance.
    :param apply_walls: A Boolean, if True reflect the mean lateral position into ``[y_lo, y_hi]``.
    :param y_lo: A float, the lower lateral wall.
    :param y_hi: A float, the upper lateral wall.
    :param y_ind: An integer, the index of the lateral position (ignored if not ``apply_walls``).

    :returns: A tuple ``(mu_n, P_n)`` of np.arrays with the same shapes as ``mu`` / ``P``.
    """
    k = mu.shape[0]
    n = mu.shape[1]
    mu_n = np.empty((k, n), dtype=np.float64)
    P_n = np.empty((k, n, n), dtype=np.float64)

    for i in prange(k):
        for r in range(n):
            s = 0.0
            for c in range(n):
                s += A[r, c] * mu[i, c]
            mu_n[i, r] = s

        tmp = np.empty((n, n), dtype=np.float64)
        for r in range(n):
            for c in range(n):
                s = 0.0
                for t in range(n):
                    s += A[r, t] * P[i, t, c]
                tmp[r, c] = s
        for r in range(n):
            for c in range(n):
                s = C_w[r, c]
                for t in range(n):
                    s += tmp[r, t] * A[c, t]
                P_n[i, r, c] = s

        if apply_walls:
            y = mu_n[i, y_ind]
            if y > y_hi:
                mu_n[i, y_ind] = 2.0 * y_hi - y
            elif y < y_lo:
                mu_n[i, y_ind] = 2.0 * y_lo - y

    return mu_n, P_n


# ---------------------------------------------------------------------------
# ActorModelVariantB continuous / inverse dynamics
# ---------------------------------------------------------------------------

@njit(cache=True)
def _valid_u_act_scalar(
    t_prev: float,
    u_act: float,
    T: float,
    t_cycle: float,
    t_lead: float,
) -> bool:
    """``c(k) = u - t_lead`` valid for activation in the current stage."""
    c = u_act - t_lead
    if c < 0.0 or c >= T:
        return False
    if t_prev > 0.0 and c < t_cycle - t_prev:
        return False
    return True


@njit(cache=True)
def _continuous_actor_dynamics_scalar(
    t: float,
    t_prev: float,
    u_act: float,
    T: float,
    t_cycle: float,
    t_lead: float,
) -> float:
    """Scalar ``t^act = f(t, t_prev, u)`` (Variant B)."""
    out = 0.0
    if t_prev > 0.0 and (t + t_prev) < t_cycle:
        out = t + t_prev

    if _valid_u_act_scalar(t_prev, u_act, T, t_cycle, t_lead):
        c = u_act - t_lead
        if t >= c:
            out = t - c
            # Edge case T == t_cycle: wrap finished cycle back to 0
            if T == t_cycle and out == t_cycle:
                out = 0.0
    return out


@njit(cache=True)
def _inverse_actor_dynamics_scalar(
    t_target: float,
    t_prev: float,
    u_act: float,
    T: float,
    t_cycle: float,
    t_lead: float,
) -> float:
    """Scalar ``t = g(t_target, t_prev, u)``; ``inf`` if unreachable."""
    out = np.inf
    if t_prev > 0.0 and (t_target - T) < t_prev and t_prev <= t_target:
        out = t_target - t_prev

    if _valid_u_act_scalar(t_prev, u_act, T, t_cycle, t_lead):
        c = u_act - t_lead
        if c + t_target < T:
            out = c + t_target
    return out


@njit(cache=True)
def _status_interval_scalar(
    t_status_begin: float,
    duration: float,
    t_prev: float,
    u_act: float,
    T: float,
    t_cycle: float,
    t_lead: float,
) -> tuple[float, float]:
    """Begin/end of one actor status within ``[0, T]`` (or ``inf``)."""
    t0 = _inverse_actor_dynamics_scalar(
        t_status_begin, t_prev, u_act, T, t_cycle, t_lead
    )
    # Avoid math.isfinite under fastmath (can treat +inf as finite).
    if t0 < np.inf:
        t1 = t0 + duration
        if t1 > T:
            t1 = T
    else:
        t1 = np.inf

    if t_status_begin <= t_prev < (t_status_begin + duration):
        t0 = 0.0
        t1 = t_status_begin + duration - t_prev
    return t0, t1


@njit(cache=True, parallel=True)
def predict_actor_state_numba(
    t_act_prev: np.ndarray,
    u_act: np.ndarray,
    t: float,
    T: float,
    t_cycle: float,
    t_activate: float,
    t_up: float,
    t_hit: float,
) -> np.ndarray:
    """
    Predicts actor internal times over one stage (Numba).

    :param t_act_prev: A np.array of previous actor internal times.
    :param u_act: A np.array of matching shape, hitting-time controls.
    :param t: A float, the absolute time / stage end used by the continuous dynamics.
    :param T: A float, the stage length.
    :param t_cycle: A float, the actor cycle time.
    :param t_activate: A float, activation duration.
    :param t_up: A float, UP duration.
    :param t_hit: A float, HIT duration.

    :returns: A np.array of the same shape as ``t_act_prev``, the predicted actor times.
    """
    k = t_act_prev.shape[0]
    t_lead = t_activate + t_up + 0.5 * t_hit
    out = np.empty(k, dtype=np.float64)
    for i in prange(k):
        out[i] = _continuous_actor_dynamics_scalar(
            t, t_act_prev[i], u_act[i], T, t_cycle, t_lead
        )
    return out


@njit(cache=True, parallel=True)
def actor_status_interval_numba(
    t_status_begin: float,
    duration: float,
    t_act_prev: np.ndarray,
    u_act: np.ndarray,
    T: float,
    t_cycle: float,
    t_activate: float,
    t_up: float,
    t_hit: float,
) -> np.ndarray:
    """
    Computes begin/end times of a named actuator status within a stage (Numba).

    :param t_status_begin: A float, status begin offset within the cycle.
    :param duration: A float, status duration.
    :param t_act_prev: A np.array, previous actor times.
    :param u_act: A np.array, hitting-time controls.
    :param T: A float, the stage length.
    :param t_cycle: A float, the actor cycle time.
    :param t_activate: A float, activation duration.
    :param t_up: A float, UP duration.
    :param t_hit: A float, HIT duration.

    :returns: A np.array of shape ``[..., 2]``, status begin/end times (``inf`` if absent).
    """
    k = t_act_prev.shape[0]
    t_lead = t_activate + t_up + 0.5 * t_hit
    out = np.empty((k, 2), dtype=np.float64)
    for i in prange(k):
        t0, t1 = _status_interval_scalar(
            t_status_begin,
            duration,
            t_act_prev[i],
            u_act[i],
            T,
            t_cycle,
            t_lead,
        )
        out[i, 0] = t0
        out[i, 1] = t1
    return out


@njit(cache=True, parallel=True)
def actor_phase_intervals_numba(
    t_act_prev: np.ndarray,
    u_act: np.ndarray,
    T: float,
    t_cycle: float,
    t_activate: float,
    t_up: float,
    t_hit: float,
    t_down: float,
) -> tuple:
    """
    Computes HIT/UP/DOWN begin/end times for a batch of actors (Numba).

    :param t_act_prev: A np.array, previous actor times.
    :param u_act: A np.array, hitting-time controls.
    :param T: A float, the stage length.
    :param t_cycle: A float, the actor cycle time.
    :param t_activate: A float, activation duration.
    :param t_up: A float, UP duration.
    :param t_hit: A float, HIT duration.
    :param t_down: A float, DOWN duration.

    :returns: A tuple ``(t_hit, t_up, t_down)``, each of shape ``[..., 2]``.
    """
    k = t_act_prev.shape[0]
    t_lead = t_activate + t_up + 0.5 * t_hit
    hit_begin = t_activate + t_up
    up_begin = t_activate
    down_begin = t_activate + t_up + t_hit

    t_hit_out = np.empty((k, 2), dtype=np.float64)
    t_up_out = np.empty((k, 2), dtype=np.float64)
    t_down_out = np.empty((k, 2), dtype=np.float64)

    for i in prange(k):
        tp = t_act_prev[i]
        ua = u_act[i]
        h0, h1 = _status_interval_scalar(
            hit_begin, t_hit, tp, ua, T, t_cycle, t_lead
        )
        u0, u1 = _status_interval_scalar(
            up_begin, t_up, tp, ua, T, t_cycle, t_lead
        )
        d0, d1 = _status_interval_scalar(
            down_begin, t_down, tp, ua, T, t_cycle, t_lead
        )
        t_hit_out[i, 0] = h0
        t_hit_out[i, 1] = h1
        t_up_out[i, 0] = u0
        t_up_out[i, 1] = u1
        t_down_out[i, 0] = d0
        t_down_out[i, 1] = d1
    return t_hit_out, t_up_out, t_down_out


@njit(cache=True, fastmath=True)
def precompute_horizon_phase_geometry_numba(
    t_act0_shared: np.ndarray,
    u_lead: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    T_step: float,
    t_cycle: float,
    t_activate: float,
    t_up: float,
    t_hit: float,
    t_down: float,
) -> tuple:
    """
    Precomputes actor times and phase rectangles over an OLF horizon (Numba).

    :param t_act0_shared: A np.array of shape ``[Na,]``, shared initial actor times.
    :param u_lead: A np.array of shape ``[B, Na, N]``, controls including lead time.
    :param y_lo: A np.array of shape ``[Na,]``, lower lateral windows.
    :param y_hi: A np.array of shape ``[Na,]``, upper lateral windows.
    :param T_step: A float, the stage length.
    :param t_cycle: A float, the actor cycle time.
    :param t_activate: A float, activation duration.
    :param t_up: A float, UP duration.
    :param t_hit: A float, HIT duration.
    :param t_down: A float, DOWN duration.

    :returns: A tuple ``(t_act_seq, lower, upper, phase_active, actuator_active)``
        with horizon-leading shapes matching the NumPy precompute helper.
    """
    B = u_lead.shape[0]
    Na = u_lead.shape[1]
    T = u_lead.shape[2]
    t_lead = t_activate + t_up + 0.5 * t_hit
    hit_begin = t_activate + t_up
    up_begin = t_activate
    down_begin = t_activate + t_up + t_hit

    t_act_seq = np.empty((T + 1, B, Na), dtype=np.float64)
    lower = np.empty((T, B, Na, 3, 2), dtype=np.float64)
    upper = np.empty((T, B, Na, 3, 2), dtype=np.float64)
    phase_active = np.empty((T, B, Na, 3), dtype=np.bool_)
    actuator_active = np.empty((T, B, Na), dtype=np.bool_)

    for b in range(B):
        t_local = np.empty(Na, dtype=np.float64)
        for a in range(Na):
            t_local[a] = t_act0_shared[a]
            t_act_seq[0, b, a] = t_act0_shared[a]

        for t in range(T):
            for a in range(Na):
                tp = t_local[a]
                ua = u_lead[b, a, t]

                h0, h1 = _status_interval_scalar(
                    hit_begin, t_hit, tp, ua, T_step, t_cycle, t_lead
                )
                u0, u1 = _status_interval_scalar(
                    up_begin, t_up, tp, ua, T_step, t_cycle, t_lead
                )
                d0, d1 = _status_interval_scalar(
                    down_begin, t_down, tp, ua, T_step, t_cycle, t_lead
                )

                # no scheduled activation → all phases inactive (u typically +inf)
                no_act = (tp == 0.0) and (not (ua < 1.0e300))

                ph0 = (h0 < np.inf) and (h1 < np.inf) and (h1 > h0) and (not no_act)
                ph1 = (u0 < np.inf) and (u1 < np.inf) and (u1 > u0) and (not no_act)
                ph2 = (d0 < np.inf) and (d1 < np.inf) and (d1 > d0) and (not no_act)

                phase_active[t, b, a, 0] = ph0
                phase_active[t, b, a, 1] = ph1
                phase_active[t, b, a, 2] = ph2
                actuator_active[t, b, a] = ph0 or ph1 or ph2

                lower[t, b, a, 0, 0] = h0
                lower[t, b, a, 0, 1] = y_lo[a]
                upper[t, b, a, 0, 0] = h1
                upper[t, b, a, 0, 1] = y_hi[a]

                lower[t, b, a, 1, 0] = u0
                lower[t, b, a, 1, 1] = y_lo[a]
                upper[t, b, a, 1, 0] = u1
                upper[t, b, a, 1, 1] = y_hi[a]

                lower[t, b, a, 2, 0] = d0
                lower[t, b, a, 2, 1] = y_lo[a]
                upper[t, b, a, 2, 0] = d1
                upper[t, b, a, 2, 1] = y_hi[a]

                t_next = _continuous_actor_dynamics_scalar(
                    T_step, tp, ua, T_step, t_cycle, t_lead
                )
                t_local[a] = t_next
                t_act_seq[t + 1, b, a] = t_next

    return t_act_seq, lower, upper, phase_active, actuator_active


# ---------------------------------------------------------------------------
# Fused phased ADF contact matching (forward + truncate + backward + compose)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _bvn_rect_cdf_one(
    m0: float,
    m1: float,
    s00: float,
    s01: float,
    s11: float,
    lo0: float,
    lo1: float,
    up0: float,
    up1: float,
) -> float:
    if (lo0 >= up0) or (lo1 >= up1):
        return 0.0
    if s00 < 1e-30:
        s00 = 1e-30
    if s11 < 1e-30:
        s11 = 1e-30
    sig1 = math.sqrt(s00)
    sig2 = math.sqrt(s11)
    rho = s01 / (sig1 * sig2)
    if rho > 1.0:
        rho = 1.0
    elif rho < -1.0:
        rho = -1.0
    xl = -math.inf if not math.isfinite(lo0) else (lo0 - m0) / sig1
    xu = math.inf if not math.isfinite(up0) else (up0 - m0) / sig1
    yl = -math.inf if not math.isfinite(lo1) else (lo1 - m1) / sig2
    yu = math.inf if not math.isfinite(up1) else (up1 - m1) / sig2
    p = (
        _bvn_tail(xl, yl, rho)
        - _bvn_tail(xu, yl, rho)
        - _bvn_tail(xl, yu, rho)
        + _bvn_tail(xu, yu, rho)
    )
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


@njit(cache=True)
def _moments_2d_rect_one(
    m0: float,
    m1: float,
    s00: float,
    s01: float,
    s11: float,
    lo0: float,
    lo1: float,
    up0: float,
    up1: float,
    eps: float,
) -> tuple[float, float, float, float, float, float]:
    """Return mass, mean0/1, cov00/01/11 of a 2D Gaussian truncated to a rectangle."""
    if abs(s01) <= 1e-12 * math.sqrt(max(s00 * s11, 0.0)):
        mass0, mu0t, v0 = _trunc1d(m0, s00, lo0, up0, eps)
        mass1, mu1t, v1 = _trunc1d(m1, s11, lo1, up1, eps)
        return mass0 * mass1, mu0t, mu1t, v0, 0.0, v1

    a0 = lo0 - m0
    a1 = lo1 - m1
    b0 = up0 - m0
    b1 = up1 - m1
    mass = _bvn_rect_cdf_one(0.0, 0.0, s00, s01, s11, a0, a1, b0, b1)
    if mass <= eps:
        return 0.0, m0, m1, s00, s01, s11

    Fa0 = (
        _face_dens(a0, a1, b1, s00, s01, s11, mass, eps)
        if math.isfinite(a0) and a0 <= b0
        else 0.0
    )
    Fb0 = (
        _face_dens(b0, a1, b1, s00, s01, s11, mass, eps)
        if math.isfinite(b0) and b0 >= a0
        else 0.0
    )
    Fa1 = (
        _face_dens(a1, a0, b0, s11, s01, s00, mass, eps)
        if math.isfinite(a1) and a1 <= b1
        else 0.0
    )
    Fb1 = (
        _face_dens(b1, a0, b0, s11, s01, s00, mass, eps)
        if math.isfinite(b1) and b1 >= a1
        else 0.0
    )

    f0 = Fa0 - Fb0
    f1 = Fa1 - Fb1
    tm0 = s00 * f0 + s01 * f1
    tm1 = s01 * f0 + s11 * f1

    det = s00 * s11 - s01 * s01
    if det < eps:
        det = eps
    inv00 = s11 / det
    inv11 = s00 / det
    inv01 = -s01 / det
    twopi_sdet = 2.0 * math.pi * math.sqrt(det)

    d_aa = _corner_dens(a0, a1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, mass)
    d_ba = _corner_dens(b0, a1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, mass)
    d_ab = _corner_dens(a0, b1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, mass)
    d_bb = _corner_dens(b0, b1, a0, a1, b0, b1, inv00, inv01, inv11, twopi_sdet, mass)
    F2_01 = (d_aa - d_ba) - (d_ab - d_bb)
    F2_10 = (d_aa - d_ab) - (d_ba - d_bb)

    e0 = (a0 * Fa0 if math.isfinite(a0) else 0.0) - (
        b0 * Fb0 if math.isfinite(b0) else 0.0
    )
    e1 = (a1 * Fa1 if math.isfinite(a1) else 0.0) - (
        b1 * Fb1 if math.isfinite(b1) else 0.0
    )
    inv_d0 = 1.0 / (s00 if s00 > eps else eps)
    inv_d1 = 1.0 / (s11 if s11 > eps else eps)

    t100 = s00 * s00 * inv_d0 * e0 + s01 * s01 * inv_d1 * e1
    t111 = s01 * s01 * inv_d0 * e0 + s11 * s11 * inv_d1 * e1
    t101 = s00 * s01 * inv_d0 * e0 + s01 * s11 * inv_d1 * e1

    tt_11 = s11 - s01 * s01 * inv_d0
    sum_s = tt_11 * F2_01
    t2_01 = s00 * sum_s
    t2_11 = s01 * sum_s

    tt_00 = s00 - s01 * s01 * inv_d1
    sum_s = tt_00 * F2_10
    t2_00 = s01 * sum_s
    t2_10 = s11 * sum_s

    exx00 = s00 + t100 + t2_00
    exx11 = s11 + t111 + t2_11
    exx01 = s01 + t101 + t2_01
    exx10 = s01 + t101 + t2_10

    c01 = 0.5 * ((exx01 - tm0 * tm1) + (exx10 - tm0 * tm1))
    return (
        mass,
        tm0 + m0,
        tm1 + m1,
        exx00 - tm0 * tm0,
        c01,
        exx11 - tm1 * tm1,
    )


@njit(cache=True, parallel=True)
def adf_phased_contact_match_numba(
    pi0: np.ndarray,
    mu0: np.ndarray,
    P0: np.ndarray,
    pi1: np.ndarray,
    mu1: np.ndarray,
    P1: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    R: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    phase_active: np.ndarray,
    p_phase: np.ndarray,
    eps: float = 1e-15,
) -> tuple:
    """
    Numba phased ADF contact moment matching for one stage.

    :param pi0: Dead-mode prior mass array.
    :param mu0: Dead-mode prior means.
    :param P0: Dead-mode prior covariances.
    :param pi1: Living-mode prior mass array.
    :param mu1: Living-mode prior means.
    :param P1: Living-mode prior covariances.
    :param H: Actuator Gauß–Taylor Jacobians.
    :param d: Actuator Gauß–Taylor offsets.
    :param R: Actuator measurement-noise covariances.
    :param lower: Phase-rectangle lower corners.
    :param upper: Phase-rectangle upper corners.
    :param phase_active: Boolean phase-activity mask.
    :param p_phase: Per-particle HIT/UP/DOWN ejection probabilities.
    :param eps: A float, numerical floor.

    :returns: Packed ADF dual-mode posterior arrays (masses, means, covariances)
        matching :func:`adf.adf_contact_model.adf_contact_moment_matching`.
    """
    B = pi0.shape[0]
    N = pi0.shape[1]
    Na = H.shape[2]
    n = mu1.shape[2]
    n_phases = 3

    pi_out = np.empty((B, N, 2), dtype=np.float64)
    mu_out = np.empty((B, N, 2, n), dtype=np.float64)
    P_out = np.empty((B, N, 2, n, n), dtype=np.float64)

    for bn in prange(B * N):
        b = bn // N
        p = bn - b * N

        p0 = pi0[b, p]
        p1 = pi1[b, p]
        M0_ej = 0.0
        M1_ej = np.zeros(n, dtype=np.float64)
        M2_ej = np.zeros((n, n), dtype=np.float64)

        for a in range(Na):
            for ph in range(n_phases):
                if not phase_active[b, a, ph]:
                    continue
                pe = p_phase[p, ph]
                if pe <= 0.0:
                    continue

                lo0 = lower[b, a, ph, 0]
                lo1 = lower[b, a, ph, 1]
                up0 = upper[b, a, ph, 0]
                up1 = upper[b, a, ph, 1]
                if (not math.isfinite(lo0)) or (not math.isfinite(up0)) or up0 <= lo0:
                    continue

                mu_xi0 = d[b, p, a, 0]
                mu_xi1 = d[b, p, a, 1]
                for c in range(n):
                    mu_xi0 += H[b, p, a, 0, c] * mu1[b, p, c]
                    mu_xi1 += H[b, p, a, 1, c] * mu1[b, p, c]

                pxi00 = R[b, p, a, 0, 0]
                pxi01 = R[b, p, a, 0, 1]
                pxi10 = R[b, p, a, 1, 0]
                pxi11 = R[b, p, a, 1, 1]
                for c in range(n):
                    t0 = 0.0
                    t1 = 0.0
                    for t in range(n):
                        t0 += H[b, p, a, 0, t] * P1[b, p, t, c]
                        t1 += H[b, p, a, 1, t] * P1[b, p, t, c]
                    pxi00 += t0 * H[b, p, a, 0, c]
                    pxi01 += t0 * H[b, p, a, 1, c]
                    pxi10 += t1 * H[b, p, a, 0, c]
                    pxi11 += t1 * H[b, p, a, 1, c]

                det = pxi00 * pxi11 - pxi01 * pxi10
                if abs(det) < 1e-30:
                    det = 1e-30 if det >= 0.0 else -1e-30
                invT00 = pxi11 / det
                invT01 = -pxi10 / det
                invT10 = -pxi01 / det
                invT11 = pxi00 / det

                K = np.empty((n, 2), dtype=np.float64)
                for r in range(n):
                    ht0 = 0.0
                    ht1 = 0.0
                    for t in range(n):
                        ht0 += P1[b, p, r, t] * H[b, p, a, 0, t]
                        ht1 += P1[b, p, r, t] * H[b, p, a, 1, t]
                    K[r, 0] = ht0 * invT00 + ht1 * invT10
                    K[r, 1] = ht0 * invT01 + ht1 * invT11

                # Use P_ξ[0,1] (not symmetrized) to match forward_project + moments path.
                w, my0, my1, py00, py01, py11 = _moments_2d_rect_one(
                    mu_xi0,
                    mu_xi1,
                    pxi00,
                    pxi01,
                    pxi11,
                    lo0,
                    lo1,
                    up0,
                    up1,
                    eps,
                )
                if w <= eps:
                    continue

                d0 = my0 - mu_xi0
                d1 = my1 - mu_xi1
                mu_ms = np.empty(n, dtype=np.float64)
                for r in range(n):
                    mu_ms[r] = mu1[b, p, r] + K[r, 0] * d0 + K[r, 1] * d1

                dp00 = py00 - pxi00
                dp11 = py11 - pxi11
                dp01 = py01 - pxi01
                dp10 = py01 - pxi10

                P_ms = np.empty((n, n), dtype=np.float64)
                for r in range(n):
                    t0 = K[r, 0] * dp00 + K[r, 1] * dp10
                    t1 = K[r, 0] * dp01 + K[r, 1] * dp11
                    for c in range(n):
                        P_ms[r, c] = P1[b, p, r, c] + t0 * K[c, 0] + t1 * K[c, 1]
                for r in range(n):
                    for c in range(r + 1, n):
                        s = 0.5 * (P_ms[r, c] + P_ms[c, r])
                        P_ms[r, c] = s
                        P_ms[c, r] = s

                wt = p1 * pe * w
                M0_ej += wt
                for r in range(n):
                    M1_ej[r] += wt * mu_ms[r]
                    for c in range(n):
                        M2_ej[r, c] += wt * (P_ms[r, c] + mu_ms[r] * mu_ms[c])

        M0_r0 = p0 + M0_ej
        M0_r1 = p1 - M0_ej
        pi_out[b, p, 0] = M0_r0 if M0_r0 > eps else 0.0
        pi_out[b, p, 1] = M0_r1 if M0_r1 > eps else 0.0

        for r in range(n):
            m1r0 = p0 * mu0[b, p, r] + M1_ej[r]
            m1r1 = p1 * mu1[b, p, r] - M1_ej[r]
            if M0_r0 > eps:
                mu_out[b, p, 0, r] = m1r0 / M0_r0
            else:
                mu_out[b, p, 0, r] = mu0[b, p, r]
            if M0_r1 > eps:
                mu_out[b, p, 1, r] = m1r1 / M0_r1
            else:
                mu_out[b, p, 1, r] = mu1[b, p, r]

        for r in range(n):
            for c in range(n):
                outer0 = mu0[b, p, r] * mu0[b, p, c]
                outer1 = mu1[b, p, r] * mu1[b, p, c]
                m2r0 = p0 * (P0[b, p, r, c] + outer0) + M2_ej[r, c]
                m2r1 = p1 * (P1[b, p, r, c] + outer1) - M2_ej[r, c]
                if M0_r0 > eps:
                    P_out[b, p, 0, r, c] = m2r0 / M0_r0 - (
                        mu_out[b, p, 0, r] * mu_out[b, p, 0, c]
                    )
                else:
                    P_out[b, p, 0, r, c] = P0[b, p, r, c]
                if M0_r1 > eps:
                    P_out[b, p, 1, r, c] = m2r1 / M0_r1 - (
                        mu_out[b, p, 1, r] * mu_out[b, p, 1, c]
                    )
                else:
                    P_out[b, p, 1, r, c] = P1[b, p, r, c]

        for mode in range(2):
            for r in range(n):
                for c in range(r + 1, n):
                    s = 0.5 * (P_out[b, p, mode, r, c] + P_out[b, p, mode, c, r])
                    P_out[b, p, mode, r, c] = s
                    P_out[b, p, mode, c, r] = s

    return pi_out, mu_out, P_out


# ---------------------------------------------------------------------------
# Fused OLF horizon (GT on-the-fly + contact + motion + terminal cost)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _gt_cv4_fill(
    m0: float,
    m1: float,
    m2: float,
    m3: float,
    x_pred: float,
    swx: float,
    swy: float,
    t_L: float,
    H0: np.ndarray,
    H1: np.ndarray,
    d_out: np.ndarray,
    R_out: np.ndarray,
) -> None:
    """Write 2×4 H, d(2), R(2×2) for 2D CV Gauß–Taylor at one mean."""
    tau = _hat_tau_cv(m0, m1, x_pred, t_L)
    dt = tau - t_L
    if (not math.isfinite(dt)) or dt < 0.0:
        dt = 0.0
    dt2 = dt * dt
    dt3 = dt2 * dt
    ex1 = m1
    if abs(ex1) < 1e-15:
        ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
    gx0 = -1.0 / ex1
    gy0 = m3 * gx0
    ex2 = m2 + dt * m3
    H0[0] = gx0
    H0[1] = gx0 * dt
    H0[2] = 0.0
    H0[3] = 0.0
    H1[0] = gy0
    H1[1] = gy0 * dt
    H1[2] = 1.0
    H1[3] = dt
    d_out[0] = tau - (H0[0] * m0 + H0[1] * m1 + H0[2] * m2 + H0[3] * m3)
    d_out[1] = ex2 - (H1[0] * m0 + H1[1] * m1 + H1[2] * m2 + H1[3] * m3)
    c00 = swx * dt3 / 3.0
    c22 = swy * dt3 / 3.0
    R_out[0, 0] = gx0 * gx0 * c00
    R_out[0, 1] = gx0 * gy0 * c00
    R_out[1, 0] = R_out[0, 1]
    R_out[1, 1] = gy0 * gy0 * c00 + c22


@njit(cache=True)
def _gt_cv2_fill(
    m0: float,
    m1: float,
    x_pred: float,
    sw: float,
    t_L: float,
    H0: np.ndarray,
    H1: np.ndarray,
    d_out: np.ndarray,
    R_out: np.ndarray,
) -> None:
    tau = _hat_tau_cv(m0, m1, x_pred, t_L)
    dt = tau - t_L
    if (not math.isfinite(dt)) or dt < 0.0:
        dt = 0.0
    dt3 = dt * dt * dt
    ex1 = m1
    if abs(ex1) < 1e-15:
        ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
    a = -1.0 / ex1
    H0[0] = a
    H0[1] = a * dt
    H1[0] = 0.0
    H1[1] = 0.0
    d_out[0] = tau - (H0[0] * m0 + H0[1] * m1)
    d_out[1] = 0.0
    c00 = sw * dt3 / 3.0
    R_out[0, 0] = a * a * c00
    R_out[0, 1] = 0.0
    R_out[1, 0] = 0.0
    R_out[1, 1] = 0.0



@njit(cache=True)
def _gt_ca3_fill(
    m0: float,
    m1: float,
    m2: float,
    x_pred: float,
    sw: float,
    t_L: float,
    H0: np.ndarray,
    H1: np.ndarray,
    d_out: np.ndarray,
    R_out: np.ndarray,
) -> None:
    """Write 2×3 H, d(2), R(2×2) for 1D CA Gauß–Taylor at one mean."""
    tau = _hat_tau_ca(m0, m1, m2, x_pred, t_L)
    dt = tau - t_L
    if (not math.isfinite(dt)) or dt < 0.0:
        dt = 0.0
    dt2 = dt * dt
    dt5 = dt2 * dt2 * dt
    ex1 = m1 + dt * m2
    if abs(ex1) < 1e-15:
        ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
    gx0 = -1.0 / ex1
    H0[0] = gx0
    H0[1] = gx0 * dt
    H0[2] = gx0 * (0.5 * dt2)
    H1[0] = 0.0
    H1[1] = 0.0
    H1[2] = 0.0
    d_out[0] = tau - (H0[0] * m0 + H0[1] * m1 + H0[2] * m2)
    d_out[1] = 0.0
    c00 = sw * dt5 / 20.0
    R_out[0, 0] = gx0 * gx0 * c00
    R_out[0, 1] = 0.0
    R_out[1, 0] = 0.0
    R_out[1, 1] = 0.0


@njit(cache=True)
def _gt_ca6_fill(
    m0: float,
    m1: float,
    m2: float,
    m3: float,
    m4: float,
    m5: float,
    x_pred: float,
    swx: float,
    swy: float,
    t_L: float,
    H0: np.ndarray,
    H1: np.ndarray,
    d_out: np.ndarray,
    R_out: np.ndarray,
) -> None:
    """Write 2×6 H, d(2), R(2×2) for 2D CA Gauß–Taylor at one mean."""
    tau = _hat_tau_ca(m0, m1, m2, x_pred, t_L)
    dt = tau - t_L
    if (not math.isfinite(dt)) or dt < 0.0:
        dt = 0.0
    dt2 = dt * dt
    dt5 = dt2 * dt2 * dt
    ex1 = m1 + dt * m2
    ex3 = m3 + dt * m4 + 0.5 * dt2 * m5
    ex4 = m4 + dt * m5
    if abs(ex1) < 1e-15:
        ex1 = 1e-15 if ex1 >= 0.0 else -1e-15
    gx0 = -1.0 / ex1
    gy0 = ex4 * gx0
    H0[0] = gx0
    H0[1] = gx0 * dt
    H0[2] = gx0 * (0.5 * dt2)
    H0[3] = 0.0
    H0[4] = 0.0
    H0[5] = 0.0
    H1[0] = gy0
    H1[1] = gy0 * dt
    H1[2] = gy0 * (0.5 * dt2)
    H1[3] = 1.0
    H1[4] = dt
    H1[5] = 0.5 * dt2
    hxmu = H0[0] * m0 + H0[1] * m1 + H0[2] * m2 + H0[3] * m3 + H0[4] * m4 + H0[5] * m5
    hymu = H1[0] * m0 + H1[1] * m1 + H1[2] * m2 + H1[3] * m3 + H1[4] * m4 + H1[5] * m5
    d_out[0] = tau - hxmu
    d_out[1] = ex3 - hymu
    c00 = swx * dt5 / 20.0
    c33 = swy * dt5 / 20.0
    R_out[0, 0] = gx0 * gx0 * c00
    R_out[0, 1] = gx0 * gy0 * c00
    R_out[1, 0] = R_out[0, 1]
    R_out[1, 1] = gy0 * gy0 * c00 + c33


@njit(cache=True)
def _predict_one(
    mu: np.ndarray,
    P: np.ndarray,
    A: np.ndarray,
    C_w: np.ndarray,
    apply_walls: bool,
    y_lo: float,
    y_hi: float,
    y_ind: int,
) -> None:
    """In-place μ' = Aμ, P' = APAᵀ + C_w (+ optional wall on mean)."""
    n = mu.shape[0]
    mu_n = np.empty(n, dtype=np.float64)
    for r in range(n):
        s = 0.0
        for c in range(n):
            s += A[r, c] * mu[c]
        mu_n[r] = s
    tmp = np.empty((n, n), dtype=np.float64)
    for r in range(n):
        for c in range(n):
            s = 0.0
            for t in range(n):
                s += A[r, t] * P[t, c]
            tmp[r, c] = s
    for r in range(n):
        for c in range(n):
            s = C_w[r, c]
            for t in range(n):
                s += tmp[r, t] * A[c, t]
            P[r, c] = s
    for r in range(n):
        mu[r] = mu_n[r]
    if apply_walls:
        y = mu[y_ind]
        if y > y_hi:
            mu[y_ind] = 2.0 * y_hi - y
        elif y < y_lo:
            mu[y_ind] = 2.0 * y_lo - y


@njit(cache=True)
def _contact_one_particle(
    p0: float,
    p1: float,
    mu0: np.ndarray,
    P0: np.ndarray,
    mu1: np.ndarray,
    P1: np.ndarray,
    lower_b: np.ndarray,
    upper_b: np.ndarray,
    phase_active_b: np.ndarray,
    p_phase_p: np.ndarray,
    actor_x: np.ndarray,
    swx: float,
    swy: float,
    state_dim: int,
    eps: float,
    H0: np.ndarray,
    H1: np.ndarray,
    d_gt: np.ndarray,
    R_gt: np.ndarray,
) -> tuple:
    """One particle, one stage: GT on-the-fly + match + compose.

    Returns updated ``(p0, p1, M0_ej)`` and writes mu0/P0/mu1/P1 in place.
    ``M0_ej`` is the ejected living mass ``π̃^{(1)} Σ p_eject w`` (step-cost term).
    """
    n = state_dim
    Na = actor_x.shape[0]
    M0_ej = 0.0
    M1_ej = np.zeros(n, dtype=np.float64)
    M2_ej = np.zeros((n, n), dtype=np.float64)
    K = np.empty((n, 2), dtype=np.float64)
    mu_ms = np.empty(n, dtype=np.float64)
    P_ms = np.empty((n, n), dtype=np.float64)

    for a in range(Na):
        any_ph = False
        for ph in range(3):
            if phase_active_b[a, ph]:
                any_ph = True
                break
        if not any_ph:
            continue

        # Gauß–Taylor at living mean vs actuator x (CV / CA)
        if n == 2:
            _gt_cv2_fill(
                mu1[0], mu1[1], actor_x[a], swx, 0.0, H0, H1, d_gt, R_gt,
            )
        elif n == 3:
            _gt_ca3_fill(
                mu1[0], mu1[1], mu1[2], actor_x[a], swx, 0.0, H0, H1, d_gt, R_gt,
            )
        elif n == 4:
            _gt_cv4_fill(
                mu1[0], mu1[1], mu1[2], mu1[3],
                actor_x[a], swx, swy, 0.0, H0, H1, d_gt, R_gt,
            )
        else:  # n == 6
            _gt_ca6_fill(
                mu1[0], mu1[1], mu1[2], mu1[3], mu1[4], mu1[5],
                actor_x[a], swx, swy, 0.0, H0, H1, d_gt, R_gt,
            )

        for ph in range(3):
            if not phase_active_b[a, ph]:
                continue
            pe = p_phase_p[ph]
            if pe <= 0.0:
                continue
            lo0 = lower_b[a, ph, 0]
            lo1 = lower_b[a, ph, 1]
            up0 = upper_b[a, ph, 0]
            up1 = upper_b[a, ph, 1]
            if (not math.isfinite(lo0)) or (not math.isfinite(up0)) or up0 <= lo0:
                continue

            mu_xi0 = d_gt[0]
            mu_xi1 = d_gt[1]
            for c in range(n):
                mu_xi0 += H0[c] * mu1[c]
                mu_xi1 += H1[c] * mu1[c]

            pxi00 = R_gt[0, 0]
            pxi01 = R_gt[0, 1]
            pxi10 = R_gt[1, 0]
            pxi11 = R_gt[1, 1]
            for c in range(n):
                t0 = 0.0
                t1 = 0.0
                for t in range(n):
                    t0 += H0[t] * P1[t, c]
                    t1 += H1[t] * P1[t, c]
                pxi00 += t0 * H0[c]
                pxi01 += t0 * H1[c]
                pxi10 += t1 * H0[c]
                pxi11 += t1 * H1[c]

            det = pxi00 * pxi11 - pxi01 * pxi10
            if abs(det) < 1e-30:
                det = 1e-30 if det >= 0.0 else -1e-30
            invT00 = pxi11 / det
            invT01 = -pxi10 / det
            invT10 = -pxi01 / det
            invT11 = pxi00 / det

            for r in range(n):
                ht0 = 0.0
                ht1 = 0.0
                for t in range(n):
                    ht0 += P1[r, t] * H0[t]
                    ht1 += P1[r, t] * H1[t]
                K[r, 0] = ht0 * invT00 + ht1 * invT10
                K[r, 1] = ht0 * invT01 + ht1 * invT11

            w, my0, my1, py00, py01, py11 = _moments_2d_rect_one(
                mu_xi0, mu_xi1, pxi00, pxi01, pxi11, lo0, lo1, up0, up1, eps,
            )
            if w <= eps:
                continue

            dd0 = my0 - mu_xi0
            dd1 = my1 - mu_xi1
            for r in range(n):
                mu_ms[r] = mu1[r] + K[r, 0] * dd0 + K[r, 1] * dd1

            dp00 = py00 - pxi00
            dp11 = py11 - pxi11
            dp01 = py01 - pxi01
            dp10 = py01 - pxi10
            for r in range(n):
                t0 = K[r, 0] * dp00 + K[r, 1] * dp10
                t1 = K[r, 0] * dp01 + K[r, 1] * dp11
                for c in range(n):
                    P_ms[r, c] = P1[r, c] + t0 * K[c, 0] + t1 * K[c, 1]
            for r in range(n):
                for c in range(r + 1, n):
                    s = 0.5 * (P_ms[r, c] + P_ms[c, r])
                    P_ms[r, c] = s
                    P_ms[c, r] = s

            wt = p1 * pe * w
            M0_ej += wt
            for r in range(n):
                M1_ej[r] += wt * mu_ms[r]
                for c in range(n):
                    M2_ej[r, c] += wt * (P_ms[r, c] + mu_ms[r] * mu_ms[c])

    M0_r0 = p0 + M0_ej
    M0_r1 = p1 - M0_ej
    p0n = M0_r0 if M0_r0 > eps else 0.0
    p1n = M0_r1 if M0_r1 > eps else 0.0

    mu0n = np.empty(n, dtype=np.float64)
    mu1n = np.empty(n, dtype=np.float64)
    for r in range(n):
        m1r0 = p0 * mu0[r] + M1_ej[r]
        m1r1 = p1 * mu1[r] - M1_ej[r]
        mu0n[r] = m1r0 / M0_r0 if M0_r0 > eps else mu0[r]
        mu1n[r] = m1r1 / M0_r1 if M0_r1 > eps else mu1[r]

    for r in range(n):
        for c in range(n):
            outer0 = mu0[r] * mu0[c]
            outer1 = mu1[r] * mu1[c]
            m2r0 = p0 * (P0[r, c] + outer0) + M2_ej[r, c]
            m2r1 = p1 * (P1[r, c] + outer1) - M2_ej[r, c]
            if M0_r0 > eps:
                P0[r, c] = m2r0 / M0_r0 - mu0n[r] * mu0n[c]
            # else keep P0
            if M0_r1 > eps:
                P1[r, c] = m2r1 / M0_r1 - mu1n[r] * mu1n[c]
    for r in range(n):
        mu0[r] = mu0n[r]
        mu1[r] = mu1n[r]
    for r in range(n):
        for c in range(r + 1, n):
            s = 0.5 * (P0[r, c] + P0[c, r])
            P0[r, c] = s
            P0[c, r] = s
            s = 0.5 * (P1[r, c] + P1[c, r])
            P1[r, c] = s
            P1[c, r] = s

    return p0n, p1n, M0_ej


@njit(cache=True, parallel=True)
def adf_olf_horizon_numba(
    pi_init: np.ndarray,
    mu_init: np.ndarray,
    P_init: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    phase_active: np.ndarray,
    p_phase: np.ndarray,
    actor_x: np.ndarray,
    A: np.ndarray,
    C_w: np.ndarray,
    swx: float,
    swy: float,
    apply_walls: bool,
    y_lo: float,
    y_hi: float,
    y_ind: int,
    is_acc: np.ndarray,
    is_rej: np.ndarray,
    c_acc: float,
    c_rej: float,
    use_terminal_cost: bool,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Numba fused ADF open-loop horizon: prediction, contact, and OLF cost.

    :param pi_init: Initial existence probabilities ``[B, N, 2]`` or shared layout.
    :param mu_init: Initial motion-state means.
    :param P_init: Initial motion-state covariances.
    :param lower: Precomputed phase-rectangle lower corners over the horizon.
    :param upper: Precomputed phase-rectangle upper corners over the horizon.
    :param phase_active: Precomputed phase-activity mask over the horizon.
    :param p_phase: Per-particle HIT/UP/DOWN ejection probabilities.
    :param actor_x: Streamwise actor positions.
    :param A: Discrete stage transition matrix.
    :param C_w: Discrete stage process-noise covariance.
    :param swx: Streamwise process-noise PSD (Gauß–Taylor).
    :param swy: Lateral process-noise PSD (Gauß–Taylor).
    :param apply_walls: Whether to apply lateral wall reflection in prediction.
    :param y_lo: Lower lateral wall.
    :param y_hi: Upper lateral wall.
    :param y_ind: Lateral position index.
    :param is_acc: Boolean accept mask over particles.
    :param is_rej: Boolean reject mask over particles.
    :param c_acc: Accept cost weight.
    :param c_rej: Reject cost weight.
    :param use_terminal_cost: If True, only terminal OLF costs; else stage costs.
    :param eps: A float, numerical floor.

    :returns: A np.array of OLF costs with leading batch shape matching ``pi_init``.
    """
    N = pi_init.shape[0]
    n = mu_init.shape[1]
    T = lower.shape[0]
    B = lower.shape[1]
    cost = np.zeros(B, dtype=np.float64)

    for b in prange(B):
        csum = 0.0
        for p in range(N):
            p0 = pi_init[p, 0]
            p1 = pi_init[p, 1]
            mu0 = np.empty(n, dtype=np.float64)
            mu1 = np.empty(n, dtype=np.float64)
            P0 = np.empty((n, n), dtype=np.float64)
            P1 = np.empty((n, n), dtype=np.float64)
            for r in range(n):
                mu0[r] = mu_init[p, r]
                mu1[r] = mu_init[p, r]
                for c in range(n):
                    P0[r, c] = P_init[p, r, c]
                    P1[r, c] = P_init[p, r, c]

            H0 = np.empty(n, dtype=np.float64)
            H1 = np.empty(n, dtype=np.float64)
            d_gt = np.empty(2, dtype=np.float64)
            R_gt = np.empty((2, 2), dtype=np.float64)

            for t in range(T):
                p0, p1, m_ej = _contact_one_particle(
                    p0, p1, mu0, P0, mu1, P1,
                    lower[t, b], upper[t, b], phase_active[t, b],
                    p_phase[p], actor_x, swx, swy, n, eps,
                    H0, H1, d_gt, R_gt,
                )
                if not use_terminal_cost:
                    # E[g_n] contrib: (c_acc 1_acc - c_rej 1_rej) * π̃^{(1)} p_ej
                    if is_acc[p]:
                        csum += c_acc * m_ej
                    if is_rej[p]:
                        csum -= c_rej * m_ej
                _predict_one(mu0, P0, A, C_w, apply_walls, y_lo, y_hi, y_ind)
                _predict_one(mu1, P1, A, C_w, apply_walls, y_lo, y_hi, y_ind)

            if use_terminal_cost:
                if is_acc[p]:
                    csum += c_acc * p0
                if is_rej[p]:
                    csum += c_rej * p1
        cost[b] = csum
    return cost
