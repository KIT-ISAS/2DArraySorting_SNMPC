"""ADF-inspired moment matching for the particle–contact model.

Implements the mass / first- / second-moment calculations from
``apdx:moment_matching_particle_and_contact_model`` (and the underlying
Gaussian forward/backward projection of
``apdx:moment_matching_disjoint_thresholds``).

Rectangular bivariate truncation follows Muthén (1990) / Manjunath & Wilhelm
(via face and corner densities of the untruncated Gaussian, as in ``tmvtnorm``).
When the ξ-covariance is diagonal, moments factor into independent 1D
truncations.

All core routines are NumPy-vectorized over a leading batch of truncation
boxes ``(j, m)``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# from scipy.special._ufuncs import _bivariate_normal_cdf as _bvnu
from scipy.stats import multivariate_normal, norm


from adf.adf_bivariate_rect_cdf import bivariate_rect_cdf
from adf.adf_numba_kernels import (
    NUMBA_AVAILABLE,
    adf_phased_contact_match_numba,
    backward_project_numba,
    forward_project_numba,
    moments_1d_truncated_gaussian_numba,
    moments_2d_correlated_numba,
    moments_2d_uncorrelated_numba,
    numba_enabled,
    warmup_adf_numba,
)

# ---------------------------------------------------------------------------
# Univariate truncated Gaussian
# ---------------------------------------------------------------------------

def moments_1d_truncated_gaussian(
    mu: np.ndarray,
    var: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mass, mean and variance of a (batch of) 1D Gaussians truncated to ``[a, b]``.

    :param mu: A np.array, the mean(s) of the untruncated Gaussian.
    :param var: A np.array, the variance(s) of the untruncated Gaussian.
    :param a: A np.array, the lower truncation bound(s).
    :param b: A np.array, the upper truncation bound(s).
    :param eps: A float, the numerical floor for mass / variance.

    :returns: A tuple ``(mass, mean, var)`` of np.arrays broadcast to the common
        shape of the inputs, the truncated probability mass, mean, and variance.
    """
    mu, var, a, b = np.broadcast_arrays(
        np.asarray(mu, dtype=float),
        np.asarray(var, dtype=float),
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
    )
    if numba_enabled() and mu.size >= 64:
        shape = mu.shape
        m, mean, vt = moments_1d_truncated_gaussian_numba(
            np.ascontiguousarray(mu, dtype=np.float64).ravel(),
            np.ascontiguousarray(var, dtype=np.float64).ravel(),
            np.ascontiguousarray(a, dtype=np.float64).ravel(),
            np.ascontiguousarray(b, dtype=np.float64).ravel(),
            eps,
        )
        return m.reshape(shape), mean.reshape(shape), vt.reshape(shape)

    sigma = np.sqrt(np.maximum(var, eps))
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    Phi_a = norm.cdf(alpha)
    Phi_b = norm.cdf(beta)
    mass = np.clip(Phi_b - Phi_a, 0.0, 1.0)

    phi_a = norm.pdf(alpha)
    phi_b = norm.pdf(beta)
    # Avoid 0/0 when mass is tiny: fall back to untruncated moments.
    safe = mass > eps
    dens_ratio = np.zeros_like(mass)
    dens_ratio[safe] = (phi_a[safe] - phi_b[safe]) / mass[safe]

    mean = np.where(safe, mu + sigma * dens_ratio, mu)

    # Var(Z | a<Z<b) = 1 + (α φ(α) − β φ(β))/Z − ((φ(α)−φ(β))/Z)^2 for Z~N(0,1)
    edge = np.zeros_like(mass)
    edge[safe] = (alpha[safe] * phi_a[safe] - beta[safe] * phi_b[safe]) / mass[safe]
    var_std = np.where(safe, 1.0 + edge - dens_ratio**2, 1.0)
    var_trunc = np.where(safe, np.maximum(var_std, 0.0) * var, var)
    return mass, mean, var_trunc


# ---------------------------------------------------------------------------
# Bivariate rectangular truncation (Muthén / Manjunath–Wilhelm, d = 2)
# ---------------------------------------------------------------------------


# TODO: Für scipy > 1.18
# def _bivariate_rect_cdf(
#     mean: np.ndarray,
#     cov: np.ndarray,
#     lower: np.ndarray,
#     upper: np.ndarray,
# ) -> np.ndarray:
#     """P(lower ≤ X ≤ upper) for a batch of bivariate Gaussians.
#
#     Parameters
#     ----------
#     mean : (..., 2)
#     cov : (..., 2, 2)
#     lower, upper : (..., 2)
#     """
#     mean = np.asarray(mean, dtype=float)
#     cov = np.asarray(cov, dtype=float)
#     lower = np.asarray(lower, dtype=float)
#     upper = np.asarray(upper, dtype=float)
#     mean, lower, upper = np.broadcast_arrays(mean, lower, upper)
#     if cov.ndim == 2:
#         cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))
#     else:
#         cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))
#
#     s00 = cov[..., 0, 0]
#     s11 = cov[..., 1, 1]
#     s01 = cov[..., 0, 1]
#     sig1 = np.sqrt(np.maximum(s00, 1e-30))
#     sig2 = np.sqrt(np.maximum(s11, 1e-30))
#     rho = np.clip(s01 / (sig1 * sig2), -1.0, 1.0)
#
#     m0 = mean[..., 0]
#     m1 = mean[..., 1]
#     lo0 = lower[..., 0]
#     lo1 = lower[..., 1]
#     up0 = upper[..., 0]
#     up1 = upper[..., 1]
#     xl = np.where(np.isfinite(lo0), (lo0 - m0) / sig1, -np.inf)
#     xu = np.where(np.isfinite(up0), (up0 - m0) / sig1, np.inf)
#     yl = np.where(np.isfinite(lo1), (lo1 - m1) / sig2, -np.inf)
#     yu = np.where(np.isfinite(up1), (up1 - m1) / sig2, np.inf)
#     out = _bvnu(xl, yl, rho) - _bvnu(xu, yl, rho) - _bvnu(xl, yu, rho) + _bvnu(xu, yu, rho)
#     out = np.clip(out, 0.0, 1.0)
#     degenerate = (lo0 >= up0) | (lo1 >= up1)
#     return np.where(degenerate, 0.0, out)


def _face_densities_bivariate(
    a: np.ndarray,
    b: np.ndarray,
    cov: np.ndarray,
    mass: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Face probability densities of a bivariate Gaussian on a rectangle boundary.

    :param a: A np.array of shape ``[..., 2]``, the lower rectangle corners.
    :param b: A np.array of shape ``[..., 2]``, the upper rectangle corners.
    :param cov: A np.array of shape ``[..., 2, 2]``, the Gaussian covariances.
    :param mass: A np.array of shape ``[...]``, the rectangle probability masses.
    :param eps: A float, the numerical floor.

    :returns: A np.array of face densities used by the correlated 2D moment formulas.
    """
    batch_shape = mass.shape
    # Faces: (dim0@a0, dim0@b0, dim1@a1, dim1@b1) stacked on last axis.
    # Coordinates are already centered (mean subtracted), so face mean is 0.
    x = np.stack([a[..., 0], b[..., 0], a[..., 1], b[..., 1]], axis=-1)
    s00 = cov[..., 0, 0][..., None]
    s11 = cov[..., 1, 1][..., None]
    s01 = cov[..., 0, 1][..., None]
    c_nn = np.concatenate([s00, s00, s11, s11], axis=-1)
    c_cross = np.concatenate([s01, s01, s01, s01], axis=-1)
    c_oo = np.concatenate([s11, s11, s00, s00], axis=-1)

    a_o = np.stack([a[..., 1], a[..., 1], a[..., 0], a[..., 0]], axis=-1)
    b_o = np.stack([b[..., 1], b[..., 1], b[..., 0], b[..., 0]], axis=-1)
    lo_n = np.stack([a[..., 0], a[..., 0], a[..., 1], a[..., 1]], axis=-1)
    up_n = np.stack([b[..., 0], b[..., 0], b[..., 1], b[..., 1]], axis=-1)

    c_nn_safe = np.maximum(c_nn, eps)
    cond_mean = x * c_cross / c_nn_safe
    cond_var = np.maximum(c_oo - c_cross * c_cross / c_nn_safe, 0.0)
    cond_sd = np.sqrt(np.maximum(cond_var, eps))
    cond_mass = norm.cdf((b_o - cond_mean) / cond_sd) - norm.cdf(
        (a_o - cond_mean) / cond_sd
    )
    unnorm = norm.pdf(x, loc=0.0, scale=np.sqrt(c_nn_safe)) * cond_mass

    inside = (lo_n <= x) & (x <= up_n) & np.isfinite(x)
    ok = inside & (mass[..., None] > eps)
    dens = np.zeros(batch_shape + (4,), dtype=float)
    if np.any(ok):
        dens = np.where(ok, unnorm / np.maximum(mass[..., None], eps), dens)
    Fa = np.stack([dens[..., 0], dens[..., 2]], axis=-1)
    Fb = np.stack([dens[..., 1], dens[..., 3]], axis=-1)
    return Fa, Fb


def _corner_densities_bivariate(
    a: np.ndarray,
    b: np.ndarray,
    cov: np.ndarray,
    mass: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Corner probability densities of a bivariate Gaussian on a rectangle.

    :param a: A np.array of shape ``[..., 2]``, the lower rectangle corners.
    :param b: A np.array of shape ``[..., 2]``, the upper rectangle corners.
    :param cov: A np.array of shape ``[..., 2, 2]``, the Gaussian covariances.
    :param mass: A np.array of shape ``[...]``, the rectangle probability masses.
    :param eps: A float, the numerical floor.

    :returns: A np.array of corner densities used by the correlated 2D moment formulas.
    """
    batch_shape = mass.shape
    x0 = np.stack([a[..., 0], b[..., 0], a[..., 0], b[..., 0]], axis=-1)
    x1 = np.stack([a[..., 1], a[..., 1], b[..., 1], b[..., 1]], axis=-1)

    dens = np.zeros(batch_shape + (4,), dtype=float)
    finite = np.isfinite(x0) & np.isfinite(x1)
    inside = (
        finite
        & (a[..., 0, None] <= x0)
        & (x0 <= b[..., 0, None])
        & (a[..., 1, None] <= x1)
        & (x1 <= b[..., 1, None])
    )
    ok = inside & (mass[..., None] > eps)
    if np.any(ok):
        s00 = cov[..., 0, 0]
        s11 = cov[..., 1, 1]
        s01 = cov[..., 0, 1]
        det = np.maximum(s00 * s11 - s01 * s01, eps)
        inv00 = s11 / det
        inv11 = s00 / det
        inv01 = -s01 / det
        d0 = np.where(ok, x0, 0.0)
        d1 = np.where(ok, x1, 0.0)
        quad = (
            inv00[..., None] * d0 * d0
            + 2.0 * inv01[..., None] * d0 * d1
            + inv11[..., None] * d1 * d1
        )
        unnorm = np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det)[..., None])
        dens = np.where(ok, unnorm / np.maximum(mass[..., None], eps), dens)

    return dens[..., 0], dens[..., 1], dens[..., 2], dens[..., 3]


def _moments_2d_rect_truncated_gaussian_correlated(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correlated bivariate rectangular truncation moments (Muthén / Manjunath–Wilhelm).

    :param mean: A np.array of shape ``[..., 2]``, the untruncated means.
    :param cov: A np.array of shape ``[..., 2, 2]``, the untruncated covariances.
    :param lower: A np.array of shape ``[..., 2]``, the lower rectangle bounds.
    :param upper: A np.array of shape ``[..., 2]``, the upper rectangle bounds.
    :param eps: A float, the numerical floor for mass / variances.

    :returns: A tuple ``(mass, mean_trunc, cov_trunc)`` of np.arrays for the
        truncated bivariate distribution.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    batch_shape = mean.shape[:-1]
    flat = int(np.prod(batch_shape)) if batch_shape else 1

    if numba_enabled() and flat >= 1:
        m, mt, ct = moments_2d_correlated_numba(
            np.ascontiguousarray(mean.reshape(flat, 2), dtype=np.float64),
            np.ascontiguousarray(cov.reshape(flat, 2, 2), dtype=np.float64),
            np.ascontiguousarray(lower.reshape(flat, 2), dtype=np.float64),
            np.ascontiguousarray(upper.reshape(flat, 2), dtype=np.float64),
            eps,
        )
        return (
            m.reshape(batch_shape),
            mt.reshape(batch_shape + (2,)),
            ct.reshape(batch_shape + (2, 2)),
        )

    a = lower - mean
    b = upper - mean
    zero = np.zeros_like(mean)

    mass = bivariate_rect_cdf(zero, cov, a, b)

    Fa, Fb = _face_densities_bivariate(a, b, cov, mass, eps=eps)
    tmean0 = np.einsum("...ij,...j->...i", cov, Fa - Fb)

    d_aa, d_ba, d_ab, d_bb = _corner_densities_bivariate(a, b, cov, mass, eps=eps)
    F2_01 = (d_aa - d_ba) - (d_ab - d_bb)
    F2_10 = (d_aa - d_ab) - (d_ba - d_bb)
    F2 = np.zeros(mean.shape[:-1] + (2, 2), dtype=float)
    F2[..., 0, 1] = F2_01
    F2[..., 1, 0] = F2_10

    edge = np.where(np.isfinite(a), a * Fa, 0.0) - np.where(np.isfinite(b), b * Fb, 0.0)

    inv_diag = 1.0 / np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), eps)
    # term1_ij = Σ_q Σ_iq Σ_jq / Σ_qq * edge_q
    term1 = np.einsum(
        "...iq,...jq,...q,...q->...ij", cov, cov, inv_diag, edge
    )

    # term2 without Python face loops: only off-diagonal (j≠q) contributions.
    # For d=2: term2[..., i, j] = Σ_{q≠j} cov[..., i, q] * Σ_s tt_{j,s|q} F2[q,s]
    term2 = np.zeros_like(cov)
    for q in (0, 1):
        j = 1 - q
        tt_j0 = cov[..., j, 0] - cov[..., q, 0] * cov[..., j, q] * inv_diag[..., q]
        tt_j1 = cov[..., j, 1] - cov[..., q, 1] * cov[..., j, q] * inv_diag[..., q]
        sum_s = tt_j0 * F2[..., q, 0] + tt_j1 * F2[..., q, 1]
        term2[..., 0, j] += cov[..., 0, q] * sum_s
        term2[..., 1, j] += cov[..., 1, q] * sum_s

    exx = cov + term1 + term2
    cov_trunc = exx - tmean0[..., :, None] * tmean0[..., None, :]
    # Symmetrize numerically
    cov_trunc = 0.5 * (cov_trunc + np.swapaxes(cov_trunc, -1, -2))

    mean_trunc = tmean0 + mean
    # Where mass vanishes, fall back to prior moments.
    tiny = mass <= eps
    if np.any(tiny):
        mean_trunc = np.where(tiny[..., None], mean, mean_trunc)
        cov_trunc = np.where(tiny[..., None, None], cov, cov_trunc)
        mass = np.where(tiny, 0.0, mass)

    return mass, mean_trunc, cov_trunc


def _moments_2d_rect_truncated_gaussian_uncorrelated(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    batch_shape = mean.shape[:-1]
    flat = int(np.prod(batch_shape)) if batch_shape else 1
    if numba_enabled() and flat >= 1:
        m, mt, ct = moments_2d_uncorrelated_numba(
            np.ascontiguousarray(mean.reshape(flat, 2), dtype=np.float64),
            np.ascontiguousarray(cov.reshape(flat, 2, 2), dtype=np.float64),
            np.ascontiguousarray(lower.reshape(flat, 2), dtype=np.float64),
            np.ascontiguousarray(upper.reshape(flat, 2), dtype=np.float64),
            eps,
        )
        return (
            m.reshape(batch_shape),
            mt.reshape(batch_shape + (2,)),
            ct.reshape(batch_shape + (2, 2)),
        )

    m0, mu0, v0 = moments_1d_truncated_gaussian(
        mean[..., 0], cov[..., 0, 0], lower[..., 0], upper[..., 0], eps=eps
    )
    m1, mu1, v1 = moments_1d_truncated_gaussian(
        mean[..., 1], cov[..., 1, 1], lower[..., 1], upper[..., 1], eps=eps
    )
    mass = m0 * m1
    mean_t = np.stack([mu0, mu1], axis=-1)
    cov_t = np.zeros(mean.shape[:-1] + (2, 2), dtype=float)
    cov_t[..., 0, 0] = v0
    cov_t[..., 1, 1] = v1
    return mass, mean_t, cov_t


def moments_2d_rect_truncated_gaussian(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: float = 1e-15,
    corr_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mass and moments of a (batch of) bivariate Gaussians truncated to a rectangle.

    :param mean: A np.array of shape ``[..., 2]``, the untruncated means.
    :param cov: A np.array of shape ``[..., 2, 2]``, the untruncated covariances.
    :param lower: A np.array of shape ``[..., 2]``, the lower rectangle bounds.
    :param upper: A np.array of shape ``[..., 2]``, the upper rectangle bounds.
    :param eps: A float, the numerical floor for mass / variances.
    :param corr_tol: A float, the absolute correlation below which the uncorrelated
        product formula is used.

    :returns: A tuple ``(mass, mean_trunc, cov_trunc)`` of np.arrays, the truncated
        probability mass, mean (``[..., 2]``), and covariance (``[..., 2, 2]``).
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mean, lower, upper = np.broadcast_arrays(mean, lower, upper)
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2)).copy()
    else:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2)).copy()
    off = cov[..., 0, 1]
    uncorr = np.abs(off) <= corr_tol * np.sqrt(
        np.maximum(cov[..., 0, 0] * cov[..., 1, 1], 0.0)
    )
    if np.all(uncorr):
        return _moments_2d_rect_truncated_gaussian_uncorrelated(
            mean, cov, lower, upper, eps=eps
        )
    if not np.any(uncorr):
        return _moments_2d_rect_truncated_gaussian_correlated(
            mean, cov, lower, upper, eps=eps
        )

    # Mixed batch: run each path only on its subset.
    mass = np.empty(mean.shape[:-1], dtype=float)
    mean_trunc = np.empty_like(mean)
    cov_trunc = np.empty_like(cov)
    idx_u = np.flatnonzero(uncorr.reshape(-1))
    idx_c = np.flatnonzero((~uncorr).reshape(-1))
    flat_n = mean.reshape(-1, 2)
    flat_cov = cov.reshape(-1, 2, 2)
    flat_lo = lower.reshape(-1, 2)
    flat_up = upper.reshape(-1, 2)
    if idx_u.size:
        m_u, mt_u, ct_u = _moments_2d_rect_truncated_gaussian_uncorrelated(
            flat_n[idx_u], flat_cov[idx_u], flat_lo[idx_u], flat_up[idx_u], eps=eps
        )
        mass.reshape(-1)[idx_u] = m_u
        mean_trunc.reshape(-1, 2)[idx_u] = mt_u
        cov_trunc.reshape(-1, 2, 2)[idx_u] = ct_u
    if idx_c.size:
        m_c, mt_c, ct_c = _moments_2d_rect_truncated_gaussian_correlated(
            flat_n[idx_c], flat_cov[idx_c], flat_lo[idx_c], flat_up[idx_c], eps=eps
        )
        mass.reshape(-1)[idx_c] = m_c
        mean_trunc.reshape(-1, 2)[idx_c] = mt_c
        cov_trunc.reshape(-1, 2, 2)[idx_c] = ct_c
    return mass, mean_trunc, cov_trunc

# ---------------------------------------------------------------------------
# Gaussian forward / backward projection
# ---------------------------------------------------------------------------

def forward_project(
    mu: np.ndarray,
    P: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-projects a Gaussian state through an affine measurement map.

    Computes ``μ_ξ = H μ + d``, ``P_ξ = H P Hᵀ + R``, and the Kalman gain ``K``.

    :param mu: A np.array of shape ``[..., state_dim]``, the prior mean.
    :param P: A np.array of shape ``[..., state_dim, state_dim]``, the prior covariance.
    :param H: A np.array of shape ``[..., m, state_dim]``, the measurement Jacobian.
    :param d: A np.array of shape ``[..., m]``, the measurement offset.
    :param R: None or a np.array of shape ``[..., m, m]``, the measurement-noise
        covariance (zeros if None).

    :returns: A tuple ``(mu_xi, P_xi, K)`` of np.arrays, the predicted measurement
        mean, measurement covariance, and Kalman gain.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    P = np.asarray(P, dtype=float)
    H = np.asarray(H, dtype=float)
    d = np.asarray(d, dtype=float)

    if H.ndim == 2:
        H = H[None, ...]
    if d.ndim == 1:
        d = np.broadcast_to(d, (H.shape[0], d.shape[0])).copy()
    if mu.ndim == 1:
        mu = np.broadcast_to(mu, (H.shape[0], mu.shape[0])).copy()
    if P.ndim == 2:
        P = np.broadcast_to(P, (H.shape[0],) + P.shape).copy()

    if R is not None:
        R = np.asarray(R, dtype=float)
        if R.ndim == 2:
            R = np.broadcast_to(R, (H.shape[0], R.shape[0], R.shape[1])).copy()
    else:
        R = np.zeros((H.shape[0], H.shape[1], H.shape[1]), dtype=float)

    # Numba path specialized to m=2 (ADF contact ξ = [τ, y]).
    if numba_enabled() and H.shape[1] == 2 and H.shape[0] >= 1:
        return forward_project_numba(
            np.ascontiguousarray(mu, dtype=np.float64),
            np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(H, dtype=np.float64),
            np.ascontiguousarray(d, dtype=np.float64),
            np.ascontiguousarray(R, dtype=np.float64),
        )

    mu_xi = np.einsum("bij,bj->bi", H, mu) + d
    P_xi = np.einsum("bij,bjk,blk->bil", H, P, H) + R

    P_HT = np.einsum("bij,bkj->bik", P, H)  # (B, n, m)
    # Degenerate actuator maps are valid: inactive lateral dimensions have
    # zero measurement variance. Match the Numba 2x2 path by using a tiny
    # diagonal floor before solving rather than failing on a singular matrix.
    solve_rhs = np.swapaxes(P_HT, -1, -2)
    eye = np.eye(P_xi.shape[-1], dtype=P_xi.dtype)
    P_xi = P_xi + 1e-30 * eye
    K = np.linalg.solve(P_xi, solve_rhs)
    K = np.swapaxes(K, -1, -2)  # (B, n, m)
    return mu_xi, P_xi, K


def backward_project(
    mu: np.ndarray,
    P: np.ndarray,
    mu_xi: np.ndarray,
    P_xi: np.ndarray,
    K: np.ndarray,
    mu_y_trunc: np.ndarray,
    P_y_trunc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Gaussian conditioning lift of truncated subspace moments.

        μ_{x|S} = μ + K (μ_{y,trunc} − μ_ξ)
        P_{x|S} = P + K (P_{y,trunc} − P_ξ) Kᵀ

    :param mu: A np.array of shape ``[..., state_dim]``, the prior mean.
    :param P: A np.array of shape ``[..., state_dim, state_dim]``, the prior covariance.
    :param mu_xi: A np.array of shape ``[..., m]``, the predicted measurement mean.
    :param P_xi: A np.array of shape ``[..., m, m]``, the predicted measurement covariance.
    :param K: A np.array of shape ``[..., state_dim, m]``, the Kalman gain.
    :param mu_y_trunc: A np.array of shape ``[..., m]``, the truncated measurement mean.
    :param P_y_trunc: A np.array of shape ``[..., m, m]``, the truncated measurement covariance.

    :returns: A tuple ``(mu_post, P_post)`` of np.arrays with the same shapes as
        ``mu`` / ``P``, the posterior mean and covariance in the original state space.
    """
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)
    if mu.ndim == 1:
        mu = np.broadcast_to(mu, (K.shape[0], mu.shape[0])).copy()
    if P.ndim == 2:
        P = np.broadcast_to(P, (K.shape[0],) + P.shape).copy()

    if numba_enabled() and K.shape[-1] == 2 and K.shape[0] >= 1:
        return backward_project_numba(
            np.ascontiguousarray(mu, dtype=np.float64),
            np.ascontiguousarray(P, dtype=np.float64),
            np.ascontiguousarray(mu_xi, dtype=np.float64),
            np.ascontiguousarray(P_xi, dtype=np.float64),
            np.ascontiguousarray(K, dtype=np.float64),
            np.ascontiguousarray(mu_y_trunc, dtype=np.float64),
            np.ascontiguousarray(P_y_trunc, dtype=np.float64),
        )

    delta_mu = mu_y_trunc - mu_xi
    mu_x = mu + np.einsum("bij,bj->bi", K, delta_mu)
    delta_P = P_y_trunc - P_xi
    mu_corr = np.einsum("bij,bjk,blk->bil", K, delta_P, K)
    P_x = P + mu_corr
    P_x = 0.5 * (P_x + np.swapaxes(P_x, -1, -2))
    return mu_x, P_x


def active_truncation_box_mask(
    lower: np.ndarray,
    upper: np.ndarray,
    p_eject: np.ndarray | None = None,
    time_atol: float = 0.0,
) -> np.ndarray:
    """
    Returns ``True`` for truncation boxes that can eject in the current stage.

    A box is active when its temporal support is non-degenerate
    (``t_lo < t_hi`` with finite bounds) and, if given, ``p_eject > 0``.

    :param lower: A np.array of shape ``[..., 2]``, the lower box corners (time, lateral).
    :param upper: A np.array of shape ``[..., 2]``, the upper box corners.
    :param p_eject: None or a np.array broadcastable to the box batch, the ejection
        probabilities; boxes with ``p_eject <= 0`` are inactive.
    :param time_atol: A float, the absolute tolerance for ``t_lo < t_hi``.

    :returns: A Boolean np.array with the leading shape of ``lower`` / ``upper``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    t_lo = lower[..., 0]
    t_hi = upper[..., 0]
    active = (
        np.isfinite(t_lo)
        & np.isfinite(t_hi)
        & (t_hi > t_lo + time_atol)
    )
    if p_eject is not None:
        active &= np.asarray(p_eject, dtype=float) > 0.0
    return active


def truncated_state_moments_from_boxes(
    mu: np.ndarray,
    P: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    R: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncates a Gaussian state to a union of soft rectangular boxes in ``ξ``-space.

    :param mu: A np.array of shape ``[..., state_dim]``, the prior mean.
    :param P: A np.array of shape ``[..., state_dim, state_dim]``, the prior covariance.
    :param H: A np.array of shape ``[..., Nb, 2, state_dim]``, the measurement
        Jacobians into ``(τ, y)``.
    :param d: A np.array of shape ``[..., Nb, 2]``, the measurement offsets.
    :param lower: A np.array of shape ``[..., Nb, 2]``, the lower box corners.
    :param upper: A np.array of shape ``[..., Nb, 2]``, the upper box corners.
    :param R: None or a np.array of shape ``[..., Nb, 2, 2]``, the measurement-noise
        covariances (zeros if None).
    :param active_mask: None or a Boolean np.array of shape ``[..., Nb]``, selecting
        which boxes participate; if None, all boxes with non-degenerate support are used.

    :returns: A tuple ``(mass, mu_trunc, P_trunc)`` of np.arrays, the total truncated
        probability mass and the mixture mean / covariance of the truncated state
        in the original coordinates.
    """
    mu = np.asarray(mu, dtype=float)
    P = np.asarray(P, dtype=float)
    H = np.asarray(H, dtype=float)
    d = np.asarray(d, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if H.ndim == 3:
        nb, m, n = H.shape
        if active_mask is None:
            active_mask = active_truncation_box_mask(lower, upper)
        else:
            active_mask = np.asarray(active_mask, dtype=bool)
        w = np.zeros(nb, dtype=float)
        mu_ms = np.zeros((nb, n), dtype=float)
        P_ms = np.zeros((nb, n, n), dtype=float)
        idx = np.flatnonzero(active_mask)
        if idx.size == 0:
            return w, mu_ms, P_ms
        R_idx = None
        if R is not None:
            R_idx = np.asarray(R, dtype=float)[idx]
        mu_xi, P_xi, K = forward_project(mu, P, H[idx], d[idx], R=R_idx)
        w[idx], mu_y, P_y = moments_2d_rect_truncated_gaussian(
            mu_xi, P_xi, lower[idx], upper[idx]
        )
        mu_ms[idx], P_ms[idx] = backward_project(
            mu, P, mu_xi, P_xi, K, mu_y, P_y
        )
        return w, mu_ms, P_ms

    *leading, nb, m, n = H.shape
    flat = int(np.prod(leading)) if leading else 1
    if mu.ndim == 1:
        mu = np.broadcast_to(mu, (*leading, n))
    if P.ndim == 2:
        P = np.broadcast_to(P, (*leading, n, n))

    if active_mask is None:
        active_mask = active_truncation_box_mask(lower, upper)
    else:
        active_mask = np.asarray(active_mask, dtype=bool)
    out_leading = tuple(leading) + (nb,)
    w = np.zeros(out_leading, dtype=float)
    mu_ms = np.zeros(out_leading + (n,), dtype=float)
    P_ms = np.zeros(out_leading + (n, n), dtype=float)
    idx = np.flatnonzero(active_mask.reshape(-1))
    if idx.size == 0:
        return w, mu_ms, P_ms

    # Compact gather: (batch_particle, box) without expanding μ/P to Nb.
    bp = idx // nb
    box = idx % nb
    mu_f = mu.reshape(flat, n)[bp]
    P_f = P.reshape(flat, n, n)[bp]
    H_f = H.reshape(flat, nb, m, n)[bp, box]
    d_f = d.reshape(flat, nb, m)[bp, box]
    lo_f = lower.reshape(flat, nb, m)[bp, box]
    up_f = upper.reshape(flat, nb, m)[bp, box]
    R_f = None
    if R is not None:
        R_f = np.asarray(R, dtype=float).reshape(flat, nb, m, m)[bp, box]

    mu_xi, P_xi, K = forward_project(mu_f, P_f, H_f, d_f, R=R_f)
    w_a, mu_y, P_y = moments_2d_rect_truncated_gaussian(mu_xi, P_xi, lo_f, up_f)
    mu_ms_a, P_ms_a = backward_project(mu_f, P_f, mu_xi, P_xi, K, mu_y, P_y)
    w.reshape(-1)[idx] = w_a
    mu_ms.reshape(flat * nb, n)[idx] = mu_ms_a
    P_ms.reshape(flat * nb, n, n)[idx] = P_ms_a
    return w, mu_ms, P_ms


def _compose_from_active_ejection(
    pi0,
    mu0: np.ndarray,
    P0: np.ndarray,
    pi1,
    mu1: np.ndarray,
    P1: np.ndarray,
    bp: np.ndarray,
    weights: np.ndarray,
    mu_ms: np.ndarray,
    P_ms: np.ndarray,
    flat: int,
    leading: tuple,
    n: int,
    eps: float = 1e-15,
) -> ADFMoments:
    """
    Composes ADF dual-mode moments from active-box ejection contributions.

    :param pi0: Dead-mode prior mass.
    :param mu0: Dead-mode prior mean.
    :param P0: Dead-mode prior covariance.
    :param pi1: Living-mode prior mass.
    :param mu1: Living-mode prior mean.
    :param P1: Living-mode prior covariance.
    :param bp: Packed truncated / projected moments from the active boxes.
    :param weights: Unnormalized box / mixture weights.
    :param mu_ms: Surviving (non-ejected) living-mode mean after contact.
    :param P_ms: Surviving living-mode covariance after contact.
    :param flat: An integer, the flattened batch size used for packing.
    :param leading: A tuple of integers, the leading batch shape to restore.
    :param n: An integer, the motion-state dimension.
    :param eps: A float, the numerical floor.

    :returns: An :class:`ADFMoments` two-mode posterior.
    """
    pi0 = np.asarray(pi0, dtype=float)
    pi1 = np.asarray(pi1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    P1 = np.asarray(P1, dtype=float)

    M0_ej = np.zeros(flat, dtype=float)
    M1_ej = np.zeros((flat, n), dtype=float)
    M2_ej = np.zeros((flat, n, n), dtype=float)
    if weights.size:
        np.add.at(M0_ej, bp, weights)
        np.add.at(M1_ej, bp, weights[:, None] * mu_ms)
        np.add.at(M2_ej, bp, weights[:, None, None] * (P_ms + _outer(mu_ms)))

    if leading:
        M0_ej = M0_ej.reshape(leading)
        M1_ej = M1_ej.reshape(leading + (n,))
        M2_ej = M2_ej.reshape(leading + (n, n))
        if pi0.ndim == 0:
            pi0 = np.broadcast_to(pi0, leading).copy()
        if pi1.ndim == 0:
            pi1 = np.broadcast_to(pi1, leading).copy()
        if mu0.ndim == 1:
            mu0 = np.broadcast_to(mu0, leading + (n,)).copy()
        if mu1.ndim == 1:
            mu1 = np.broadcast_to(mu1, leading + (n,)).copy()
        if P0.ndim == 2:
            P0 = np.broadcast_to(P0, leading + (n, n)).copy()
        if P1.ndim == 2:
            P1 = np.broadcast_to(P1, leading + (n, n)).copy()
    else:
        # scalar batch: arrays stay 0-d / 1-d / 2-d
        pass

    M0_r0 = pi0 + M0_ej
    M1_r0 = pi0[..., None] * mu0 + M1_ej
    M2_r0 = pi0[..., None, None] * (P0 + _outer(mu0)) + M2_ej

    M0_r1 = pi1 - M0_ej
    M1_r1 = pi1[..., None] * mu1 - M1_ej
    M2_r1 = pi1[..., None, None] * (P1 + _outer(mu1)) - M2_ej

    M0 = np.stack([M0_r0, M0_r1], axis=-1)
    M1 = np.stack([M1_r0, M1_r1], axis=-2)
    M2 = np.stack([M2_r0, M2_r1], axis=-3)

    pi = M0.copy()
    alive = pi > eps
    pi_safe = np.where(alive, pi, 1.0)

    mu_r0 = np.where(alive[..., 0, None], M1[..., 0, :] / pi_safe[..., 0, None], mu0)
    mu_r1 = np.where(alive[..., 1, None], M1[..., 1, :] / pi_safe[..., 1, None], mu1)
    mu = np.stack([mu_r0, mu_r1], axis=-2)

    P_r0 = np.where(
        alive[..., 0, None, None],
        M2[..., 0, :, :] / pi_safe[..., 0, None, None] - _outer(mu_r0),
        P0,
    )
    P_r1 = np.where(
        alive[..., 1, None, None],
        M2[..., 1, :, :] / pi_safe[..., 1, None, None] - _outer(mu_r1),
        P1,
    )
    P = np.stack([P_r0, P_r1], axis=-3)
    P = 0.5 * (P + np.swapaxes(P, -1, -2))
    pi = np.where(alive, pi, 0.0)
    return ADFMoments(M0=M0, M1=M1, M2=M2, pi=pi, mu=mu, P=P)


# ---------------------------------------------------------------------------
# Contact-model ADF composition
# ---------------------------------------------------------------------------

class ADFMoments(NamedTuple):
    """Unnormalized moments and reconstructed scaled-Gaussian parameters."""

    M0: np.ndarray  # (2,) mass for r ∈ {0,1}
    M1: np.ndarray  # (2, n)
    M2: np.ndarray  # (2, n, n)
    pi: np.ndarray  # (2,) = M0
    mu: np.ndarray  # (2, n)
    P: np.ndarray  # (2, n, n)


def _outer(v: np.ndarray) -> np.ndarray:
    return v[..., :, None] * v[..., None, :]


def compose_contact_adf_moments(
    pi0,
    mu0: np.ndarray,
    P0: np.ndarray,
    pi1,
    mu1: np.ndarray,
    P1: np.ndarray,
    w: np.ndarray,
    mu_ms: np.ndarray,
    P_ms: np.ndarray,
    p_eject: np.ndarray,
    eps: float = 1e-15,
) -> ADFMoments:
    """
    Composes two-mode ADF moments from truncated living / ejected contributions.

    :param pi0: A np.array, the dead-mode prior mass.
    :param mu0: A np.array, the dead-mode prior mean.
    :param P0: A np.array, the dead-mode prior covariance.
    :param pi1: A np.array, the living-mode prior mass.
    :param mu1: A np.array, the living-mode prior mean.
    :param P1: A np.array, the living-mode prior covariance.
    :param w: A np.array, the unnormalized mixture / box weights from truncation.
    :param mu_ms: A np.array, the surviving (non-ejected) living-mode mean after contact.
    :param P_ms: A np.array, the surviving living-mode covariance after contact.
    :param p_eject: A np.array, the per-particle (or batched) ejection probabilities.
    :param eps: A float, the numerical floor for mixing weights.

    :returns: An :class:`ADFMoments` instance with two-mode posterior ``(π, μ, P)``
        and raw moments ``(M0, M1, M2)``.
    """
    pi0 = np.asarray(pi0, dtype=float)
    pi1 = np.asarray(pi1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    P0 = np.asarray(P0, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    w = np.asarray(w, dtype=float)
    mu_ms = np.asarray(mu_ms, dtype=float)
    P_ms = np.asarray(P_ms, dtype=float)
    p_eject = np.asarray(p_eject, dtype=float)

    weights = pi1[..., None] * p_eject * w  # (..., Nb)

    M0_ej = np.sum(weights, axis=-1)
    M1_ej = np.einsum("...b,...bi->...i", weights, mu_ms)
    M2_ej = np.einsum("...b,...bij->...ij", weights, P_ms + _outer(mu_ms))

    M0_r0 = pi0 + M0_ej
    M1_r0 = pi0[..., None] * mu0 + M1_ej
    M2_r0 = pi0[..., None, None] * (P0 + _outer(mu0)) + M2_ej

    M0_r1 = pi1 - M0_ej
    M1_r1 = pi1[..., None] * mu1 - M1_ej
    M2_r1 = pi1[..., None, None] * (P1 + _outer(mu1)) - M2_ej

    M0 = np.stack([M0_r0, M0_r1], axis=-1)  # (..., 2)
    M1 = np.stack([M1_r0, M1_r1], axis=-2)  # (..., 2, n)
    M2 = np.stack([M2_r0, M2_r1], axis=-3)  # (..., 2, n, n)

    pi = M0.copy()
    alive = pi > eps
    pi_safe = np.where(alive, pi, 1.0)

    mu0_b = np.broadcast_to(mu0, M1.shape[:-2] + M1.shape[-1:])
    mu1_b = np.broadcast_to(mu1, M1.shape[:-2] + M1.shape[-1:])
    P0_b = np.broadcast_to(P0, M2.shape[:-3] + M2.shape[-2:])
    P1_b = np.broadcast_to(P1, M2.shape[:-3] + M2.shape[-2:])

    mu_r0 = np.where(alive[..., 0, None], M1[..., 0, :] / pi_safe[..., 0, None], mu0_b)
    mu_r1 = np.where(alive[..., 1, None], M1[..., 1, :] / pi_safe[..., 1, None], mu1_b)
    mu = np.stack([mu_r0, mu_r1], axis=-2)

    P_r0 = np.where(
        alive[..., 0, None, None],
        M2[..., 0, :, :] / pi_safe[..., 0, None, None] - _outer(mu_r0),
        P0_b,
    )
    P_r1 = np.where(
        alive[..., 1, None, None],
        M2[..., 1, :, :] / pi_safe[..., 1, None, None] - _outer(mu_r1),
        P1_b,
    )
    P = np.stack([P_r0, P_r1], axis=-3)
    P = 0.5 * (P + np.swapaxes(P, -1, -2))
    pi = np.where(alive, pi, 0.0)

    return ADFMoments(M0=M0, M1=M1, M2=M2, pi=pi, mu=mu, P=P)

def adf_contact_moment_matching(
    pi0,
    mu0: np.ndarray,
    P0: np.ndarray,
    pi1,
    mu1: np.ndarray,
    P1: np.ndarray,
    H: np.ndarray,
    d: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    p_eject: np.ndarray | None = None,
    R: np.ndarray | None = None,
    phase_active: np.ndarray | None = None,
    p_phase: np.ndarray | None = None,
) -> ADFMoments:
    """
    End-to-end ADF moment matching for one prediction step (contact model).

    Truncation / projection uses only the living prior ``(pi1, mu1, P1)``.
    The dead prior ``s = 0`` contributes its full mass to mode ``r = 0``.

    Supports dense flat boxes and compact actuator–phase layouts (see module docs).

    :param pi0: A np.array, the dead-mode prior mass.
    :param mu0: A np.array, the dead-mode prior mean.
    :param P0: A np.array, the dead-mode prior covariance.
    :param pi1: A np.array, the living-mode prior mass.
    :param mu1: A np.array, the living-mode prior mean.
    :param P1: A np.array, the living-mode prior covariance.
    :param H: A np.array, the measurement Jacobian(s) into ``(τ, y)``.
    :param d: A np.array, the measurement offset(s).
    :param lower: A np.array, the truncation-box lower corners.
    :param upper: A np.array, the truncation-box upper corners.
    :param p_eject: None or a np.array, dense per-box ejection probabilities.
    :param R: None or a np.array, the measurement-noise covariance in ``ξ``-space.
    :param phase_active: None or a Boolean np.array of shape ``[B, Na, 3]``, sparse
        phase activity.
    :param p_phase: None or a np.array of shape ``[N, 3]``, phase success probabilities.

    :returns: An :class:`ADFMoments` two-mode posterior after the contact update.
    """
    H = np.asarray(H, dtype=float)
    d = np.asarray(d, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if R is not None:
        R = np.asarray(R, dtype=float)

    if H.size == 0:
        pi0 = np.asarray(pi0, dtype=float)
        pi1 = np.asarray(pi1, dtype=float)
        mu0 = np.asarray(mu0, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        P0 = np.asarray(P0, dtype=float)
        P1 = np.asarray(P1, dtype=float)
        pi = np.stack([pi0, pi1], axis=-1)
        mu = np.stack([mu0, mu1], axis=-2)
        P = np.stack([P0, P1], axis=-3)
        return ADFMoments(
            M0=pi,
            M1=pi[..., None] * mu,
            M2=pi[..., None, None] * (P + _outer(mu)),
            pi=pi,
            mu=mu,
            P=P,
        )

    # Compact actuator–phase layout: H (B,N,Na,2,n), lower (B,Na,3,2)
    if (
        H.ndim == 5
        and lower.ndim == 4
        and H.shape[2] == lower.shape[1]
        and lower.shape[-1] == 2
        and (phase_active is not None or (p_eject is not None and np.asarray(p_eject).ndim == 4))
    ):
        return _adf_contact_moment_matching_phased(
            pi0,
            mu0,
            P0,
            pi1,
            mu1,
            P1,
            H,
            d,
            lower,
            upper,
            p_eject,
            R,
            phase_active=phase_active,
            p_phase=p_phase,
        )

    if p_eject is None:
        raise ValueError("p_eject is required for dense flat-box matching")
    p_eject = np.asarray(p_eject, dtype=float)

    active = active_truncation_box_mask(lower, upper, p_eject=p_eject)
    w, mu_ms, P_ms = truncated_state_moments_from_boxes(
        mu1, P1, H, d, lower, upper, R=R, active_mask=active
    )
    return compose_contact_adf_moments(
        pi0, mu0, P0, pi1, mu1, P1, w, mu_ms, P_ms, p_eject
    )


def _adf_contact_moment_matching_phased(
    pi0,
    mu0: np.ndarray,
    P0: np.ndarray,
    pi1,
    mu1: np.ndarray,
    P1: np.ndarray,
    H_act: np.ndarray,
    d_act: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    p_eject: np.ndarray | None,
    R_act: np.ndarray | None,
    phase_active: np.ndarray | None = None,
    p_phase: np.ndarray | None = None,
    eps: float = 1e-15,
) -> ADFMoments:
    """
    Phased (compact actuator–phase) ADF contact moment matching.

    :param pi0: Dead-mode prior mass.
    :param mu0: Dead-mode prior mean.
    :param P0: Dead-mode prior covariance.
    :param pi1: Living-mode prior mass.
    :param mu1: Living-mode prior mean.
    :param P1: Living-mode prior covariance.
    :param H_act: Actuator Gauß–Taylor Jacobians.
    :param d_act: Actuator Gauß–Taylor offsets.
    :param lower: Phase-rectangle lower corners.
    :param upper: Phase-rectangle upper corners.
    :param p_eject: Dense per-box ejection probabilities (fallback layout).
    :param R_act: Actuator measurement-noise covariances.
    :param phase_active: Boolean phase-activity mask.
    :param p_phase: Per-particle HIT/UP/DOWN ejection probabilities.
    :param eps: A float, numerical floor.

    :returns: An :class:`ADFMoments` two-mode posterior after the contact update.
    """
    mu1 = np.asarray(mu1, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    batch_size, num_particles, num_actors, _, state_dim = H_act.shape
    leading = (batch_size, num_particles)
    flat = batch_size * num_particles

    if phase_active is None:
        if p_eject is None:
            raise ValueError("Need phase_active+p_phase or dense p_eject")
        p_eject = np.asarray(p_eject, dtype=float)
        phase_active = (
            np.any(p_eject > 0.0, axis=1)
            & np.isfinite(lower[..., 0])
            & np.isfinite(upper[..., 0])
            & (upper[..., 0] > lower[..., 0])
        )
        # Recover p_phase from any positive (b,a) slice; fallback zeros.
        pe_pos = p_eject > 0.0
        if np.any(pe_pos):
            # Take max over (B, Na) → (N, 3)
            p_phase = np.max(p_eject, axis=(0, 2))
        else:
            p_phase = np.zeros((num_particles, p_eject.shape[-1]), dtype=float)
    else:
        phase_active = np.asarray(phase_active, dtype=bool)
        if p_phase is None:
            raise ValueError("p_phase required with phase_active")
        p_phase = np.asarray(p_phase, dtype=float)

    if numba_enabled() and R_act is not None:
        pi_o, mu_o, P_o = adf_phased_contact_match_numba(
            np.ascontiguousarray(np.asarray(pi0, dtype=np.float64), dtype=np.float64),
            np.ascontiguousarray(np.asarray(mu0, dtype=np.float64), dtype=np.float64),
            np.ascontiguousarray(np.asarray(P0, dtype=np.float64), dtype=np.float64),
            np.ascontiguousarray(np.asarray(pi1, dtype=np.float64), dtype=np.float64),
            np.ascontiguousarray(mu1, dtype=np.float64),
            np.ascontiguousarray(P1, dtype=np.float64),
            np.ascontiguousarray(H_act, dtype=np.float64),
            np.ascontiguousarray(d_act, dtype=np.float64),
            np.ascontiguousarray(R_act, dtype=np.float64),
            np.ascontiguousarray(lower, dtype=np.float64),
            np.ascontiguousarray(upper, dtype=np.float64),
            np.ascontiguousarray(phase_active, dtype=np.bool_),
            np.ascontiguousarray(p_phase, dtype=np.float64),
            eps,
        )
        return ADFMoments(
            M0=pi_o,
            M1=pi_o[..., None] * mu_o,
            M2=pi_o[..., None, None] * (P_o + _outer(mu_o)),
            pi=pi_o,
            mu=mu_o,
            P=P_o,
        )

    # NumPy fallback: enumerate only active (b, a, phase), expand by particles.
    b_idx, a_idx, ph_idx = np.nonzero(phase_active)
    if b_idx.size == 0:
        return _compose_from_active_ejection(
            pi0, mu0, P0, pi1, mu1, P1,
            bp=np.empty(0, dtype=int),
            weights=np.empty(0, dtype=float),
            mu_ms=np.empty((0, state_dim), dtype=float),
            P_ms=np.empty((0, state_dim, state_dim), dtype=float),
            flat=flat,
            leading=leading,
            n=state_dim,
            eps=eps,
        )

    # Particles with positive phase probability
    part_lists = []
    for ph in range(p_phase.shape[1]):
        part_lists.append(np.flatnonzero(p_phase[:, ph] > 0.0))

    b_all = []
    n_all = []
    a_all = []
    ph_all = []
    for b, a, ph in zip(b_idx, a_idx, ph_idx):
        parts = part_lists[int(ph)]
        if parts.size == 0:
            continue
        b_all.append(np.full(parts.shape, b, dtype=int))
        n_all.append(parts)
        a_all.append(np.full(parts.shape, a, dtype=int))
        ph_all.append(np.full(parts.shape, ph, dtype=int))
    if not b_all:
        return _compose_from_active_ejection(
            pi0, mu0, P0, pi1, mu1, P1,
            bp=np.empty(0, dtype=int),
            weights=np.empty(0, dtype=float),
            mu_ms=np.empty((0, state_dim), dtype=float),
            P_ms=np.empty((0, state_dim, state_dim), dtype=float),
            flat=flat,
            leading=leading,
            n=state_dim,
            eps=eps,
        )

    i_batch = np.concatenate(b_all)
    i_part = np.concatenate(n_all)
    i_act = np.concatenate(a_all)
    i_phase = np.concatenate(ph_all)
    bp = i_batch * num_particles + i_part

    H_a = H_act[i_batch, i_part, i_act]
    d_a = d_act[i_batch, i_part, i_act]
    lo_a = lower[i_batch, i_act, i_phase]
    up_a = upper[i_batch, i_act, i_phase]
    R_a = None if R_act is None else R_act[i_batch, i_part, i_act]
    mu_a = mu1[i_batch, i_part]
    P_a = P1[i_batch, i_part]

    mu_xi, P_xi, K = forward_project(mu_a, P_a, H_a, d_a, R=R_a)
    w_a, mu_y, P_y = moments_2d_rect_truncated_gaussian(mu_xi, P_xi, lo_a, up_a)
    mu_ms_a, P_ms_a = backward_project(mu_a, P_a, mu_xi, P_xi, K, mu_y, P_y)

    pi1_a = np.asarray(pi1, dtype=float)
    if pi1_a.ndim == 0:
        pi1_w = np.full(i_batch.shape, float(pi1_a))
    else:
        pi1_w = pi1_a[i_batch, i_part]
    weights = pi1_w * p_phase[i_part, i_phase] * w_a
    return _compose_from_active_ejection(
        pi0, mu0, P0, pi1, mu1, P1,
        bp=bp,
        weights=weights,
        mu_ms=mu_ms_a,
        P_ms=P_ms_a,
        flat=flat,
        leading=leading,
        n=state_dim,
        eps=eps,
    )


def expand_actuator_phase_boxes(
    H_act: np.ndarray,
    d_act: np.ndarray,
    lower_phase: np.ndarray,
    upper_phase: np.ndarray,
    p_eject_phase: np.ndarray,
    R_act: np.ndarray | None = None,
) -> tuple:
    """
    Expands compact actuator–phase geometry to a dense flat box layout.

    :param H_act: A np.array of shape ``[..., Na, 2, state_dim]``, actuator GT Jacobians.
    :param d_act: A np.array of shape ``[..., Na, 2]``, actuator GT offsets.
    :param lower_phase: A np.array of shape ``[..., Na, 3, 2]``, phase lower corners.
    :param upper_phase: A np.array of shape ``[..., Na, 3, 2]``, phase upper corners.
    :param p_eject_phase: A np.array of shape ``[..., Na, 3]``, phase ejection probabilities.
    :param R_act: None or a np.array of shape ``[..., Na, 2, 2]``, actuator noise covariances.

    :returns: A tuple ``(H, d, lower, upper, p_eject, R)`` with a flattened box axis
        of length ``Na * 3``.
    """
    H_act = np.asarray(H_act, dtype=float)
    d_act = np.asarray(d_act, dtype=float)
    lower_phase = np.asarray(lower_phase, dtype=float)
    upper_phase = np.asarray(upper_phase, dtype=float)
    p_eject_phase = np.asarray(p_eject_phase, dtype=float)

    n_a, n_m = lower_phase.shape[:2]
    H = np.repeat(H_act, n_m, axis=0)
    d = np.repeat(d_act, n_m, axis=0)
    lower = lower_phase.reshape(n_a * n_m, 2)
    upper = upper_phase.reshape(n_a * n_m, 2)
    p_eject = np.tile(p_eject_phase, n_a)
    if R_act is None:
        return H, d, lower, upper, p_eject
    R = np.repeat(np.asarray(R_act, dtype=float), n_m, axis=0)
    return H, d, lower, upper, p_eject, R


# ---------------------------------------------------------------------------
# Smoke / sanity checks (run as ``python -m adf_contact_model``)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.default_rng(0)

    # 1D known formula
    mass, mean, var = moments_1d_truncated_gaussian(0.0, 1.0, -1.0, 1.0)
    assert abs(mass - (norm.cdf(1) - norm.cdf(-1))) < 1e-12
    assert abs(mean) < 1e-12

    # Uncorrelated 2D: mass factorizes
    mean2 = np.array([0.0, 0.0])
    cov2 = np.diag([1.0, 4.0])
    lo = np.array([-1.0, -2.0])
    hi = np.array([1.0, 2.0])
    w, mu_t, P_t = moments_2d_rect_truncated_gaussian(mean2, cov2, lo, hi)
    w0, _, _ = moments_1d_truncated_gaussian(0.0, 1.0, -1.0, 1.0)
    w1, _, _ = moments_1d_truncated_gaussian(0.0, 4.0, -2.0, 2.0)
    assert abs(w - w0 * w1) < 1e-12
    assert abs(P_t[0, 1]) < 1e-12

    # Batch of identical boxes
    B = 5
    wB, muB, PB = moments_2d_rect_truncated_gaussian(
        np.broadcast_to(mean2, (B, 2)),
        np.broadcast_to(cov2, (B, 2, 2)),
        np.broadcast_to(lo, (B, 2)),
        np.broadcast_to(hi, (B, 2)),
    )
    assert np.allclose(wB, w)
    assert np.allclose(muB, mu_t)
    assert np.allclose(PB, P_t)

    # Correlated rectangle vs Monte Carlo
    cov_c = np.array([[1.0, 0.4], [0.4, 1.0]])
    w_c, mu_c, P_c = moments_2d_rect_truncated_gaussian(
        mean2, cov_c, np.array([-0.5, -0.5]), np.array([1.0, 1.5])
    )
    samples = rng.multivariate_normal(mean2, cov_c, size=200_000)
    mask = np.all(
        (samples >= np.array([-0.5, -0.5])) & (samples <= np.array([1.0, 1.5])),
        axis=1,
    )
    w_mc = mask.mean()
    mu_mc = samples[mask].mean(axis=0)
    P_mc = np.cov(samples[mask], rowvar=False)
    assert abs(w_c - w_mc) < 5e-3
    assert np.linalg.norm(mu_c - mu_mc) < 2e-2
    assert np.linalg.norm(P_c - P_mc) < 3e-2

    # Mixed uncorrelated / correlated batch
    mean_m = np.stack([mean2, mean2], axis=0)
    cov_m = np.stack([cov2, cov_c], axis=0)
    lo_m = np.stack([lo, np.array([-0.5, -0.5])], axis=0)
    hi_m = np.stack([hi, np.array([1.0, 1.5])], axis=0)
    w_m, _, _ = moments_2d_rect_truncated_gaussian(mean_m, cov_m, lo_m, hi_m)
    assert abs(w_m[0] - w) < 1e-12
    assert abs(w_m[1] - w_c) < 1e-12

    # End-to-end contact ADF: single identity box, p_eject = 1
    n = 2
    pi0, pi1 = 0.1, 0.9
    mu0 = np.zeros(n)
    P0 = np.eye(n)
    mu1 = np.array([0.2, -0.1])
    P1 = np.diag([1.0, 1.5])
    H = np.eye(2)[None, ...]
    d = np.zeros((1, 2))
    lower = np.array([[-0.5, -0.5]])
    upper = np.array([[0.5, 0.5]])
    p_eject = np.array([1.0])
    out = adf_contact_moment_matching(
        pi0, mu0, P0, pi1, mu1, P1, H, d, lower, upper, p_eject
    )
    assert np.isclose(out.pi.sum(), pi0 + pi1)
    assert out.pi[0] >= pi0 - 1e-12
    assert out.pi[1] <= pi1 + 1e-12

    # Masking: inactive boxes (degenerate time interval) are skipped
    H2 = np.tile(H, (2, 1, 1))
    d2 = np.tile(d, (2, 1))
    lower2 = np.vstack([lower, np.array([[0.0, -0.5]])])  # t_lo == t_hi → inactive
    upper2 = np.vstack([upper, np.array([[0.0, 0.5]])])
    p2 = np.array([1.0, 0.5])
    out2 = adf_contact_moment_matching(
        pi0, mu0, P0, pi1, mu1, P1, H2, d2, lower2, upper2, p2
    )
    assert np.allclose(out2.pi, out.pi, rtol=1e-10)
    mask = active_truncation_box_mask(lower2, upper2, p_eject=p2)
    assert mask.tolist() == [True, False]

    # Phased compact layout ↔ dense flat layout
    B, N, Na, dx = 2, 3, 2, 4
    H_act = rng.normal(size=(B, N, Na, 2, dx))
    d_act = rng.normal(size=(B, N, Na, 2))
    R_act = np.einsum("...ij,...kj->...ik", rng.normal(size=(B, N, Na, 2, 2)),
                      rng.normal(size=(B, N, Na, 2, 2))) * 0.01
    lower_ph = np.array(
        [
            [[[0.1, -0.5], [0.3, -0.5], [0.5, -0.5]],
             [[0.2, -0.4], [0.4, -0.4], [0.4, -0.4]]],  # last phase degenerate
            [[[0.1, -0.5], [0.1, -0.5], [0.6, -0.5]],
             [[0.15, -0.4], [0.35, -0.4], [0.55, -0.4]]],
        ],
        dtype=float,
    )
    upper_ph = np.array(
        [
            [[[0.25, 0.5], [0.45, 0.5], [0.7, 0.5]],
             [[0.35, 0.4], [0.55, 0.4], [0.4, 0.4]]],  # t_lo==t_hi
            [[[0.2, 0.5], [0.1, 0.5], [0.8, 0.5]],  # mid degenerate
             [[0.3, 0.4], [0.5, 0.4], [0.7, 0.4]]],
        ],
        dtype=float,
    )
    p_ph = np.full((B, N, Na, 3), 0.4)
    p_ph[:, :, :, 2] = 0.0  # one inactive phase by weight
    # densify for reference
    H_dense = np.repeat(H_act[..., None, :, :], 3, axis=3).reshape(B, N, Na * 3, 2, dx)
    d_dense = np.repeat(d_act[..., None, :], 3, axis=3).reshape(B, N, Na * 3, 2)
    R_dense = np.repeat(R_act[..., None, :, :], 3, axis=3).reshape(B, N, Na * 3, 2, 2)
    lo_dense = np.broadcast_to(lower_ph[:, None], (B, N, Na, 3, 2)).reshape(B, N, Na * 3, 2)
    up_dense = np.broadcast_to(upper_ph[:, None], (B, N, Na, 3, 2)).reshape(B, N, Na * 3, 2)
    p_dense = p_ph.reshape(B, N, Na * 3)
    mu0_b = rng.normal(size=(B, N, dx))
    P0_b = np.broadcast_to(np.eye(dx), (B, N, dx, dx)).copy()
    mu1_b = rng.normal(size=(B, N, dx))
    P1_b = P0_b + 0.1 * np.eye(dx)
    pi0_b = np.full((B, N), 0.2)
    pi1_b = np.full((B, N), 0.8)
    out_dense = adf_contact_moment_matching(
        pi0_b, mu0_b, P0_b, pi1_b, mu1_b, P1_b,
        H_dense, d_dense, lo_dense, up_dense, p_dense, R=R_dense,
    )
    out_ph = adf_contact_moment_matching(
        pi0_b, mu0_b, P0_b, pi1_b, mu1_b, P1_b,
        H_act, d_act, lower_ph, upper_ph, p_ph, R=R_act,
    )
    assert np.allclose(out_ph.pi, out_dense.pi, rtol=1e-10, atol=1e-12)
    assert np.allclose(out_ph.mu, out_dense.mu, rtol=1e-10, atol=1e-12)
    assert np.allclose(out_ph.P, out_dense.P, rtol=1e-10, atol=1e-12)

    # Vectorized rectangle CDF vs scipy reference (subset)
    n_cdf = 256
    mean_b = rng.normal(size=(n_cdf, 2))
    A = rng.normal(size=(n_cdf, 2, 2))
    cov_b = np.einsum("...ij,...kj->...ik", A, A) + np.eye(2) * 0.1
    lo_b = mean_b - rng.uniform(0.5, 2.0, size=(n_cdf, 2))
    up_b = mean_b + rng.uniform(0.5, 2.0, size=(n_cdf, 2))
    ref_cdf = np.empty(n_cdf)
    for i in range(n_cdf):
        sd = np.sqrt(np.maximum(np.diag(cov_b[i]), 1e-30))
        lo_i = np.where(np.isfinite(lo_b[i]), lo_b[i], mean_b[i] - 1e6 * sd)
        up_i = np.where(np.isfinite(up_b[i]), up_b[i], mean_b[i] + 1e6 * sd)
        ref_cdf[i] = (
                multivariate_normal.cdf(up_i, mean=mean_b[i], cov=cov_b[i])
                - multivariate_normal.cdf(
            np.array([lo_i[0], up_i[1]]), mean=mean_b[i], cov=cov_b[i]
        )
                - multivariate_normal.cdf(
            np.array([up_i[0], lo_i[1]]), mean=mean_b[i], cov=cov_b[i]
        )
                + multivariate_normal.cdf(lo_i, mean=mean_b[i], cov=cov_b[i])
        )
    est_cdf = bivariate_rect_cdf(mean_b, cov_b, lo_b, up_b)
    assert np.max(np.abs(est_cdf - ref_cdf)) < 1e-14

    print("self-test OK")
    print("  numba:", "enabled" if numba_enabled() else f"disabled (available={NUMBA_AVAILABLE})")
    if numba_enabled():
        warmup_adf_numba()
        print("  numba warmup OK")
    print("  pi =", out.pi)
    print("  mu[0] =", out.mu[0])
    print("  mu[1] =", out.mu[1])


if __name__ == "__main__":
    _self_test()
