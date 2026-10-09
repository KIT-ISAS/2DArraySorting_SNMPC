"""Batched rectangular CDF for bivariate Gaussians (Genz / optional approxcdf).

Used by ADF truncated-Gaussian moment matching for the correlated 2D path.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Backend: "genz" (default) | "approxcdf"
_BVN_RECT_CDF_BACKEND = "genz"

# Gauss–Legendre nodes for Genz bivariate-normal tail (Drezner/Genz rules).
_GL_W3 = np.array(
    [0.1713244923791705, 0.3607615730481384, 0.4679139345726904], dtype=float
)
_GL_X3 = np.array(
    [0.9324695142031522, 0.6612093864662647, 0.2386191860831970], dtype=float
)
_GL_W6 = np.array(
    [
        0.04717533638651177,
        0.1069393259953183,
        0.1600783285433464,
        0.2031674267230659,
        0.2334925365383547,
        0.2491470458134029,
    ],
    dtype=float,
)
_GL_X6 = np.array(
    [
        0.9815606342467191,
        0.9041172563704750,
        0.7699026741943050,
        0.5873179542866171,
        0.3678314989981802,
        0.1252334085114692,
    ],
    dtype=float,
)
_GL_W10 = np.array(
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
    ],
    dtype=float,
)
_GL_X10 = np.array(
    [
        0.9931285991850949,
        0.9639719272779138,
        0.9122344282513259,
        0.8391169718222188,
        0.7463319064601508,
        0.6360536807265150,
        0.5108670019508271,
        0.3737060887154196,
        0.2277858511416451,
        0.07652652113349733,
    ],
    dtype=float,
)
_GENZ_GL = {
    6: (np.concatenate([_GL_W3, _GL_W3]), np.concatenate([1.0 - _GL_X3, 1.0 + _GL_X3])),
    12: (np.concatenate([_GL_W6, _GL_W6]), np.concatenate([1.0 - _GL_X6, 1.0 + _GL_X6])),
    20: (np.concatenate([_GL_W10, _GL_W10]), np.concatenate([1.0 - _GL_X10, 1.0 + _GL_X10])),
}
_TWOPI = 2.0 * np.pi

_APPROXCDF_MVN_CDF = False  # False = not loaded; None = unavailable; callable = loaded


def bvn_rect_cdf_backend() -> str:
    """
    Returns the active backend name for :func:`bivariate_rect_cdf`.

    :returns: A string, either ``"genz"`` or ``"approxcdf"``.
    """
    return _BVN_RECT_CDF_BACKEND


def set_bvn_rect_cdf_backend(backend: str) -> None:
    """
    Switches the rectangular bivariate CDF backend at runtime.

    :param backend: A string, either ``"genz"`` or ``"approxcdf"``.
    """
    global _BVN_RECT_CDF_BACKEND
    backend = backend.strip().lower()
    if backend not in ("genz", "approxcdf"):
        raise ValueError("backend must be 'genz' or 'approxcdf'")
    _BVN_RECT_CDF_BACKEND = backend


def _approxcdf_mvn_cdf():
    """
    Lazily imports ``approxcdf.mvn_cdf`` (optional dependency).

    :returns: The callable ``approxcdf.mvn_cdf``.

    :raises ImportError: If the ``approxcdf`` package is not installed or not
        available.
    """
    global _APPROXCDF_MVN_CDF
    if _APPROXCDF_MVN_CDF is False:
        try:
            from approxcdf import mvn_cdf as _fn
        except ImportError as exc:
            _APPROXCDF_MVN_CDF = None
            raise ImportError(
                "approxcdf backend selected but package is not installed. "
                "Install with: pip install git+https://github.com/david-cortes/approxcdf.git"
            ) from exc
        _APPROXCDF_MVN_CDF = _fn
    if _APPROXCDF_MVN_CDF is None:
        raise ImportError("approxcdf is not available.")
    return _APPROXCDF_MVN_CDF


def _genz_tail_quadrature(hh: np.ndarray, kk: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """
    Correlated Genz tail for finite ``h, k, r`` (vectorized, adaptive GL order).

    :param hh: A np.array, finite standardized upper limits for the first coordinate.
    :param kk: A np.array, finite standardized upper limits for the second coordinate.
    :param rr: A np.array, correlations in ``(-1, 1)``, same shape as ``hh`` / ``kk``.

    :returns: A np.array of the same shape as ``hh``, the tail probabilities
        ``P(X>hh, Y>kk)`` clipped to ``[0, 1]``.
    """
    hs = (hh * hh + kk * kk) / 2.0
    hk = hh * kk
    asr = np.arcsin(rr) / 2.0
    abs_r = np.abs(rr)
    out = norm.cdf(-hh) * norm.cdf(-kk)

    for n_pts, (gl_w, gl_x) in _GENZ_GL.items():
        mask = (abs_r < {6: 0.3, 12: 0.75, 20: 1.01}[n_pts]) & (
            abs_r >= {6: 0.0, 12: 0.3, 20: 0.75}[n_pts]
        )
        if not np.any(mask):
            continue
        asr_m = asr[mask]
        sn = np.sin(asr_m[:, None] * gl_x)
        num = hk[mask, None] * sn - hs[mask, None]
        den = 1.0 - sn * sn
        integral = np.sum(np.exp(num / den) * gl_w, axis=1)
        out[mask] += integral * (asr_m / _TWOPI)
    return np.clip(out, 0.0, 1.0)


def genz_bivariate_normal_tail(
    h: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """
    Computes ``P(X > h, Y > k)`` for a standard bivariate normal with correlation ``r``.

    Uses vectorized Genz ``bvnl`` quadrature with adaptive 6/12/20-point rules.

    :param h: A np.array, upper limits for the first coordinate (may be ±inf).
    :param k: A np.array, upper limits for the second coordinate (may be ±inf).
    :param r: A np.array, correlation coefficients (clipped to ``[-1, 1]``).

    :returns: A np.array broadcast to the common shape of ``h``, ``k``, and ``r``,
        the bivariate normal tail probabilities.
    """
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    r = np.clip(np.asarray(r, dtype=float), -1.0, 1.0)
    h, k, r = np.broadcast_arrays(h, k, r)
    out = np.zeros(h.shape, dtype=float)

    pos_inf = (h == np.inf) | (k == np.inf)
    out[pos_inf] = 0.0

    both_neg_inf = (h == -np.inf) & (k == -np.inf) & ~pos_inf
    out[both_neg_inf] = 1.0

    h_neg = (h == -np.inf) & (k != -np.inf) & ~pos_inf
    out[h_neg] = norm.cdf(-k[h_neg])

    k_neg = (k == -np.inf) & (h != -np.inf) & ~pos_inf
    out[k_neg] = norm.cdf(-h[k_neg])

    indep = (r == 0.0) & ~both_neg_inf & ~h_neg & ~k_neg & ~pos_inf
    if np.any(indep):
        out[indep] = norm.cdf(-h[indep]) * norm.cdf(-k[indep])

    general = (
        ~pos_inf
        & ~both_neg_inf
        & ~h_neg
        & ~k_neg
        & ~indep
        & np.isfinite(h)
        & np.isfinite(k)
        & np.isfinite(r)
    )
    if np.any(general):
        out[general] = _genz_tail_quadrature(h[general], k[general], r[general])
    return out


def _bivariate_rect_cdf_genz(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Genz/Drezner batched rectangular CDF for bivariate Gaussians.

    :param mean: A np.array of shape ``[..., 2]``, the Gaussian means.
    :param cov: A np.array of shape ``[..., 2, 2]``, the Gaussian covariances.
    :param lower: A np.array of shape ``[..., 2]``, the lower rectangle bounds.
    :param upper: A np.array of shape ``[..., 2]``, the upper rectangle bounds.

    :returns: A np.array of shape ``[...]``, the probability mass of each rectangle.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mean, lower, upper = np.broadcast_arrays(mean, lower, upper)
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))
    else:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))

    s00 = cov[..., 0, 0]
    s11 = cov[..., 1, 1]
    s01 = cov[..., 0, 1]
    sig1 = np.sqrt(np.maximum(s00, 1e-30))
    sig2 = np.sqrt(np.maximum(s11, 1e-30))
    rho = np.clip(s01 / (sig1 * sig2), -1.0, 1.0)

    m0 = mean[..., 0]
    m1 = mean[..., 1]
    lo0 = lower[..., 0]
    lo1 = lower[..., 1]
    up0 = upper[..., 0]
    up1 = upper[..., 1]

    xl = np.where(np.isfinite(lo0), (lo0 - m0) / sig1, -np.inf)
    xu = np.where(np.isfinite(up0), (up0 - m0) / sig1, np.inf)
    yl = np.where(np.isfinite(lo1), (lo1 - m1) / sig2, -np.inf)
    yu = np.where(np.isfinite(up1), (up1 - m1) / sig2, np.inf)

    out = (
        genz_bivariate_normal_tail(xl, yl, rho)
        - genz_bivariate_normal_tail(xu, yl, rho)
        - genz_bivariate_normal_tail(xl, yu, rho)
        + genz_bivariate_normal_tail(xu, yu, rho)
    )
    out = np.clip(out, 0.0, 1.0)
    degenerate = (lo0 >= up0) | (lo1 >= up1)
    return np.where(degenerate, 0.0, out)


def _finite_bounds_for_approxcdf(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    clip_sigma: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clips infinite rectangle bounds to wide finite limits for ``approxcdf``.

    :param mean: A np.array of shape ``[..., 2]``, the Gaussian means.
    :param cov: A np.array of shape ``[..., 2, 2]``, the Gaussian covariances.
    :param lower: A np.array of shape ``[..., 2]``, possibly infinite lower bounds.
    :param upper: A np.array of shape ``[..., 2]``, possibly infinite upper bounds.
    :param clip_sigma: A float, the multiple of the marginal standard deviation used
        when replacing ``±inf`` with a finite clip.

    :returns: A tuple ``(lower_finite, upper_finite)`` of np.arrays with the same
        shapes as ``lower`` / ``upper``, with infinite entries replaced by finite clips.
    """
    sd = np.sqrt(np.maximum(np.diagonal(cov), 1e-30))
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    m = np.asarray(mean, dtype=float)
    lo = np.where(np.isfinite(lo), lo, m - clip_sigma * sd)
    up = np.where(np.isfinite(up), up, m + clip_sigma * sd)
    return lo, up


def _bivariate_rect_cdf_approxcdf_one(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Rectangular CDF for a single bivariate Gaussian via ``approxcdf``.

    :param mean: A np.array of shape ``(2,)``, the Gaussian mean.
    :param cov: A np.array of shape ``(2, 2)``, the Gaussian covariance.
    :param lower: A np.array of shape ``(2,)``, the lower rectangle bounds.
    :param upper: A np.array of shape ``(2,)``, the upper rectangle bounds.

    :returns: A float, the probability mass of the rectangle.
    """
    mvn_cdf = _approxcdf_mvn_cdf()
    mean = np.asarray(mean, dtype=float)
    cov = np.ascontiguousarray(cov, dtype=np.float64)
    lo, up = _finite_bounds_for_approxcdf(mean, cov, lower, upper)
    p = (
        mvn_cdf(up, cov, mean=mean)
        - mvn_cdf(np.array([lo[0], up[1]], dtype=np.float64), cov, mean=mean)
        - mvn_cdf(np.array([up[0], lo[1]], dtype=np.float64), cov, mean=mean)
        + mvn_cdf(lo, cov, mean=mean)
    )
    return float(np.clip(p, 0.0, 1.0))


def _bivariate_rect_cdf_approxcdf(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Batched rectangular CDF via the optional ``approxcdf`` backend (scalar loop).

    :param mean: A np.array of shape ``[..., 2]``, the Gaussian means.
    :param cov: A np.array of shape ``[..., 2, 2]``, the Gaussian covariances.
    :param lower: A np.array of shape ``[..., 2]``, the lower rectangle bounds.
    :param upper: A np.array of shape ``[..., 2]``, the upper rectangle bounds.

    :returns: A np.array of shape ``[...]``, the probability mass of each rectangle.
    """
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mean, lower, upper = np.broadcast_arrays(mean, lower, upper)
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))
    else:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))

    batch_shape = mean.shape[:-1]
    flat = int(np.prod(batch_shape)) if batch_shape else 1
    mean_f = mean.reshape(flat, 2)
    cov_f = cov.reshape(flat, 2, 2)
    lo_f = lower.reshape(flat, 2)
    up_f = upper.reshape(flat, 2)

    out = np.empty(flat, dtype=float)
    for i in range(flat):
        lo0, lo1 = lo_f[i, 0], lo_f[i, 1]
        up0, up1 = up_f[i, 0], up_f[i, 1]
        if (lo0 >= up0) or (lo1 >= up1):
            out[i] = 0.0
            continue
        out[i] = _bivariate_rect_cdf_approxcdf_one(mean_f[i], cov_f[i], lo_f[i], up_f[i])
    return out.reshape(batch_shape)


def bivariate_rect_cdf(
    mean: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Computes ``P(lower ≤ X ≤ upper)`` for a batch of bivariate Gaussians.

    Backend (PTCR config ``adf.bvn_rect_cdf_backend`` or :func:`set_bvn_rect_cdf_backend`):

    * ``genz`` (default) — vectorized Genz/Drezner quadrature, batched.
      Uses Numba when available (``adf.use_numba`` / :func:`~adf.adf_numba_kernels.numba_enabled`).
    * ``approxcdf`` — optional C++ package `approxcdf`_, scalar loop.

    .. _approxcdf: https://github.com/david-cortes/approxcdf

    :param mean: A np.array of shape ``[..., 2]``, the Gaussian means.
    :param cov: A np.array of shape ``[..., 2, 2]``, the Gaussian covariances.
    :param lower: A np.array of shape ``[..., 2]``, the inclusive lower rectangle bounds.
    :param upper: A np.array of shape ``[..., 2]``, the inclusive upper rectangle bounds.

    :returns: A np.array of shape ``[...]``, the rectangular CDF mass for each
        bivariate Gaussian in the batch.
    """
    if _BVN_RECT_CDF_BACKEND == "approxcdf":
        return _bivariate_rect_cdf_approxcdf(mean, cov, lower, upper)

    try:
        from adf.adf_numba_kernels import bivariate_rect_cdf_numba, numba_enabled
    except ImportError:
        return _bivariate_rect_cdf_genz(mean, cov, lower, upper)

    if not numba_enabled():
        return _bivariate_rect_cdf_genz(mean, cov, lower, upper)

    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mean, lower, upper = np.broadcast_arrays(mean, lower, upper)
    if cov.ndim == 2:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))
    else:
        cov = np.broadcast_to(cov, mean.shape[:-1] + (2, 2))
    batch_shape = mean.shape[:-1]
    flat = int(np.prod(batch_shape)) if batch_shape else 1
    out = bivariate_rect_cdf_numba(
        np.ascontiguousarray(mean.reshape(flat, 2), dtype=np.float64),
        np.ascontiguousarray(cov.reshape(flat, 2, 2), dtype=np.float64),
        np.ascontiguousarray(lower.reshape(flat, 2), dtype=np.float64),
        np.ascontiguousarray(upper.reshape(flat, 2), dtype=np.float64),
    )
    return out.reshape(batch_shape)
