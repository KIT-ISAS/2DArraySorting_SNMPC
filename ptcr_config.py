"""Normalize nested PTCR controller JSON (``adf``, ``tree_search``) and apply runtime flags."""

from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_adf_config(adf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalizes the nested ``adf`` controller-config block.

    Expected shape::

        {
          "use_numba": true,
          "bvn_rect_cdf_backend": "genz",
          "mc_comparison": { ... }  # optional; see AdfMcComparisonHooks
        }

    :param adf: None or a dict with ADF-related settings.

    :returns: A dict with keys ``use_numba``, ``bvn_rect_cdf_backend``,
        and ``mc_comparison`` (None or nested dict).
    """
    cfg = dict(adf or {})
    return {
        "use_numba": bool(cfg.pop("use_numba", True)),
        "bvn_rect_cdf_backend": str(cfg.pop("bvn_rect_cdf_backend", "genz")),
        "mc_comparison": cfg.pop("mc_comparison", None),
    }


def normalize_tree_search_config(
    tree_search: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalizes the nested ``tree_search`` controller-config block.

    :param tree_search: None or a dict with ``use_fast`` and ``use_numba`` Booleans.

    :returns: A dict with keys ``use_fast`` and ``use_numba`` (both Booleans).
    """
    cfg = dict(tree_search or {})
    return {
        "use_fast": bool(cfg.pop("use_fast", True)),
        "use_numba": bool(cfg.pop("use_numba", True)),
    }


def apply_adf_runtime_settings(adf_cfg: Dict[str, Any]) -> None:
    """Applies ADF module toggles from a normalized ``adf`` config dict.

    :param adf_cfg: Output of :func:`normalize_adf_config`.
    """
    from adf.adf_bivariate_rect_cdf import set_bvn_rect_cdf_backend
    from adf.adf_numba_kernels import set_numba_enabled

    set_numba_enabled(adf_cfg["use_numba"])
    set_bvn_rect_cdf_backend(adf_cfg["bvn_rect_cdf_backend"])


def apply_tree_search_runtime_settings(tree_cfg: Dict[str, Any]) -> None:
    """Applies tree-search module toggles from a normalized ``tree_search`` config dict.

    :param tree_cfg: Output of :func:`normalize_tree_search_config`.
    """
    from ptcr_tree_search import set_tree_numba_enabled

    set_tree_numba_enabled(tree_cfg["use_numba"])


def pop_ptcr_nested_config(controller_config: Dict[str, Any]) -> Dict[str, Any]:
    """Pops ``adf`` and ``tree_search`` from a controller config dict in place.

    :param controller_config: Mutable controller JSON dict (modified in place).

    :returns: A dict with normalized ``adf`` and ``tree_search`` sub-dicts.
    """
    adf = normalize_adf_config(controller_config.pop("adf", None))
    tree_search = normalize_tree_search_config(controller_config.pop("tree_search", None))
    return {"adf": adf, "tree_search": tree_search}
