"""ADF vs. Monte-Carlo validation plots (live PTCR hooks + CSV export)."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from adf.adf_gauss_taylor import lateral_index


def _gaussian_pdf_1d(x, mean, var, mass=1.0):
    """Evaluates a 1D Gaussian density (optionally scaled by a mode mass).

    :param x: A float or np.array, the evaluation points.
    :param mean: A float, the Gaussian mean.
    :param var: A float, the Gaussian variance (clipped from below for stability).
    :param mass: A float, multiplicative mode weight (e.g. existence probability).

    :returns: The density values with the same shape as ``x``.
    """
    std = np.sqrt(max(float(var), 1e-18))
    z = (x - float(mean)) / std
    return mass * np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi))


def _finite_moments(
    mu_adf: np.ndarray, P_adf: np.ndarray, mode: int, state_comp: int = 0
) -> bool:
    """Whether ADF mean/variance for one mode and state component are usable for plotting.

    :param mu_adf: A np.array of shape ``[2, state_dim]``, dual-mode means.
    :param P_adf: A np.array of shape ``[2, state_dim, state_dim]``, dual-mode covariances.
    :param mode: An integer, existence mode index (0 = dead, 1 = living).
    :param state_comp: An integer, state-component index (e.g. 0 for streamwise ``x``).

    :returns: A Boolean, ``True`` if mean and variance are finite and variance is non-negative.
    """
    return bool(
        np.isfinite(mu_adf[mode, state_comp])
        and np.isfinite(P_adf[mode, state_comp, state_comp])
        and float(P_adf[mode, state_comp, state_comp]) >= 0.0
    )


def _draw_stage_panel(
    ax,
    *,
    ms: np.ndarray,
    ex: np.ndarray,
    pi_adf: np.ndarray,
    mu_adf: np.ndarray,
    P_adf: np.ndarray,
    particle_index: int,
    stage_index: int,
    state_comp: int = 0,
    coord_label: str = "$x$",
    pi_mc: Optional[np.ndarray] = None,
    bins: Optional[int] = None,
    show_legend: bool = True,
    title: Optional[str] = None,
):
    """Draws one stage panel (MC histograms + ADF overlays) into a matplotlib axis.

    :param ax: A matplotlib Axes to draw into.
    :param ms: A np.array of MC motion-state samples with a particle axis.
    :param ex: A np.array of MC existence indicators matching ``ms``.
    :param pi_adf: A np.array of ADF dual-mode existence masses for the particle.
    :param mu_adf: A np.array of ADF dual-mode means for the particle.
    :param P_adf: A np.array of ADF dual-mode covariances for the particle.
    :param particle_index: An integer, the particle to plot.
    :param stage_index: An integer, the stage index used in titles.
    :param state_comp: An integer, the motion-state component to histogram.
    :param coord_label: A string, the axis label for that component.
    :param pi_mc: None or a np.array of empirical MC existence masses for labels.
    :param bins: None or an integer, the histogram bin count (auto if None).
    :param show_legend: A Boolean, whether to draw the legend.
    :param title: None or a string, optional panel title override.

    :returns: None.
    """
    vals_dead = ms[ex[:, particle_index, 0] == 1, particle_index, state_comp]
    vals_live = ms[ex[:, particle_index, 1] == 1, particle_index, state_comp]

    def _n_bins(n: int) -> int:
        if bins is not None:
            return int(bins)
        if n <= 1:
            return 1
        return int(max(8, min(30, round(np.sqrt(n)))))

    hist_ymax = 0.0
    pi_mc_arr = None if pi_mc is None else np.asarray(pi_mc, dtype=float)

    def _mc_label(mode: int, n: int) -> str:
        if pi_mc_arr is not None and pi_mc_arr.shape[-1] >= 2:
            return rf"MC ex={mode} (n={n}, $\tilde\pi$={pi_mc_arr[mode]:.3f})"
        return f"MC ex={mode} (n={n})"

    if vals_dead.size:
        counts, _edges, _ = ax.hist(
            vals_dead,
            bins=_n_bins(vals_dead.size),
            density=True,
            alpha=0.35,
            color="C3",
            label=_mc_label(0, int(vals_dead.size)) if show_legend else None,
            zorder=1,
        )
        if len(counts):
            hist_ymax = max(hist_ymax, float(np.max(counts)))
    if vals_live.size:
        counts, _edges, _ = ax.hist(
            vals_live,
            bins=_n_bins(vals_live.size),
            density=True,
            alpha=0.35,
            color="C0",
            label=_mc_label(1, int(vals_live.size)) if show_legend else None,
            zorder=1,
        )
        if len(counts):
            hist_ymax = max(hist_ymax, float(np.max(counts)))

    x_parts = [a for a in (vals_dead, vals_live) if a.size]
    for mode in (0, 1):
        if _finite_moments(mu_adf, P_adf, mode, state_comp):
            sig = np.sqrt(max(float(P_adf[mode, state_comp, state_comp]), 1e-18))
            mean = float(mu_adf[mode, state_comp])
            x_parts.append(np.array([mean - 4.0 * sig, mean + 4.0 * sig]))
    if not x_parts:
        x_parts = [np.array([0.0, 1.0])]
    x_lo = float(np.min([np.min(a) for a in x_parts]))
    x_hi = float(np.max([np.max(a) for a in x_parts]))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_lo, x_hi = -1.0, 1.0
    xs = np.linspace(x_lo, x_hi, 400)

    styles = {
        0: ("C3", "-", rf"ADF ex=0 ($\tilde\pi$={pi_adf[0]:.3f})"),
        1: ("C0", "--", rf"ADF ex=1 ($\tilde\pi$={pi_adf[1]:.3f})"),
    }
    adf_ymax = 0.0
    for mode in (0, 1):
        if not _finite_moments(mu_adf, P_adf, mode, state_comp):
            continue
        y = _gaussian_pdf_1d(
            xs,
            mu_adf[mode, state_comp],
            P_adf[mode, state_comp, state_comp],
            mass=1.0,
        )
        adf_ymax = max(adf_ymax, float(np.max(y)))
        c, ls, lab = styles[mode]
        ax.plot(
            xs,
            y,
            color=c,
            lw=2.5,
            ls=ls,
            zorder=5,
            label=lab if show_legend else None,
        )

    if adf_ymax > 0 and hist_ymax > 3.0 * adf_ymax:
        ax.set_ylim(0.0, 1.15 * max(adf_ymax * 1.25, hist_ymax * 0.35))

    if title is None:
        title = f"stage {stage_index}"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(coord_label)
    if show_legend:
        ax.legend(loc="best", fontsize=7)


def _position_components(state_dim: int) -> list[Tuple[int, str]]:
    """Returns streamwise ``x`` and, if present, lateral ``y`` state indices with labels.

    :param state_dim: An integer, the motion-state dimension.

    :returns: A list of ``(component_index, latex_label)`` pairs.
    """
    comps: list[Tuple[int, str]] = [(0, "$x$")]
    y_idx = lateral_index(int(state_dim))
    if y_idx is not None:
        comps.append((int(y_idx), "$y$"))
    return comps


def _finalize_fig(fig, save_path: Optional[str], show: bool):
    """Finalizes a matplotlib figure: layout, optional save, show or close.

    :param fig: A matplotlib Figure.
    :param save_path: None or a string, path to write a PNG; ``None`` skips saving.
    :param show: A Boolean, if ``True`` call ``plt.show()``, else close the figure.
    """
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def partial_hit_stage_window(
    stage_k: int,
    n_stages_total: int,
    *,
    n_after: int = 4,
) -> Optional[list[int]]:
    """Builds stages ``[k-1, k, ..., k+n_after-1]`` if the horizon is long enough.

    ``stage_k`` is the partial-hit contact stage from
    :func:`find_partial_hit_particle`.

    :param stage_k: An integer, the contact stage index (must be ``>= 1``).
    :param n_stages_total: An integer, number of stages in ``pi_stages`` (``N+1``).
    :param n_after: An integer, number of stages after ``k-1`` to include (default 4
        yields ``[k-1, k, k+1, k+2, k+3]``).

    :returns: A list of stage indices, or ``None`` if ``k < 1`` or the window would
        exceed ``0 .. n_stages_total-1``.
    """
    k = int(stage_k)
    n = int(n_stages_total)
    n_after = int(n_after)
    if k < 1 or n_after < 1:
        return None
    stages = [k - 1 + i for i in range(1 + n_after)]
    if stages[0] < 0 or stages[-1] >= n:
        return None
    return stages


def summarize_mc_eject_actor_counts(
    mc_diag: dict,
    particle_index: int,
    stage_k: int,
) -> Optional[dict]:
    """Histograms ``first_contact_actor`` among MC samples ejected at ``stage_k``.

    Uses samples that were living at ``stage_k-1`` and dead at ``stage_k``.

    :param mc_diag: A dict from :func:`forward_chain_mc` / :meth:`_forward_chain_mc`
        with keys ``ex_stages`` and ``first_contact_actors_stages``.
    :param particle_index: An integer, particle index into the sample arrays.
    :param stage_k: An integer, contact stage at which ejection is attributed (``>= 1``).

    :returns: ``None`` if diagnostics lack the required fields or the stage is out of
        range; otherwise a dict with ``mc_eject_actor_counts``
        (``{actor_idx_str: count}``, ``"-1"`` = no box hit) and ``n_ejected``.
    """
    stages = mc_diag.get("first_contact_actors_stages")
    ex_stages = mc_diag.get("ex_stages")
    if stages is None or ex_stages is None:
        return None
    k = int(stage_k)
    p = int(particle_index)
    if k < 1 or k >= len(ex_stages) or k >= len(stages):
        return None
    ex_prev = np.asarray(ex_stages[k - 1])
    ex_cur = np.asarray(ex_stages[k])
    if ex_prev.ndim != 3 or ex_cur.ndim != 3:
        return None
    if p >= ex_prev.shape[1]:
        return None
    was_living = ex_prev[:, p, 1].astype(bool)
    now_dead = ex_cur[:, p, 0].astype(bool)
    ejected = was_living & now_dead
    n_ejected = int(np.count_nonzero(ejected))
    actors = np.asarray(stages[k])[:, p]
    if actors.shape[0] != ejected.shape[0]:
        return None
    counts: dict[str, int] = {}
    if n_ejected > 0:
        vals, cnts = np.unique(actors[ejected], return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            counts[str(int(v))] = int(c)
    return {"mc_eject_actor_counts": counts, "n_ejected": n_ejected}


def summarize_mc_eject_actor_counts_from_stage(
    mc_diag: dict,
    particle_index: int,
    stage_k: int,
    *,
    stage_indices: Optional[Sequence[int]] = None,
) -> Optional[dict]:
    """Per-stage eject-actor histograms from ``stage_k`` onward.

    Includes ``stage_k`` and later contact stages (samples that survive a partial
    hit can contact further actuators). If ``stage_indices`` is given, only those
    stages with index ``>= stage_k`` are reported.

    :param mc_diag: A dict from :func:`forward_chain_mc` / :meth:`_forward_chain_mc`
        with keys ``ex_stages`` and ``first_contact_actors_stages``.
    :param particle_index: An integer, particle index into the sample arrays.
    :param stage_k: An integer, first contact stage to include (``>= 1``).
    :param stage_indices: None or a sequence of stage indices restricting which
        stages (with index ``>= stage_k``) are reported. ``None`` uses all stages
        from ``stage_k`` to the end of the MC horizon.

    :returns: ``None`` if no stage could be summarized; otherwise a dict with
        ``mc_eject_actor_counts_by_stage``, ``n_ejected_by_stage``, and for
        compatibility ``mc_eject_actor_counts`` / ``n_ejected`` for ``stage_k``
        (same fields as :func:`summarize_mc_eject_actor_counts`).
    """
    stages = mc_diag.get("first_contact_actors_stages")
    ex_stages = mc_diag.get("ex_stages")
    if stages is None or ex_stages is None:
        return None
    k0 = int(stage_k)
    if k0 < 1:
        return None
    if stage_indices is None:
        contact_stages = list(range(k0, len(ex_stages)))
    else:
        contact_stages = sorted(
            {int(s) for s in stage_indices if int(s) >= k0}
        )
    by_stage_counts: dict[str, dict[str, int]] = {}
    by_stage_n: dict[str, int] = {}
    for k in contact_stages:
        one = summarize_mc_eject_actor_counts(mc_diag, particle_index, k)
        if one is None:
            continue
        key = str(int(k))
        by_stage_counts[key] = one["mc_eject_actor_counts"]
        by_stage_n[key] = one["n_ejected"]
    if not by_stage_counts:
        return None
    out: dict = {
        "mc_eject_actor_counts_by_stage": by_stage_counts,
        "n_ejected_by_stage": by_stage_n,
    }
    # Compat fields for the primary partial-hit stage.
    k0s = str(k0)
    if k0s in by_stage_counts:
        out["mc_eject_actor_counts"] = by_stage_counts[k0s]
        out["n_ejected"] = by_stage_n[k0s]
    return out

def export_adf_mc_position_csvs(
    out_dir: str,
    adf_states,
    mc_diag,
    *,
    particle_index: int,
    stage_indices: Sequence[int],
    meta: Optional[dict] = None,
) -> str:
    """Writes MC samples + ADF ``(μ, σ)`` + ADF/MC ``π`` + PDF curves for paper TikZ.

    Layout under ``out_dir``:

    - ``samples_s{stage}_{x|y}_ex{0|1}.csv`` — column ``pos``
    - ``adf_pdf_s{stage}_{coord}_ex{mode}.csv`` — ``pos,dens`` (ADF Gaussian)
    - ``limits_s{stage}_{coord}.csv`` — ``xmin,xmax,ymax``
    - ``pi_s{stage}.csv`` — ``pi_adf_1,pi_mc_1`` (living-mode only)
    - ``nstar.csv`` — ``nstar`` (partial-hit stage index)
    - ``adf_params.csv``, ``pi_stages.csv``, ``axis_limits.csv``, ``meta.json``

    :param out_dir: A string, output directory (created if missing).
    :param adf_states: A dict from :meth:`_forward_chain_states` with
        ``pi_stages``, ``mu_stages``, ``P_stages``.
    :param mc_diag: A dict from :meth:`_forward_chain_mc` with ``ms_stages``,
        ``ex_stages``, ``pi_stages``.
    :param particle_index: An integer, particle index to export.
    :param stage_indices: A sequence of integers, stages to write.
    :param meta: None or a dict merged into ``meta.json`` (e.g. partial-hit info,
        eject-actor counts).

    :returns: The string ``out_dir``.
    """
    import json

    os.makedirs(out_dir, exist_ok=True)
    stage_indices = list(stage_indices)
    p = int(particle_index)
    state_dim = int(mc_diag["ms_stages"][stage_indices[0]].shape[-1])
    comps = _position_components(state_dim)
    coord_names = {0: "x"}
    for comp, label in comps:
        if label.strip("$") == "y":
            coord_names[int(comp)] = "y"

    param_rows: list[str] = ["stage,coord,mode,mu,sigma,pi_adf"]
    pi_rows: list[str] = ["stage,pi_adf_0,pi_adf_1,pi_mc_0,pi_mc_1"]
    for k in stage_indices:
        ms = mc_diag["ms_stages"][k]
        ex = mc_diag["ex_stages"][k]
        pi_adf = np.asarray(adf_states["pi_stages"][k][0, p], dtype=float)
        pi_mc = np.asarray(mc_diag["pi_stages"][k][p], dtype=float)
        pi_rows.append(
            f"{int(k)},{float(pi_adf[0]):.10g},{float(pi_adf[1]):.10g},"
            f"{float(pi_mc[0]):.10g},{float(pi_mc[1]):.10g}"
        )
        with open(
            os.path.join(out_dir, f"pi_s{int(k)}.csv"), "w", encoding="utf-8"
        ) as f:
            f.write("pi_adf_1,pi_mc_1\n")
            f.write(f"{float(pi_adf[1]):.10g},{float(pi_mc[1]):.10g}\n")
        mu_adf = np.asarray(adf_states["mu_stages"][k][0, p], dtype=float)
        P_adf = np.asarray(adf_states["P_stages"][k][0, p], dtype=float)
        for comp, label in comps:
            cname = coord_names.get(int(comp), label.strip("$"))
            for mode in (0, 1):
                mask = ex[:, p, mode] == 1
                vals = ms[mask, p, comp]
                path = os.path.join(
                    out_dir, f"samples_s{int(k)}_{cname}_ex{mode}.csv"
                )
                with open(path, "w", encoding="utf-8") as f:
                    f.write("pos\n")
                    for v in np.asarray(vals, dtype=float).ravel():
                        if np.isfinite(v):
                            f.write(f"{float(v):.10g}\n")
                if _finite_moments(mu_adf, P_adf, mode, comp):
                    var = float(P_adf[mode, comp, comp])
                    sigma = float(np.sqrt(max(var, 1e-18)))
                    mu = float(mu_adf[mode, comp])
                    param_rows.append(
                        f"{int(k)},{cname},{mode},{mu:.10g},{sigma:.10g},"
                        f"{float(pi_adf[mode]):.10g}"
                    )

    params_path = os.path.join(out_dir, "adf_params.csv")
    with open(params_path, "w", encoding="utf-8") as f:
        f.write("\n".join(param_rows) + "\n")
    with open(os.path.join(out_dir, "pi_stages.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(pi_rows) + "\n")

    from collections import defaultdict

    by_panel: dict[tuple[str, str], list[tuple[int, float, float, float]]] = defaultdict(
        list
    )
    for row in param_rows[1:]:
        stage_s, cname, mode_s, mu_s, sigma_s, pi_s = row.split(",")
        by_panel[(stage_s, cname)].append(
            (int(mode_s), float(mu_s), float(sigma_s), float(pi_s))
        )
    pi_by_stage = {
        r.split(",")[0]: r.split(",")[1:]
        for r in pi_rows[1:]
    }
    limit_rows = ["stage,coord,xmin,xmax,ymax"]
    for (stage_s, cname), items in sorted(by_panel.items()):
        xs: list[np.ndarray] = []
        adf_peak = 0.0
        sample_by_mode: dict[int, np.ndarray] = {}
        for mode, mu, sigma, _pi in items:
            sp = os.path.join(out_dir, f"samples_s{stage_s}_{cname}_ex{mode}.csv")
            if os.path.isfile(sp):
                raw = np.loadtxt(sp, delimiter=",", skiprows=1)
                if np.size(raw):
                    arr = np.atleast_1d(np.asarray(raw, dtype=float))
                    xs.append(arr)
                    sample_by_mode[int(mode)] = arr
            xs.append(np.array([mu - 4.0 * sigma, mu + 4.0 * sigma], dtype=float))
            adf_peak = max(adf_peak, 1.0 / (sigma * np.sqrt(2.0 * np.pi)))
            n_pdf = 120
            lo_g, hi_g = mu - 4.0 * sigma, mu + 4.0 * sigma
            xs_pdf = np.linspace(lo_g, hi_g, n_pdf)
            dens = _gaussian_pdf_1d(xs_pdf, mu, sigma * sigma, mass=1.0)
            pdf_path = os.path.join(
                out_dir, f"adf_pdf_s{stage_s}_{cname}_ex{mode}.csv"
            )
            with open(pdf_path, "w", encoding="utf-8") as f:
                f.write("pos,dens\n")
                for x, y in zip(xs_pdf, dens):
                    f.write(f"{float(x):.10g},{float(y):.10g}\n")
        if not xs:
            continue
        lo = float(np.min([a.min() for a in xs]))
        hi = float(np.max([a.max() for a in xs]))
        pad = 0.02 * (hi - lo + 1e-9)
        xmin, xmax = lo - pad, hi + pad
        hist_peak = 0.0
        for arr in sample_by_mode.values():
            if arr.size == 0 or xmax <= xmin:
                continue
            counts, _ = np.histogram(arr, bins=18, range=(xmin, xmax), density=True)
            if len(counts):
                hist_peak = max(hist_peak, float(np.max(counts)))
        ymax = 1.12 * max(hist_peak, adf_peak, 1e-6)
        limit_rows.append(
            f"{stage_s},{cname},{xmin:.10g},{xmax:.10g},{ymax:.10g}"
        )
        with open(
            os.path.join(out_dir, f"limits_s{stage_s}_{cname}.csv"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("xmin,xmax,ymax\n")
            f.write(f"{xmin:.10g},{xmax:.10g},{ymax:.10g}\n")
    with open(os.path.join(out_dir, "axis_limits.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(limit_rows) + "\n")

    nstar = None
    if meta and "partial_hit_stage" in meta:
        nstar = int(meta["partial_hit_stage"])
    if nstar is None and len(stage_indices) >= 2:
        nstar = int(stage_indices[1])  # window is [k-1, k, ...]
    if nstar is not None:
        with open(os.path.join(out_dir, "nstar.csv"), "w", encoding="utf-8") as f:
            f.write("nstar\n")
            f.write(f"{int(nstar)}\n")

    meta_out = dict(meta or {})
    meta_out.setdefault("particle_index", p)
    meta_out.setdefault("stage_indices", [int(s) for s in stage_indices])
    if nstar is not None:
        meta_out.setdefault("partial_hit_stage", int(nstar))
    if pi_by_stage:
        meta_out["pi_stages"] = {
            s: {
                "pi_adf": [float(v[0]), float(v[1])],
                "pi_mc": [float(v[2]), float(v[3])],
            }
            for s, v in pi_by_stage.items()
        }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_dir


def find_partial_hit_particle(
    pi_stages: Sequence[np.ndarray],
    particle_class: Optional[np.ndarray] = None,
    *,
    pi_lo: float = 0.3,
    pi_hi: float = 0.7,
    prefer_reject: bool = True,
    n_associated_meas: Optional[np.ndarray] = None,
    min_n_associated_meas: int = 0,
) -> Optional[Tuple[int, int, float]]:
    """Picks a particle with mid-range existence after at least one contact stage.

    Scans ``pi_stages[1:]`` (post-contact; stage 0 is the prior). A particle
    qualifies if some stage has living mass ``π̃⁽¹⁾ ∈ (pi_lo, pi_hi)`` (equivalently
    for dead mass, since the two sum to one).

    :param pi_stages: A sequence of length ``N+1`` with arrays of shape
        ``[batch, num_particles, 2]`` or ``[num_particles, 2]``.
    :param particle_class: None or an integer np.array of shape ``[num_particles]``
        with ``1`` = reject. When ``prefer_reject`` and any reject qualifies, accept
        particles are ignored.
    :param pi_lo: A float in ``[0, 1]``, lower open bound for living existence.
    :param pi_hi: A float in ``[0, 1]``, upper open bound for living existence
        (must satisfy ``pi_lo < pi_hi``).
    :param prefer_reject: A Boolean, prefer reject-class particles when available.
    :param n_associated_meas: None or a np.array of shape ``[num_particles]``, count
        of associated MTT measurements per track. When given with
        ``min_n_associated_meas > 0``, only tracks with at least that many
        associations are considered (tracker near saturation).
    :param min_n_associated_meas: An integer, minimum associations required when
        ``n_associated_meas`` is provided; ``0`` disables the age gate.

    :returns: None, or a tuple ``(particle_index, stage_index, pi_live)`` where
    ``particle_index`` / ``stage_index`` are integers and ``pi_live`` is a float,
    for the qualifying stage
        whose ``π̃⁽¹⁾`` is closest to ``0.5``, or ``None`` if none qualify.

    :raises ValueError: If ``pi_lo`` / ``pi_hi`` are not a valid open interval in
        ``[0, 1]``, or a stage array has an unexpected shape.
    """
    if len(pi_stages) <= 1:
        return None
    pi_lo = float(pi_lo)
    pi_hi = float(pi_hi)
    if not (0.0 <= pi_lo < pi_hi <= 1.0):
        raise ValueError(f"Need 0 <= pi_lo < pi_hi <= 1, got {pi_lo}, {pi_hi}")
    min_n = int(min_n_associated_meas)
    n_meas_arr = None if n_associated_meas is None else np.asarray(n_associated_meas)

    best: Optional[Tuple[float, int, int, float]] = None  # (dist, p, k, pi1)

    def _age_ok(p: int) -> bool:
        if min_n <= 0 or n_meas_arr is None:
            return True
        if p >= n_meas_arr.shape[0]:
            return False
        return int(n_meas_arr[p]) >= min_n

    def _consider(p: int, k: int, pi1: float) -> None:
        nonlocal best
        if not _age_ok(p):
            return
        if not (pi_lo < pi1 < pi_hi):
            return
        dist = abs(pi1 - 0.5)
        cand = (dist, p, k, pi1)
        if best is None or cand < best:
            best = cand

    class_arr = None if particle_class is None else np.asarray(particle_class, dtype=int)

    for k in range(1, len(pi_stages)):
        pi_k = np.asarray(pi_stages[k], dtype=float)
        if pi_k.ndim == 3:
            pi_k = pi_k[0]
        if pi_k.ndim != 2 or pi_k.shape[-1] != 2:
            raise ValueError(f"pi_stages[{k}] has unexpected shape {pi_k.shape}")
        num_particles = pi_k.shape[0]
        for p in range(num_particles):
            if prefer_reject and class_arr is not None and class_arr.shape[0] == num_particles:
                if int(class_arr[p]) != 1:
                    continue
            _consider(p, k, float(pi_k[p, 1]))

    if best is None and prefer_reject and class_arr is not None:
        # Fall back to any class if no reject qualified.
        for k in range(1, len(pi_stages)):
            pi_k = np.asarray(pi_stages[k], dtype=float)
            if pi_k.ndim == 3:
                pi_k = pi_k[0]
            for p in range(pi_k.shape[0]):
                _consider(p, k, float(pi_k[p, 1]))

    if best is None:
        return None
    _dist, p_idx, stage_idx, pi_live = best
    return int(p_idx), int(stage_idx), float(pi_live)


def run_adf_mc_diagnostics(
    controller,
    mu0,
    P0,
    pi0,
    particle_class,
    t_act,
    u,
    *,
    num_samples: int,
    num_micro_steps: int,
    seed: int,
    contact_mode: str = "geometric",
) -> Tuple[Any, Any]:
    """Runs shared ADF states + MC diagnostics for a single control sequence ``u``.

    :param controller: An ADF-capable controller (``ADFOLFCostMixin``) providing
        ``_forward_chain_states`` and ``_forward_chain_mc``.
    :param mu0: A np.array of shape ``[num_particles, state_dim]``, initial means.
    :param P0: A np.array of shape ``[num_particles, state_dim, state_dim]``, initial
        covariances.
    :param pi0: A np.array of shape ``[num_particles, 2]``, initial existence
        probabilities.
    :param particle_class: An integer np.array of shape ``[num_particles]``.
    :param t_act: A np.array of shape ``[num_actors]``, actors' internal time.
    :param u: A np.array of shape ``[batch_size, num_actors, num_stages]``, open-loop
        controls.
    :param num_samples: An integer, number of MC samples.
    :param num_micro_steps: An integer, fine-``Δt`` steps per stage (geometric mode).
    :param seed: An integer, RNG seed for the MC forward pass.
    :param contact_mode: A string, ``"geometric"`` or ``"gt_box"``.

    :returns: A tuple ``(adf_states, mc_diag)`` as returned by
        ``_forward_chain_states`` / ``_forward_chain_mc`` with diagnostics.
    """
    controller._particle_model.particle_class = np.asarray(particle_class, dtype=int)
    rng = np.random.default_rng(seed)
    _cost_adf, adf_states = controller._forward_chain_states(
        u, mu0, P0, pi0, t_act
    )
    _cost_mc, mc_diag = controller._forward_chain_mc(
        u,
        mu0,
        P0,
        pi0,
        t_act,
        num_samples=num_samples,
        num_micro_steps=num_micro_steps,
        return_diagnostics=True,
        rng=rng,
        contact_mode=contact_mode,
    )
    return adf_states, mc_diag


def plot_position_histograms_multistage(
    controller,
    mu0,
    P0,
    pi0,
    particle_class,
    t_act,
    u,
    particle_index: int,
    num_samples: int,
    num_micro_steps: int,
    seed: int,
    save_path: str | None = None,
    show: bool = True,
    stage_indices: Optional[Sequence[int]] = None,
    particle_id: Optional[int] = None,
    contact_mode: str = "geometric",
    csv_export_dir: Optional[str] = None,
    csv_meta: Optional[dict] = None,
    adf_states=None,
    mc_diag=None,
):
    """Plots multipanel ADF vs MC position densities (rows = stages, columns = ``x``/``y``).

    Stages default to ``0..N-1`` when ``stage_indices`` is ``None``. With a lateral
    state component the layout is ``n_stages × 2`` (``x`` | ``y``); otherwise
    ``n_stages × 1`` (``x`` only). If ``csv_export_dir`` is set, writes paper-ready
    sample/parameter CSVs via :func:`export_adf_mc_position_csvs`.

    :param controller: An ADF-capable controller providing forward-chain methods.
    :param mu0: A np.array of shape ``[num_particles, state_dim]``, initial means.
    :param P0: A np.array of shape ``[num_particles, state_dim, state_dim]``, initial
        covariances.
    :param pi0: A np.array of shape ``[num_particles, 2]``, initial existence
        probabilities.
    :param particle_class: An integer np.array of shape ``[num_particles]``.
    :param t_act: A np.array of shape ``[num_actors]``, actors' internal time.
    :param u: A np.array of shape ``[batch_size, num_actors, num_stages]``, open-loop
        controls.
    :param particle_index: An integer, which particle to plot.
    :param num_samples: An integer, number of MC samples (if ``mc_diag`` is not given).
    :param num_micro_steps: An integer, fine-``Δt`` steps per stage.
    :param seed: An integer, RNG seed for MC sampling.
    :param save_path: None or a string, PNG path; ``None`` skips saving.
    :param show: A Boolean, whether to call ``plt.show()``.
    :param stage_indices: None or a sequence of stage indices to plot.
    :param particle_id: None or an integer, id used in the figure title.
    :param contact_mode: A string, ``"geometric"`` or ``"gt_box"``.
    :param csv_export_dir: None or a string, directory for CSV export.
    :param csv_meta: None or a dict merged into CSV ``meta.json``.
    :param adf_states: None or precomputed ADF states; computed if ``None``.
    :param mc_diag: None or precomputed MC diagnostics; computed if ``None``.

    :returns: A tuple ``(fig, adf_states, mc_diag)``.
    """
    if adf_states is None or mc_diag is None:
        adf_states, mc_diag = run_adf_mc_diagnostics(
            controller,
            mu0,
            P0,
            pi0,
            particle_class,
            t_act,
            u,
            num_samples=num_samples,
            num_micro_steps=num_micro_steps,
            seed=seed,
            contact_mode=contact_mode,
        )
    n_stages_total = len(adf_states["pi_stages"])  # N+1
    if stage_indices is None:
        stage_indices = list(range(0, n_stages_total - 1)) if n_stages_total > 1 else [0]
    stage_indices = list(stage_indices)
    nrows = len(stage_indices)
    state_dim = int(mc_diag["ms_stages"][stage_indices[0]].shape[-1])
    comps = _position_components(state_dim)
    ncols = len(comps)

    meta = dict(csv_meta or {})
    # Partial-hit contact stage: prefer explicit meta, else second window entry (k).
    stage_k_meta = meta.get("partial_hit_stage")
    if stage_k_meta is None and len(stage_indices) >= 2:
        stage_k_meta = stage_indices[1]
    eject_summary = None
    if stage_k_meta is not None:
        eject_summary = summarize_mc_eject_actor_counts_from_stage(
            mc_diag,
            particle_index,
            int(stage_k_meta),
            stage_indices=stage_indices,
        )
        if eject_summary is not None:
            meta.update(eject_summary)
            for sk, n_ej in eject_summary["n_ejected_by_stage"].items():
                logging.info(
                    "ADF/MC partial-hit particle_index=%s stage=%s n_ejected=%s "
                    "mc_eject_actor_counts=%s",
                    particle_index,
                    sk,
                    n_ej,
                    eject_summary["mc_eject_actor_counts_by_stage"][sk],
                )

    if csv_export_dir:
        meta.setdefault("particle_id", particle_id)
        meta.setdefault("seed", int(seed))
        meta.setdefault("num_samples", int(num_samples))
        meta.setdefault("num_micro_steps", int(num_micro_steps))
        meta.setdefault("contact_mode", mc_diag.get("contact_mode", contact_mode))
        export_adf_mc_position_csvs(
            csv_export_dir,
            adf_states,
            mc_diag,
            particle_index=particle_index,
            stage_indices=stage_indices,
            meta=meta,
        )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.2 * max(nrows, 1)),
        squeeze=False,
    )
    for row, k in enumerate(stage_indices):
        for col, (comp, label) in enumerate(comps):
            _draw_stage_panel(
                axes[row, col],
                ms=mc_diag["ms_stages"][k],
                ex=mc_diag["ex_stages"][k],
                pi_adf=adf_states["pi_stages"][k][0, particle_index],
                mu_adf=adf_states["mu_stages"][k][0, particle_index],
                P_adf=adf_states["P_stages"][k][0, particle_index],
                particle_index=particle_index,
                stage_index=k,
                state_comp=comp,
                coord_label=label,
                pi_mc=mc_diag["pi_stages"][k][particle_index],
                show_legend=(col == 0),
                title=f"stage {k}",
            )
            if col == 0:
                axes[row, col].set_ylabel("density")
            if row == 0:
                axes[row, col].set_title(
                    f"{label.strip('$')}  |  stage {k}", fontsize=9
                )

    pid_str = f"id={particle_id}" if particle_id is not None else f"idx={particle_index}"
    mode = mc_diag.get("contact_mode", contact_mode)
    fig.suptitle(f"ADF vs MC positions — particle {pid_str} (MC={mode})")
    _finalize_fig(fig, save_path, show)
    return fig, adf_states, mc_diag
