"""ADF vs MC terminal-cost evaluation over all candidate control sequences.

Computes signed per-particle cost gaps ``(J_MC - J_ADF) / N_P`` and whether
MC would select a different open-loop sequence than ADF.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class CallCostEvalResult:
    """One controller call: all candidate sequences scored by ADF and MC.

    Attributes:
        call_index: An integer, controller-call index within the run.
        j_adf: A np.array of shape ``[num_sequences]``, ADF terminal OLF costs.
        j_mc: A np.array of shape ``[num_sequences]``, MC terminal OLF costs.
        delta_per_particle: A np.array of shape ``[num_sequences]``,
            ``(J_MC - J_ADF) / N_P`` per sequence.
        num_particles: An integer, ``N_P`` used for per-particle normalization.
        argmin_adf: An integer, index of the ADF-optimal sequence.
        argmin_mc: An integer, index of the MC-optimal sequence.
        decision_differs: A Boolean, ``True`` if ``argmin_adf != argmin_mc``.
        regret_mc_per_particle: A float,
            ``(J_MC(u*_ADF) - J_MC(u*_MC)) / N_P``.
    """

    call_index: int
    j_adf: np.ndarray  # (num_sequences,)
    j_mc: np.ndarray  # (num_sequences,)
    delta_per_particle: np.ndarray  # (J_MC - J_ADF) / N_P
    num_particles: int
    argmin_adf: int
    argmin_mc: int
    decision_differs: bool
    regret_mc_per_particle: float  # (J_MC(u*_ADF) - J_MC(u*_MC)) / N_P


@dataclass
class AdfMcCostEvalAccumulator:
    """Run-level aggregation of :class:`CallCostEvalResult`.

    Attributes:
        calls: A list of :class:`CallCostEvalResult` recorded so far.
    """

    calls: List[CallCostEvalResult] = field(default_factory=list)

    def record(self, result: CallCostEvalResult) -> None:
        """Appends one controller-call cost-eval result.

        :param result: A :class:`CallCostEvalResult` to store.
        """
        self.calls.append(result)

    @property
    def all_deltas(self) -> np.ndarray:
        """All sequence-level ``(J_MC - J_ADF) / N_P`` values across calls.

        :returns: A 1D np.array (empty if no calls were recorded).
        """
        if not self.calls:
            return np.array([], dtype=float)
        return np.concatenate([c.delta_per_particle for c in self.calls])

    @property
    def decision_disagree_rate(self) -> float:
        """Fraction of controller calls where ADF and MC select different ``u*``.

        :returns: A float in ``[0, 1]``, or ``nan`` if no calls were recorded.
        """
        if not self.calls:
            return float("nan")
        return float(np.mean([c.decision_differs for c in self.calls]))

    @property
    def num_calls(self) -> int:
        """Number of recorded controller calls.

        :returns: An integer.
        """
        return len(self.calls)

    @property
    def num_sequence_evals(self) -> int:
        """Total number of sequences scored across all calls.

        :returns: An integer.
        """
        return int(sum(c.j_adf.size for c in self.calls))

    def summary(self) -> Dict[str, Any]:
        """Aggregates run-level statistics for logging and paper CSVs.

        :returns: A dict with call/sequence counts, decision-disagree rate,
            delta and regret percentiles (keys omitted when no data).
        """
        deltas = self.all_deltas
        regrets = np.array([c.regret_mc_per_particle for c in self.calls], dtype=float)
        regrets_disagree = np.array(
            [c.regret_mc_per_particle for c in self.calls if c.decision_differs],
            dtype=float,
        )
        seq_counts = np.array([c.j_adf.size for c in self.calls], dtype=float)
        out: Dict[str, Any] = {
            "num_calls": self.num_calls,
            "num_sequence_evals": self.num_sequence_evals,
            "sequences_per_call_mean": (
                float(np.mean(seq_counts)) if seq_counts.size else float("nan")
            ),
            "sequences_per_call_min": (
                int(np.min(seq_counts)) if seq_counts.size else 0
            ),
            "sequences_per_call_max": (
                int(np.max(seq_counts)) if seq_counts.size else 0
            ),
            "decision_disagree_rate": self.decision_disagree_rate,
            "decision_disagree_count": int(
                sum(1 for c in self.calls if c.decision_differs)
            ),
        }
        if deltas.size:
            out.update(
                {
                    "delta_mean": float(np.mean(deltas)),
                    "delta_std": float(np.std(deltas)),
                    "delta_p05": float(np.percentile(deltas, 5)),
                    "delta_p50": float(np.percentile(deltas, 50)),
                    "delta_p95": float(np.percentile(deltas, 95)),
                    "delta_frac_adf_under": float(np.mean(deltas > 0.0)),
                    "delta_frac_adf_over": float(np.mean(deltas < 0.0)),
                }
            )
        if regrets.size:
            out.update(
                {
                    "regret_mean": float(np.mean(regrets)),
                    "regret_p95": float(np.percentile(regrets, 95)),
                    "regret_max": float(np.max(regrets)),
                }
            )
        if regrets_disagree.size:
            out.update(
                {
                    "regret_mean_when_disagree": float(np.mean(regrets_disagree)),
                    "regret_p50_when_disagree": float(np.percentile(regrets_disagree, 50)),
                    "regret_p95_when_disagree": float(np.percentile(regrets_disagree, 95)),
                    "regret_max_when_disagree": float(np.max(regrets_disagree)),
                }
            )
        return out

    @property
    def all_regrets(self) -> np.ndarray:
        """Per-call MC regrets ``(J_MC(u*_ADF) - J_MC(u*_MC)) / N_P``.

        :returns: A 1D np.array of length ``num_calls`` (empty if none).
        """
        if not self.calls:
            return np.array([], dtype=float)
        return np.array([c.regret_mc_per_particle for c in self.calls], dtype=float)

    @property
    def regrets_when_disagree(self) -> np.ndarray:
        """MC regrets restricted to calls where ADF and MC disagree on ``u*``.

        :returns: A 1D np.array (empty if none disagree or no calls).
        """
        if not self.calls:
            return np.array([], dtype=float)
        return np.array(
            [c.regret_mc_per_particle for c in self.calls if c.decision_differs],
            dtype=float,
        )


def evaluate_adf_mc_terminal_costs(
    controller: Any,
    control_sequences: np.ndarray,
    estimated_motion_state_mean: np.ndarray,
    estimated_motion_state_cov: np.ndarray,
    est_existence_probs: np.ndarray,
    t_act_k_at_T: np.ndarray,
    particle_class: np.ndarray,
    *,
    num_samples: int,
    num_micro_steps: int,
    seed: int,
    call_index: int = 0,
    j_adf: Optional[np.ndarray] = None,
    contact_mode: str = "geometric",
) -> CallCostEvalResult:
    """Scores all candidate sequences with ADF and MC (terminal costs only).

    :param controller: An ADF-capable controller with ``_forward_chain`` /
        ``_forward_chain_mc`` and ``_use_only_terminal_costs=True``.
    :param control_sequences: A np.array of shape
        ``[num_sequences, num_actors, num_stages]``, open-loop controls.
    :param estimated_motion_state_mean: A np.array of shape
        ``[num_particles, state_dim]``.
    :param estimated_motion_state_cov: A np.array of shape
        ``[num_particles, state_dim, state_dim]``.
    :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` (or
        compatible), initial existence probabilities.
    :param t_act_k_at_T: A np.array of shape ``[num_actors]``, actors' internal time.
    :param particle_class: An integer np.array of shape ``[num_particles]``.
    :param num_samples: An integer, MC samples per forward pass.
    :param num_micro_steps: An integer, fine-``Δt`` steps per stage.
    :param seed: An integer, base RNG seed (offset by ``call_index``).
    :param call_index: An integer, controller-call index stored in the result.
    :param j_adf: None or a np.array of shape ``[num_sequences]``, precomputed ADF
        costs (e.g. from ``_adf_cum_cost``); recomputed if ``None``.
    :param contact_mode: A string, ``"geometric"`` or ``"gt_box"``.

    :returns: A :class:`CallCostEvalResult` with per-sequence costs, deltas, and
        decision / regret fields.

    :raises ValueError: If terminal costs are disabled, shapes are invalid, or there
        are no particles.
    """
    if not bool(getattr(controller, "_use_only_terminal_costs", False)):
        raise ValueError(
            "ADF/MC terminal-cost evaluation requires use_only_terminal_costs=True "
            "(stage costs are signed and not in [0, N_P])."
        )

    u = np.asarray(control_sequences, dtype=float)
    if u.ndim != 3:
        raise ValueError(
            f"control_sequences must be (M, Na, N), got shape {u.shape}"
        )
    num_sequences = int(u.shape[0])
    mu0 = np.asarray(estimated_motion_state_mean, dtype=float)
    P0 = np.asarray(estimated_motion_state_cov, dtype=float)
    pi0 = np.asarray(est_existence_probs, dtype=float)
    particle_class = np.asarray(particle_class, dtype=int)
    num_particles = int(mu0.shape[0])
    if num_particles < 1:
        raise ValueError("Need at least one particle for cost evaluation")

    controller._particle_model.particle_class = particle_class

    if j_adf is None:
        j_adf = np.asarray(
            controller._forward_chain(u, mu0, P0, pi0, t_act_k_at_T),
            dtype=float,
        ).reshape(-1)
    else:
        j_adf = np.asarray(j_adf, dtype=float).reshape(-1)
    if j_adf.shape != (num_sequences,):
        raise ValueError(
            f"j_adf shape {j_adf.shape} != ({num_sequences},) sequences"
        )

    mc_out = controller._forward_chain_mc(
        u,
        mu0,
        P0,
        pi0,
        t_act_k_at_T,
        num_samples=num_samples,
        num_micro_steps=num_micro_steps,
        return_diagnostics=False,
        rng=np.random.default_rng(int(seed) + int(call_index)),
        contact_mode=contact_mode,
    )
    j_mc = np.asarray(mc_out, dtype=float).reshape(-1)
    if j_mc.shape != (num_sequences,):
        raise ValueError(f"j_mc shape {j_mc.shape} != ({num_sequences},)")

    n_p = float(num_particles)
    delta = (j_mc - j_adf) / n_p
    i_adf = int(np.argmin(j_adf))
    i_mc = int(np.argmin(j_mc))
    regret = float((j_mc[i_adf] - j_mc[i_mc]) / n_p)

    return CallCostEvalResult(
        call_index=int(call_index),
        j_adf=j_adf,
        j_mc=j_mc,
        delta_per_particle=delta,
        num_particles=num_particles,
        argmin_adf=i_adf,
        argmin_mc=i_mc,
        decision_differs=(i_adf != i_mc),
        regret_mc_per_particle=regret,
    )


def export_adf_mc_cost_eval_csvs(
    accumulator: AdfMcCostEvalAccumulator,
    out_dir: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Writes paper-ready CSVs (samples + one-row summary) under ``out_dir``.

    Files written:

    - ``delta_per_particle.csv`` — column ``delta``: all sequence-level
      ``(J_MC - J_ADF) / N_P``.
    - ``regret_when_disagree.csv`` — column ``regret``: per-call MC regret for
      calls with decision mismatch (empty data rows if none disagree).
    - ``summary.csv`` — single row with keys from
      :meth:`AdfMcCostEvalAccumulator.summary`.
    - ``meta.json`` — optional run metadata plus the summary dict.

    :param accumulator: An :class:`AdfMcCostEvalAccumulator` with recorded calls.
    :param out_dir: A string, output directory (created if missing).
    :param meta: None or a dict merged into ``meta.json``.

    :returns: The string ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = accumulator.summary()
    deltas = accumulator.all_deltas
    regrets_disagree = accumulator.regrets_when_disagree

    delta_path = os.path.join(out_dir, "delta_per_particle.csv")
    with open(delta_path, "w", encoding="utf-8") as f:
        f.write("delta\n")
        for v in deltas:
            f.write(f"{float(v):.10g}\n")

    regret_path = os.path.join(out_dir, "regret_when_disagree.csv")
    with open(regret_path, "w", encoding="utf-8") as f:
        f.write("regret\n")
        for v in regrets_disagree:
            f.write(f"{float(v):.10g}\n")

    # Stable column order for LaTeX readers.
    summary_keys = [
        "num_calls",
        "num_sequence_evals",
        "sequences_per_call_mean",
        "sequences_per_call_min",
        "sequences_per_call_max",
        "decision_disagree_rate",
        "decision_disagree_count",
        "delta_mean",
        "delta_std",
        "delta_p05",
        "delta_p50",
        "delta_p95",
        "delta_frac_adf_under",
        "delta_frac_adf_over",
        "regret_mean",
        "regret_p95",
        "regret_max",
        "regret_mean_when_disagree",
        "regret_p50_when_disagree",
        "regret_p95_when_disagree",
        "regret_max_when_disagree",
    ]
    row = {k: summary.get(k, float("nan")) for k in summary_keys}
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(",".join(summary_keys) + "\n")
        f.write(
            ",".join(
                (
                    f"{int(row[k])}"
                    if k
                    in (
                        "num_calls",
                        "num_sequence_evals",
                        "sequences_per_call_min",
                        "sequences_per_call_max",
                        "decision_disagree_count",
                    )
                    and np.isfinite(float(row[k]))
                    else f"{float(row[k]):.10g}"
                )
                for k in summary_keys
            )
            + "\n"
        )

    meta_path = os.path.join(out_dir, "meta.json")
    meta_out = dict(meta or {})
    meta_out["summary"] = summary
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    logging.info(
        "ADF/MC cost eval CSVs written to %s (deltas=%d, regrets_disagree=%d)",
        out_dir,
        int(deltas.size),
        int(regrets_disagree.size),
    )
    return out_dir


def plot_adf_mc_cost_eval_summary(
    accumulator: AdfMcCostEvalAccumulator,
    save_path: Optional[str] = None,
    show: bool = False,
    *,
    csv_dir: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
):
    """Plots boxplots for cost gap and MC regret, plus a decision-divergence statement.

    Optionally writes a PNG, a JSON summary next to it, and paper CSVs via
    :func:`export_adf_mc_cost_eval_csvs`.

    :param accumulator: An :class:`AdfMcCostEvalAccumulator` with recorded calls.
    :param save_path: None or a string, PNG path; also drives default CSV/JSON
        sibling paths when set.
    :param show: A Boolean, whether to call ``plt.show()``.
    :param csv_dir: None or a string, CSV export directory. If ``None`` and
        ``save_path`` is set, uses ``<save_path without ext>_csv``.
    :param meta: None or a dict passed to the CSV exporter.

    :returns: A tuple ``(fig, summary_dict)`` where ``summary_dict`` is
        :meth:`AdfMcCostEvalAccumulator.summary`.
    """
    deltas = accumulator.all_deltas
    regrets_all = accumulator.all_regrets
    regrets_disagree = accumulator.regrets_when_disagree
    summary = accumulator.summary()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    def _boxplot(ax, data, ylabel, title, stats_lines):
        if data.size:
            ax.boxplot(
                data,
                vert=True,
                widths=0.45,
                patch_artist=True,
                boxprops=dict(facecolor="C0", alpha=0.55),
                medianprops=dict(color="C3", lw=2.0),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
            )
            ax.axhline(0.0, color="k", lw=1.0, ls="--")
            ax.set_xticks([1])
            ax.set_xticklabels([ylabel])
            ax.set_ylabel(ylabel)
            ax.text(
                0.98,
                0.98,
                "\n".join(stats_lines),
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
            )
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    _boxplot(
        axes[0],
        deltas,
        r"$(J_{\mathrm{MC}}-J_{\mathrm{ADF}})/N_{\mathrm{P}}$",
        "Cost gap (all sequences)",
        [
            f"n={deltas.size}",
            f"mean={summary.get('delta_mean', float('nan')):.4g}",
            f"median={summary.get('delta_p50', float('nan')):.4g}",
            f"ADF under: {100.0 * summary.get('delta_frac_adf_under', float('nan')):.1f}%",
            f"ADF over:  {100.0 * summary.get('delta_frac_adf_over', float('nan')):.1f}%",
        ],
    )

    # Regret conditional on disagreement (zeros from agreeing calls excluded).
    n_diff = int(summary.get("decision_disagree_count", 0))
    _boxplot(
        axes[1],
        regrets_disagree if regrets_disagree.size else regrets_all,
        r"MC regret $/N_{\mathrm{P}}$",
        (
            f"MC regret when u* differs (n={regrets_disagree.size})"
            if regrets_disagree.size
            else f"MC regret all calls (n={regrets_all.size})"
        ),
        [
            f"n_disagree={n_diff}",
            f"mean|≠={summary.get('regret_mean_when_disagree', summary.get('regret_mean', float('nan'))):.4g}",
            f"median|≠={summary.get('regret_p50_when_disagree', float('nan')):.4g}",
            f"p95|≠={summary.get('regret_p95_when_disagree', summary.get('regret_p95', float('nan'))):.4g}",
            f"max|≠={summary.get('regret_max_when_disagree', summary.get('regret_max', float('nan'))):.4g}",
            f"mean(all)={summary.get('regret_mean', float('nan')):.4g}",
        ],
    )

    ax2 = axes[2]
    ax2.axis("off")
    rate = summary.get("decision_disagree_rate", float("nan"))
    n_calls = int(summary.get("num_calls", 0))
    n_same = n_calls - n_diff
    if n_calls > 0 and np.isfinite(rate):
        pct = 100.0 * rate
        statement = (
            f"Decision divergence\n\n"
            f"In {n_diff} of {n_calls} controller calls\n"
            f"({pct:.1f}%), MC would have selected\n"
            f"a different open-loop sequence u*\n"
            f"than ADF.\n\n"
            f"Same decision: {n_same}/{n_calls} "
            f"({100.0 - pct:.1f}%)\n\n"
            f"Sequences/call: mean="
            f"{summary.get('sequences_per_call_mean', float('nan')):.2f}, "
            f"min={summary.get('sequences_per_call_min', 0)}, "
            f"max={summary.get('sequences_per_call_max', 0)}"
        )
    else:
        statement = "No controller calls evaluated."
    ax2.text(
        0.5,
        0.55,
        statement,
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )
    ax2.set_title("ADF vs MC control selection")

    fig.suptitle("ADF vs MC terminal-cost evaluation (run summary)")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        summary_path = os.path.splitext(save_path)[0] + "_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        paper_csv_dir = csv_dir
        if paper_csv_dir is None:
            paper_csv_dir = os.path.splitext(save_path)[0] + "_csv"
        export_adf_mc_cost_eval_csvs(accumulator, paper_csv_dir, meta=meta)
    elif csv_dir:
        export_adf_mc_cost_eval_csvs(accumulator, csv_dir, meta=meta)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, summary


def log_cost_eval_summary(accumulator: AdfMcCostEvalAccumulator) -> None:
    """Logs a one-line run-level ADF/MC cost-eval summary via absl logging.

    :param accumulator: An :class:`AdfMcCostEvalAccumulator` with recorded calls.
    """
    s = accumulator.summary()
    logging.info(
        "ADF/MC cost eval: calls=%s sequences=%s (mean %.2f/call) "
        "decision_disagree=%.1f%% delta_mean=%s delta_p50=%s "
        "regret_mean=%s regret_mean|≠=%s",
        s.get("num_calls"),
        s.get("num_sequence_evals"),
        s.get("sequences_per_call_mean", float("nan")),
        100.0 * s.get("decision_disagree_rate", float("nan")),
        s.get("delta_mean"),
        s.get("delta_p50"),
        s.get("regret_mean"),
        s.get("regret_mean_when_disagree"),
    )
