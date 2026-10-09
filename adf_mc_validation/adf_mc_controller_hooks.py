"""Optional ADF vs Monte-Carlo diagnostic hooks for PTCRController.

Lives outside the Numba OLF / tree-search hot path. Constructed by
``PTCRController`` and invoked after a control decision when enabled via
nested config ``adf.mc_comparison``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from absl import logging

from adf_mc_validation.adf_mc_cost_eval import (
    AdfMcCostEvalAccumulator,
    evaluate_adf_mc_terminal_costs,
    log_cost_eval_summary,
    plot_adf_mc_cost_eval_summary,
)
from adf_mc_validation.adf_mc_plots import (
    find_partial_hit_particle,
    partial_hit_stage_window,
    plot_position_histograms_multistage,
)


def normalize_adf_mc_comparison_kwargs(mc_comparison=None):
    """Normalizes nested ``adf.mc_comparison`` to flat fields with defaults.

    Expected nested shape (JSON / Python dict)::

        {
          "validate": false,
          "save_dir": null,
          "show": false,
          "max_calls": 20,
          "num_samples": 800,
          "num_micro_steps": 50,
          "seed": 42,
          "contact_mode": "geometric",
          "plot_histograms": true,
          "plot_partial_hit": true,
          "partial_hit": {"pi_lo": 0.3, "pi_hi": 0.7, "min_n_meas": 8},
          "cost_eval": {"enable": false, "max_calls": null, "plot": true}
        }

    :param mc_comparison: None or a dict with the nested ADF/MC validation and
        cost-eval settings. Missing keys fall back to the defaults above. ``None`` or ``{}``
        yields all defaults with ``validate`` / ``cost_eval`` disabled.

    :returns: A flat dict with keys ``validate``, ``save_dir``, ``show``, ``max_calls``,
        ``num_samples``, ``num_micro_steps``, ``seed``, ``contact_mode``, ``plot_histograms``,
        ``plot_partial_hit``, ``partial_pi_lo``, ``partial_pi_hi``, ``partial_min_n_meas``,
        ``cost_eval``, ``cost_eval_max_calls``, ``cost_eval_plot``.
    """
    cfg = dict(mc_comparison or {})
    partial = dict(cfg.get("partial_hit") or {})
    cost_eval = dict(cfg.get("cost_eval") or {})
    max_calls_ce = cost_eval.get("max_calls", None)
    return {
        "validate": bool(cfg.get("validate", False)),
        "save_dir": cfg.get("save_dir"),
        "show": bool(cfg.get("show", False)),
        "max_calls": int(cfg.get("max_calls", 20)),
        "num_samples": int(cfg.get("num_samples", 800)),
        "num_micro_steps": int(cfg.get("num_micro_steps", 50)),
        "seed": int(cfg.get("seed", 42)),
        "contact_mode": str(cfg.get("contact_mode", "geometric")),
        "plot_histograms": bool(cfg.get("plot_histograms", True)),
        "plot_partial_hit": bool(cfg.get("plot_partial_hit", True)),
        "partial_pi_lo": float(partial.get("pi_lo", 0.3)),
        "partial_pi_hi": float(partial.get("pi_hi", 0.7)),
        "partial_min_n_meas": int(partial.get("min_n_meas", 8)),
        "cost_eval": bool(cost_eval.get("enable", False)),
        "cost_eval_max_calls": None if max_calls_ce is None else int(max_calls_ce),
        "cost_eval_plot": bool(cost_eval.get("plot", True)),
    }


class AdfMcComparisonHooks:
    """Owns ADF/MC validation and terminal-cost evaluation state for a controller.

    :param controller: The owning controller instance (must provide ADF forward helpers
        when validation / cost eval run).
    :param mc_comparison: None or a nested dict (config ``adf.mc_comparison``); see
        :func:`normalize_adf_mc_comparison_kwargs`.
    """

    def __init__(self, controller, mc_comparison=None):
        """Initializes hooks from nested ``adf.mc_comparison`` settings.

        :param controller: The owning controller instance.
        :param mc_comparison: None or a nested dict of ADF/MC settings.
        """
        self._controller = controller
        amc = normalize_adf_mc_comparison_kwargs(mc_comparison)
        self.validate = amc["validate"]
        self.plot_histograms = amc["plot_histograms"]
        self.save_dir = amc["save_dir"]
        self.show = amc["show"]
        self.num_samples = amc["num_samples"]
        self.num_micro_steps = amc["num_micro_steps"]
        self.seed = amc["seed"]
        self.max_calls = amc["max_calls"]
        self.contact_mode = amc["contact_mode"]
        self.plot_partial_hit = amc["plot_partial_hit"]
        self.partial_pi_lo = amc["partial_pi_lo"]
        self.partial_pi_hi = amc["partial_pi_hi"]
        self.partial_min_n_meas = amc["partial_min_n_meas"]
        self.cost_eval = amc["cost_eval"]
        self.cost_eval_max_calls = amc["cost_eval_max_calls"]
        self.cost_eval_plot = amc["cost_eval_plot"]
        self._cost_eval_call_counter = 0
        self._cost_eval_acc = AdfMcCostEvalAccumulator()
        self._call_counter = 0
        if self.validate and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        if self.cost_eval and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def maybe_cost_eval(
        self,
        control_sequences_delta_t,
        costs_adf,
        estimated_motion_state_mean,
        estimated_motion_state_cov,
        init_existence_probs,
        particle_class,
        t_act,
    ):
        """Scores all candidate sequences with MC and accumulates run-level cost metrics.

        Outside the Numba OLF / tree-search hot path. Requires
        ``use_only_terminal_costs=True`` on the controller. Failures are logged and never
        propagate to the controller return path. No-op if cost evaluation is disabled.

        :param control_sequences_delta_t: A np.array of shape
            ``[num_sequences, num_actors, N]``, the candidate open-loop activation
            sequences (delta times w.r.t. the current stage).
        :param costs_adf: None or a np.array of shape ``[num_sequences]``, the ADF OLF
            costs already computed for these sequences (reused when not ``None``).
        :param estimated_motion_state_mean: A np.array of shape ``[num_particles, state_dim]``,
            the estimated motion-state means.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``, the estimated motion-state covariances.
        :param init_existence_probs: A np.array of shape ``[num_particles]``, the initial
            living probabilities (track scores) used to build ``π`` for the forward passes.
        :param particle_class: An integer np.array of shape ``[num_particles]`` where each
            component is either 0 ("keep") or 1 ("eject").
        :param t_act: A np.array of shape ``[num_actors]``, the actors' current internal time.
        """
        if not self.cost_eval:
            return

        max_calls = self.cost_eval_max_calls
        if max_calls is not None and self._cost_eval_call_counter >= max_calls:
            return

        call_idx = self._cost_eval_call_counter
        self._cost_eval_call_counter += 1
        controller = self._controller

        try:
            particle_class = np.asarray(particle_class, dtype=int)
            pi0 = np.c_[
                1.0 - np.asarray(init_existence_probs, dtype=float),
                np.asarray(init_existence_probs, dtype=float),
            ]
            result = evaluate_adf_mc_terminal_costs(
                controller,
                control_sequences_delta_t,
                estimated_motion_state_mean,
                estimated_motion_state_cov,
                pi0,
                t_act,
                particle_class,
                num_samples=self.num_samples,
                num_micro_steps=self.num_micro_steps,
                seed=self.seed,
                call_index=call_idx,
                j_adf=costs_adf,
                contact_mode=self.contact_mode,
            )
            self._cost_eval_acc.record(result)
            logging.info(
                "ADF/MC cost eval call %d: M=%d sequences, decision_differs=%s, "
                "delta_mean/N_P=%.4g, regret/N_P=%.4g",
                call_idx,
                result.j_adf.size,
                result.decision_differs,
                float(np.mean(result.delta_per_particle)),
                result.regret_mc_per_particle,
            )
            if self.cost_eval_plot and self.save_dir:
                plot_adf_mc_cost_eval_summary(
                    self._cost_eval_acc,
                    save_path=os.path.join(
                        self.save_dir, "adf_mc_cost_eval_summary.png"
                    ),
                    show=False,
                )
        except Exception:  # noqa: BLE001 — never break the controller return path
            logging.exception(
                "ADF/MC cost evaluation failed at controller call %d", call_idx
            )

    def finalize_cost_eval(
        self, save_path: Optional[str] = None, show: Optional[bool] = None
    ):
        """Writes the final ADF/MC cost-eval summary at the end of a simulation run.

        Logs aggregate statistics and optionally writes/updates the summary PNG (+ JSON / CSVs)
        under the configured ``save_dir``. No-op if cost evaluation was disabled or
        no calls were recorded.

        :param save_path: None or a string, path for the summary PNG. If ``None`` and a save
            directory was configured, uses ``<save_dir>/adf_mc_cost_eval_summary.png``.
        :param show: None or a Boolean. If ``None``, uses the configured ``show`` flag;
            otherwise overrides whether ``plt.show()`` is called.

        :returns: ``None`` if nothing was recorded / cost eval disabled; otherwise the return
            value of :func:`plot_adf_mc_cost_eval_summary` (typically ``(fig, summary_dict)``),
            or ``(None, summary_dict)`` if neither saving nor showing is requested.
        """
        if not self.cost_eval:
            return None
        if self._cost_eval_acc.num_calls < 1:
            logging.info("ADF/MC cost eval finalize: no calls recorded")
            return None
        log_cost_eval_summary(self._cost_eval_acc)
        path = save_path
        if path is None and self.save_dir:
            path = os.path.join(self.save_dir, "adf_mc_cost_eval_summary.png")
        do_show = self.show if show is None else bool(show)
        if path or do_show:
            return plot_adf_mc_cost_eval_summary(
                self._cost_eval_acc, save_path=path, show=do_show
            )
        return None, self._cost_eval_acc.summary()

    def maybe_validate(
        self,
        u_star,
        estimated_motion_state_mean,
        estimated_motion_state_cov,
        particle_class,
        particle_id,
        t_act,
    ):
        """Optional ADF vs MC diagnostic plots after selecting the optimal open-loop sequence.

        Depending on comparison kwargs, plots multipanel position histograms for a random
        reject particle and/or a partial-hit particle, and may export paper CSVs. Failures
        are logged and never propagate to the controller return path. Stops after
        ``max_calls`` controller calls. No-op if validation is disabled.

        :param u_star: A np.array of shape ``[1, num_actors, N]`` or
            ``[num_actors, N]``, the selected open-loop activation sequence.
        :param estimated_motion_state_mean: A np.array of shape ``[num_particles, state_dim]``,
            the estimated motion-state means.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``, the estimated motion-state covariances.
        :param particle_class: An integer np.array of shape ``[num_particles]`` where each
            component is either 0 ("keep") or 1 ("eject").
        :param particle_id: An integer np.array of shape ``[num_particles]``, the particle /
            track ids (used in plot titles and filenames).
        :param t_act: A np.array of shape ``[num_actors]``, the actors' current internal time.
        """
        if not self.validate:
            return

        if self._call_counter >= self.max_calls:
            return

        call_idx = self._call_counter
        self._call_counter += 1
        controller = self._controller

        try:
            particle_class = np.asarray(particle_class, dtype=int)
            particle_id = np.asarray(particle_id, dtype=int)
            pi0 = np.c_[
                1.0 - np.ones(estimated_motion_state_mean.shape[0]),
                np.ones(estimated_motion_state_mean.shape[0]),
            ]
            rng = np.random.default_rng(self.seed + call_idx)
            save_dir = self.save_dir
            show = self.show

            if self.plot_histograms:
                reject_idx = np.flatnonzero(particle_class == 1)
                if reject_idx.size == 0:
                    logging.info(
                        "ADF/MC validate call %d: no reject particles — skip random histograms",
                        call_idx,
                    )
                else:
                    p_idx = int(rng.choice(reject_idx))
                    pid = int(particle_id[p_idx]) if particle_id.size else p_idx
                    hist_path = None
                    if save_dir:
                        hist_path = os.path.join(
                            save_dir, f"t{call_idx:04d}_pid{pid}_hist.png"
                        )
                    plot_position_histograms_multistage(
                        controller,
                        estimated_motion_state_mean,
                        estimated_motion_state_cov,
                        pi0,
                        particle_class,
                        t_act,
                        u_star,
                        particle_index=p_idx,
                        num_samples=self.num_samples,
                        num_micro_steps=self.num_micro_steps,
                        seed=self.seed + call_idx,
                        save_path=hist_path,
                        show=show,
                        particle_id=pid,
                        contact_mode=self.contact_mode,
                    )

            if self.plot_partial_hit:
                controller._particle_model.particle_class = particle_class
                _cost, adf_states = controller._forward_chain_states(
                    u_star,
                    estimated_motion_state_mean,
                    estimated_motion_state_cov,
                    pi0,
                    t_act,
                )
                del _cost
                track_info = getattr(controller._mtt_tracker, "track_info", None) or {}
                n_meas_arr = track_info.get("n_associated_meas")
                hit = find_partial_hit_particle(
                    adf_states["pi_stages"],
                    particle_class,
                    pi_lo=self.partial_pi_lo,
                    pi_hi=self.partial_pi_hi,
                    prefer_reject=True,
                    n_associated_meas=n_meas_arr,
                    min_n_associated_meas=self.partial_min_n_meas,
                )
                if hit is None:
                    logging.info(
                        "ADF/MC validate call %d: no partial-hit particle "
                        "(π̃ ∈ (%.2f, %.2f), min_n_meas=%d) — skip partial histograms",
                        call_idx,
                        self.partial_pi_lo,
                        self.partial_pi_hi,
                        self.partial_min_n_meas,
                    )
                else:
                    p_idx, stage_k, pi_live = hit
                    pid = int(particle_id[p_idx]) if particle_id.size else p_idx
                    stage_indices = partial_hit_stage_window(
                        stage_k, len(adf_states["pi_stages"]), n_after=4
                    )
                    if stage_indices is None:
                        logging.info(
                            "ADF/MC validate call %d: partial-hit id=%s stage=%d "
                            "but horizon too short for [k-1 .. k+3] — skip",
                            call_idx,
                            pid,
                            stage_k,
                        )
                    else:
                        hist_path = None
                        csv_dir = None
                        if save_dir:
                            hist_path = os.path.join(
                                save_dir,
                                f"t{call_idx:04d}_pid{pid}_partial_hist.png",
                            )
                            csv_dir = os.path.join(
                                save_dir,
                                f"t{call_idx:04d}_pid{pid}_partial_csv",
                            )
                        if (
                            n_meas_arr is not None
                            and getattr(n_meas_arr, "shape", None) is not None
                            and n_meas_arr.shape[0] > p_idx
                        ):
                            n_meas_val = int(n_meas_arr[p_idx])
                        else:
                            n_meas_val = None
                        logging.info(
                            "ADF/MC validate call %d: partial-hit particle id=%s "
                            "idx=%d stage=%d π̃¹=%.3f stages=%s n_associated_meas=%s",
                            call_idx,
                            pid,
                            p_idx,
                            stage_k,
                            pi_live,
                            stage_indices,
                            n_meas_val if n_meas_val is not None else "?",
                        )
                        csv_meta = {
                            "call_idx": int(call_idx),
                            "partial_hit_stage": int(stage_k),
                            "partial_hit_pi_live": float(pi_live),
                        }
                        if n_meas_val is not None:
                            csv_meta["n_associated_meas"] = n_meas_val
                        plot_position_histograms_multistage(
                            controller,
                            estimated_motion_state_mean,
                            estimated_motion_state_cov,
                            pi0,
                            particle_class,
                            t_act,
                            u_star,
                            particle_index=p_idx,
                            num_samples=self.num_samples,
                            num_micro_steps=self.num_micro_steps,
                            seed=self.seed + call_idx + 10_000,
                            save_path=hist_path,
                            show=show,
                            stage_indices=stage_indices,
                            particle_id=pid,
                            contact_mode=self.contact_mode,
                            csv_export_dir=csv_dir,
                            csv_meta=csv_meta,
                        )
        except Exception:  # noqa: BLE001 — never break the controller return path
            logging.exception(
                "ADF/MC validation failed at controller call %d", call_idx
            )
