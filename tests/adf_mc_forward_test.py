"""Smoke tests: ADF vs fine-``Δt`` Monte-Carlo OLF costs under small uncertainty."""

from __future__ import annotations

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np


import tensorflow as tf

from adf_mc_validation.adf_mc_fixture import build_2d_controller, default_scenario
from adf_mc_validation.adf_mc_plots import (
    find_partial_hit_particle,
    partial_hit_stage_window,
    plot_position_histograms_multistage,
)
from adf_mc_validation.adf_mc_cost_eval import (
    AdfMcCostEvalAccumulator,
    evaluate_adf_mc_terminal_costs,
    plot_adf_mc_cost_eval_summary,
)


class AdfMcForwardSmokeTest(tf.test.TestCase):
    def test_costs_close_for_small_uncertainty(self):
        mu0, P0, pi0, particle_class, t_act, u = default_scenario(cov_scale=1e-5)
        controller = build_2d_controller(T=5.0, S_w=1e-8, seed=123)
        controller._particle_model.particle_class = particle_class

        j_adf = controller._forward_chain(u, mu0, P0, pi0, t_act)
        j_mc, diag = controller._forward_chain_mc(
            u,
            mu0,
            P0,
            pi0,
            t_act,
            num_samples=2500,
            num_micro_steps=100,
            return_diagnostics=True,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(j_adf.shape, (1,))
        self.assertEqual(j_mc.shape, (1,))
        tol = 3.0 * float(diag["cost_stderr"][0]) + 0.05
        self.assertLess(
            abs(float(j_adf[0]) - float(j_mc[0])),
            tol,
            msg=f"ADF={j_adf[0]:.4f}, MC={j_mc[0]:.4f}±{diag['cost_stderr'][0]:.4f}",
        )

    def test_forward_chain_states_shapes(self):
        mu0, P0, pi0, particle_class, t_act, u = default_scenario(cov_scale=1e-3)
        controller = build_2d_controller(T=5.0, S_w=1e-4, seed=1)
        controller._particle_model.particle_class = particle_class
        cost, states = controller._forward_chain_states(u, mu0, P0, pi0, t_act)
        self.assertEqual(cost.shape, (1,))
        self.assertEqual(len(states["pi_stages"]), u.shape[2] + 1)
        self.assertEqual(states["mu_stages"][0].shape, (1, 2, 2, 4))
        self.assertEqual(states["P_stages"][0].shape, (1, 2, 2, 4, 4))

    def test_mc_diagnostics_shapes(self):
        mu0, P0, pi0, particle_class, t_act, u = default_scenario(cov_scale=1e-3)
        controller = build_2d_controller(T=5.0, S_w=1e-4, seed=1)
        controller._particle_model.particle_class = particle_class
        cost, diag = controller._forward_chain_mc(
            u,
            mu0,
            P0,
            pi0,
            t_act,
            num_samples=200,
            num_micro_steps=40,
            return_diagnostics=True,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(cost.shape, (1,))
        self.assertEqual(len(diag["pi_stages"]), u.shape[2] + 1)
        self.assertEqual(diag["ms_stages"][0].shape, (200, 2, 4))

    def test_gt_box_mc_means_close_to_adf(self):
        """ADF soft-Bernoulli + GT maps should match gt_box MC (not geometric)."""
        mu0, P0, pi0, particle_class, t_act, _ = default_scenario(cov_scale=1.0)
        controller = build_2d_controller(T=5.0, S_w=1e-2, seed=1)
        controller._particle_model.particle_class = particle_class
        u = np.full((1, 2, 8), 0.5, dtype=float)

        _j, adf = controller._forward_chain_states(u, mu0, P0, pi0, t_act)
        _j_mc, mc = controller._forward_chain_mc(
            u,
            mu0,
            P0,
            pi0,
            t_act,
            num_samples=2500,
            num_micro_steps=40,
            return_diagnostics=True,
            rng=np.random.default_rng(7),
            contact_mode="gt_box",
        )
        k = 8
        p = 0
        mu_adf = adf["mu_stages"][k][0, p]
        ms = mc["ms_stages"][k][:, p, 0]
        ex = mc["ex_stages"][k][:, p]
        for mode in (0, 1):
            sel = ms[ex[:, mode] == 1]
            self.assertGreater(sel.size, 20)
            self.assertLess(
                abs(float(mu_adf[mode, 0]) - float(sel.mean())),
                2.0,
                msg=f"mode {mode}: ADF={mu_adf[mode, 0]:.3f} GT-MC={sel.mean():.3f}",
            )

    def test_multistage_hist_plot_saves(self):
        import tempfile

        mu0, P0, pi0, particle_class, t_act, u = default_scenario(cov_scale=1e-3)
        controller = build_2d_controller(T=5.0, S_w=1e-4, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = os.path.join(tmp, "hist.png")
            plot_position_histograms_multistage(
                controller,
                mu0,
                P0,
                pi0,
                particle_class,
                t_act,
                u,
                particle_index=0,
                num_samples=150,
                num_micro_steps=30,
                seed=0,
                save_path=hist_path,
                show=False,
                particle_id=99,
            )
            self.assertTrue(os.path.isfile(hist_path))
            self.assertGreater(os.path.getsize(hist_path), 1000)


class PartialHitSelectionTest(tf.test.TestCase):
    def test_find_partial_hit_prefers_reject_closest_to_half(self):
        # stage 0 prior all living; stage 1/2 mid-range on reject idx 1
        pi0 = np.array([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]])
        pi1 = np.array([[[0.05, 0.95], [0.45, 0.55], [0.2, 0.8]]])
        pi2 = np.array([[[0.1, 0.9], [0.35, 0.65], [0.5, 0.5]]])
        # accept at idx 2 also mid-range at stage 2, but reject idx 1 closer to 0.5 at stage 1
        particle_class = np.array([0, 1, 0], dtype=int)
        hit = find_partial_hit_particle(
            [pi0, pi1, pi2], particle_class, pi_lo=0.3, pi_hi=0.7, prefer_reject=True
        )
        self.assertIsNotNone(hit)
        p_idx, stage_idx, pi_live = hit
        self.assertEqual(p_idx, 1)
        self.assertEqual(stage_idx, 1)
        self.assertAlmostEqual(pi_live, 0.55)

    def test_find_partial_hit_none_when_outside_band(self):
        pi0 = np.array([[[0.0, 1.0], [0.0, 1.0]]])
        pi1 = np.array([[[0.05, 0.95], [0.9, 0.1]]])
        hit = find_partial_hit_particle(
            [pi0, pi1], np.array([1, 1]), pi_lo=0.3, pi_hi=0.7
        )
        self.assertIsNone(hit)

    def test_find_partial_hit_filters_young_tracks(self):
        pi0 = np.array([[[0.0, 1.0], [0.0, 1.0]]])
        # idx 0 mid-range but young; idx 1 mid-range and mature
        pi1 = np.array([[[0.45, 0.55], [0.48, 0.52]]])
        particle_class = np.array([1, 1], dtype=int)
        hit_young = find_partial_hit_particle(
            [pi0, pi1],
            particle_class,
            pi_lo=0.3,
            pi_hi=0.7,
            n_associated_meas=np.array([2, 10]),
            min_n_associated_meas=8,
        )
        self.assertIsNotNone(hit_young)
        self.assertEqual(hit_young[0], 1)

        hit_none = find_partial_hit_particle(
            [pi0, pi1],
            particle_class,
            pi_lo=0.3,
            pi_hi=0.7,
            n_associated_meas=np.array([2, 3]),
            min_n_associated_meas=8,
        )
        self.assertIsNone(hit_none)

    def test_partial_hit_stage_window(self):
        self.assertEqual(partial_hit_stage_window(1, 5), [0, 1, 2, 3, 4])
        self.assertIsNone(partial_hit_stage_window(1, 4))
        self.assertIsNone(partial_hit_stage_window(0, 5))
        self.assertEqual(partial_hit_stage_window(2, 6), [1, 2, 3, 4, 5])


class AdfMcCostEvalTest(tf.test.TestCase):
    def test_raises_without_terminal_costs(self):
        mu0, P0, pi0, particle_class, t_act, u = default_scenario(cov_scale=1e-3)
        controller = build_2d_controller(
            T=5.0, S_w=1e-4, seed=1, use_only_terminal_costs=False
        )
        controller._particle_model.particle_class = particle_class
        with self.assertRaises(ValueError):
            evaluate_adf_mc_terminal_costs(
                controller,
                u,
                mu0,
                P0,
                pi0,
                t_act,
                particle_class,
                num_samples=50,
                num_micro_steps=20,
                seed=0,
            )

    def test_eval_all_sequences_and_summary(self):
        mu0, P0, pi0, particle_class, t_act, u0 = default_scenario(cov_scale=1e-3)
        controller = build_2d_controller(T=5.0, S_w=1e-4, seed=1)
        controller._particle_model.particle_class = particle_class
        # Two candidate sequences (batch)
        u = np.concatenate([u0, u0], axis=0)
        u[1, 0, 0] = 1.0
        j_adf = controller._forward_chain(u, mu0, P0, pi0, t_act)
        result = evaluate_adf_mc_terminal_costs(
            controller,
            u,
            mu0,
            P0,
            pi0,
            t_act,
            particle_class,
            num_samples=200,
            num_micro_steps=30,
            seed=3,
            call_index=0,
            j_adf=j_adf,
        )
        self.assertEqual(result.j_adf.shape, (2,))
        self.assertEqual(result.j_mc.shape, (2,))
        self.assertEqual(result.delta_per_particle.shape, (2,))
        np.testing.assert_allclose(
            result.delta_per_particle,
            (result.j_mc - result.j_adf) / mu0.shape[0],
        )
        acc = AdfMcCostEvalAccumulator()
        acc.record(result)
        # Second fake call with swapped preference possible
        acc.record(result)
        summary = acc.summary()
        self.assertEqual(summary["num_calls"], 2)
        self.assertEqual(summary["num_sequence_evals"], 4)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cost_eval.png")
            _fig, s2 = plot_adf_mc_cost_eval_summary(acc, save_path=path, show=False)
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(os.path.isfile(path.replace(".png", "_summary.json")))
            csv_dir = path.replace(".png", "_csv")
            self.assertTrue(os.path.isdir(csv_dir))
            self.assertTrue(os.path.isfile(os.path.join(csv_dir, "delta_per_particle.csv")))
            self.assertTrue(os.path.isfile(os.path.join(csv_dir, "regret_when_disagree.csv")))
            self.assertTrue(os.path.isfile(os.path.join(csv_dir, "summary.csv")))
            self.assertTrue(os.path.isfile(os.path.join(csv_dir, "meta.json")))
            self.assertIn("decision_disagree_rate", s2)


if __name__ == "__main__":
    tf.test.main()
