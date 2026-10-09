"""Unit tests for fine-Δt geometric contact first_contact_actor attribution."""

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np


import tensorflow as tf

from adf_mc_validation.adf_mc_contact import sample_stage_motion_and_ejection


class FirstContactActorTest(tf.test.TestCase):
    def test_first_contact_actor_set_on_box_crossing(self):
        # One particle, CV [x, v], one actuator at x=5; start at x=0, v=1.
        # Stage T=10, M=10 → Δt=1; crosses x=5 at t≈5.
        # HIT phase box covers all time and y (1D: y unused / zeros).
        num_samples = 32
        ms = np.zeros((num_samples, 1, 2), dtype=float)
        ms[..., 0] = 0.0
        ms[..., 1] = 1.0
        living = np.ones((num_samples, 1), dtype=bool)
        lower = np.full((1, 3, 2), np.nan, dtype=float)
        upper = np.full((1, 3, 2), np.nan, dtype=float)
        # HIT phase (m=0): time [0, 10), y [-1e9, 1e9]
        lower[0, 0, 0] = 0.0
        upper[0, 0, 0] = 10.0
        lower[0, 0, 1] = -1e9
        upper[0, 0, 1] = 1e9
        phase_active = np.zeros((1, 3), dtype=bool)
        phase_active[0, 0] = True
        actor_x = np.array([5.0])
        p_phase = np.array([[1.0, 0.0, 0.0]])  # certain HIT
        rng = np.random.default_rng(0)
        # Zero process noise so paths are deterministic
        result = sample_stage_motion_and_ejection(
            ms=ms,
            living=living,
            lower=lower,
            upper=upper,
            phase_active=phase_active,
            actor_x=actor_x,
            p_phase=p_phase,
            S_w=0.0,
            stage_T=10.0,
            num_micro_steps=10,
            rng=rng,
            boundaries=None,
        )
        np.testing.assert_array_equal(result.first_contact_actor[:, 0], 0)
        np.testing.assert_array_equal(result.ejection[:, 0], 1)

    def test_first_contact_actor_minus_one_without_box(self):
        num_samples = 16
        ms = np.zeros((num_samples, 1, 2), dtype=float)
        ms[..., 0] = 0.0
        ms[..., 1] = 1.0
        living = np.ones((num_samples, 1), dtype=bool)
        # No active phases → crossing does not enter a box
        lower = np.full((1, 3, 2), np.nan, dtype=float)
        upper = np.full((1, 3, 2), np.nan, dtype=float)
        phase_active = np.zeros((1, 3), dtype=bool)
        actor_x = np.array([5.0])
        p_phase = np.array([[1.0, 0.0, 0.0]])
        result = sample_stage_motion_and_ejection(
            ms=ms,
            living=living,
            lower=lower,
            upper=upper,
            phase_active=phase_active,
            actor_x=actor_x,
            p_phase=p_phase,
            S_w=0.0,
            stage_T=10.0,
            num_micro_steps=10,
            rng=np.random.default_rng(1),
            boundaries=None,
        )
        np.testing.assert_array_equal(result.first_contact_actor[:, 0], -1)
        np.testing.assert_array_equal(result.ejection[:, 0], 0)

    def test_summarize_mc_eject_actor_counts(self):
        from adf_mc_validation.adf_mc_plots import summarize_mc_eject_actor_counts

        # stage 0 prior all living; stage 1: half dead with actor 2, half still living
        S = 4
        ex0 = np.tile(np.array([[0, 1]]), (S, 1, 1))
        ex1 = np.array(
            [
                [[1, 0]],
                [[1, 0]],
                [[0, 1]],
                [[0, 1]],
            ],
            dtype=int,
        )
        fc0 = np.full((S, 1), -1, dtype=np.int32)
        fc1 = np.array([[2], [2], [-1], [7]], dtype=np.int32)
        mc_diag = {
            "ex_stages": [ex0, ex1],
            "first_contact_actors_stages": [fc0, fc1],
        }
        out = summarize_mc_eject_actor_counts(mc_diag, particle_index=0, stage_k=1)
        self.assertIsNotNone(out)
        self.assertEqual(out["n_ejected"], 2)
        self.assertEqual(out["mc_eject_actor_counts"], {"2": 2})

    def test_summarize_mc_eject_actor_counts_from_stage(self):
        from adf_mc_validation.adf_mc_plots import summarize_mc_eject_actor_counts_from_stage

        # Samples 0,1 die at stage 1 (actor 2); samples 2,3 die at stage 2 (actor 5)
        S = 4
        ex0 = np.tile(np.array([[0, 1]]), (S, 1, 1))
        ex1 = np.array(
            [[[1, 0]], [[1, 0]], [[0, 1]], [[0, 1]]],
            dtype=int,
        )
        ex2 = np.array(
            [[[1, 0]], [[1, 0]], [[1, 0]], [[1, 0]]],
            dtype=int,
        )
        fc0 = np.full((S, 1), -1, dtype=np.int32)
        fc1 = np.array([[2], [2], [-1], [-1]], dtype=np.int32)
        fc2 = np.array([[-1], [-1], [5], [5]], dtype=np.int32)
        mc_diag = {
            "ex_stages": [ex0, ex1, ex2],
            "first_contact_actors_stages": [fc0, fc1, fc2],
        }
        out = summarize_mc_eject_actor_counts_from_stage(
            mc_diag, particle_index=0, stage_k=1, stage_indices=[0, 1, 2]
        )
        self.assertIsNotNone(out)
        self.assertEqual(out["n_ejected"], 2)
        self.assertEqual(out["mc_eject_actor_counts"], {"2": 2})
        self.assertEqual(
            out["mc_eject_actor_counts_by_stage"],
            {"1": {"2": 2}, "2": {"5": 2}},
        )
        self.assertEqual(out["n_ejected_by_stage"], {"1": 2, "2": 2})


if __name__ == "__main__":
    tf.test.main()
