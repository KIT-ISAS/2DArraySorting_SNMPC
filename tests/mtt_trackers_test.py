"""Unit tests for TrackSortMTTTracker feed filters."""

import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np


import tensorflow as tf

from mtt_trackers import TrackSortMTTTracker


def _bare_tracker(**kwargs):
    """Construct TrackSortMTTTracker without building the Adaptive_Filtering model."""
    obj = TrackSortMTTTracker.__new__(TrackSortMTTTracker)
    obj._array_end = kwargs.get("array_end", 100.0)
    obj._area_width = kwargs.get("area_width", 50.0)
    obj._measurement_interval = kwargs.get("measurement_interval", 20.0)
    obj._feed_track_scores_to_controller = kwargs.get("feed_track_scores_to_controller", False)
    obj._feed_tracks_with_position_greater = kwargs.get("feed_tracks_with_position_greater", None)
    obj._time_step = 0
    obj._track_info = {
        "n_associated_meas": np.empty((0,), dtype=int),
        "last_meas_idx": np.empty((0,), dtype=int),
    }
    return obj


class TrackSortFeedFiltersTest(tf.test.TestCase):
    def test_position_gate_keeps_only_tracks_past_threshold(self):
        tracker = _bare_tracker(feed_tracks_with_position_greater=40.0)

        mean = np.array(
            [
                [10.0, 20.0, 5.0, 0.0],   # too early
                [41.0, 20.0, 5.0, 0.0],   # keep
                [40.0, 20.0, 5.0, 0.0],   # boundary: strict >
                [80.0, 20.0, 5.0, 0.0],   # keep
            ],
            dtype=float,
        )
        cov = np.stack([np.eye(4) for _ in range(4)])
        labels = np.array([0, 1, 0, 1], dtype=int)
        track_ids = np.array([1, 2, 3, 4], dtype=int)
        scores = np.array([0.5, 0.6, 0.7, 0.8], dtype=float)
        is_artificial = np.array([False, False, False, False])
        info = {
            "n_associated_meas": np.array([2, 9, 10, 12], dtype=int),
            "last_meas_idx": np.array([1, 8, 9, 11], dtype=int),
        }

        def fake_ts_model(measurements, time_step):
            del measurements, time_step
            return None, (mean, cov, labels, track_ids, scores, is_artificial, info)

        tracker._ts_model = fake_ts_model

        out_mean, out_cov, out_lab, out_id, out_score = tracker(
            np.array([[50.0, 10.0, 1.0]])
        )
        self.assertEqual(out_mean.shape[0], 2)
        np.testing.assert_array_equal(out_mean[:, 0], [41.0, 80.0])
        np.testing.assert_array_equal(out_id, [2, 4])
        np.testing.assert_array_equal(tracker.track_info["n_associated_meas"], [9, 12])
        np.testing.assert_array_equal(tracker.track_info["last_meas_idx"], [8, 11])
        # velocity denorm: / measurement_interval
        np.testing.assert_allclose(out_mean[:, 1], [1.0, 1.0])

    def test_position_gate_none_keeps_all_non_artificial(self):
        tracker = _bare_tracker(feed_tracks_with_position_greater=None)

        mean = np.array([[10.0, 20.0, 5.0, 0.0], [80.0, 20.0, 5.0, 0.0]], dtype=float)
        cov = np.stack([np.eye(4), np.eye(4)])
        labels = np.array([0, 1], dtype=int)
        track_ids = np.array([1, 2], dtype=int)
        scores = np.array([0.5, 0.9], dtype=float)
        is_artificial = np.array([True, False])
        info = {
            "n_associated_meas": np.array([1, 8], dtype=int),
            "last_meas_idx": np.array([0, 7], dtype=int),
        }

        def fake_ts_model(measurements, time_step):
            del measurements, time_step
            return None, (mean, cov, labels, track_ids, scores, is_artificial, info)

        tracker._ts_model = fake_ts_model
        out_mean, _, _, out_id, out_score = tracker(np.empty((0, 3)))
        self.assertEqual(out_mean.shape[0], 1)
        np.testing.assert_array_equal(out_id, [2])
        np.testing.assert_array_equal(out_score, [1.0])
        np.testing.assert_array_equal(tracker.track_info["n_associated_meas"], [8])


if __name__ == "__main__":
    tf.test.main()
