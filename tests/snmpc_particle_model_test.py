import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np

import tensorflow as tf

from snmpc_particle_model import ParticleModel


class ParticleModelTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        T = 1
        p_detect = 1.0
        p_clutter = 0.0
        S_w = 0.001
        S_v = 0.01

        cls.particle_model_1d = ParticleModel(T=T, p_detect=p_detect, p_clutter=p_clutter, S_w=S_w, S_v=S_v, seed=1234)
        cls.particle_model_2d = ParticleModel(T=T, p_detect=p_detect, p_clutter=p_clutter, S_w=(S_w, S_w),
                                              S_v=(S_v, S_v), seed=1234)

    def test_predict_motion(self):
        with self.subTest(name='Test for 1D motion'):
            motion_state_mean = np.array([[0.0, 1.0], [2.0, 4.0], [3.0, 5.0]])
            motion_state_cov = np.stack(3 * [np.eye(2) * 0.001], axis=0)

            mean_outp, cov_outp = self.particle_model_1d.predict_motion(motion_state_mean, motion_state_cov)
            self.assertAllClose(np.array([[1.0, 1.0], [6.0, 4.0], [8.0, 5.0]]), mean_outp, atol=0.3)
            self.assertAllEqual(motion_state_cov.shape, cov_outp.shape)

        with self.subTest(name='Test for 2D motion'):
            motion_state_mean = np.array([[0.0, 1.0, 4.0, 0.0], [2.0, 4.0, 3.0, 0.0], [3.0, 5.0, 2.0, 2.0]])
            motion_state_cov = np.stack(3 * [np.eye(4) * 0.001], axis=0)

            mean_outp, cov_outp = self.particle_model_2d.predict_motion(motion_state_mean, motion_state_cov)
            self.assertAllClose(np.array([[1.0, 1.0, 4.0, 0.0],
                                          [6.0, 4.0, 3.0, 0.0],
                                          [8.0, 5.0, 4.0, 2.0]]), mean_outp, atol=0.3)
            self.assertAllEqual(motion_state_cov.shape, cov_outp.shape)

    def test_predict_existence(self):
        existence_probs = np.array([[0, 1]])
        p_eject = np.array([0])
        existence_probs_outp = self.particle_model_1d.predict_existence(existence_probs, p_eject)
        self.assertAllEqual(np.array([[0, 1]]), existence_probs_outp)

        p_eject = np.array([1])
        existence_probs_outp = self.particle_model_1d.predict_existence(existence_probs, p_eject)
        self.assertAllEqual(np.array([[1, 0]]), existence_probs_outp)

        existence_probs = np.array([[0, 1], [0, 1], [1, 0]])
        p_eject = np.array([0.9, 0, 1])
        existence_probs_outp = self.particle_model_1d.predict_existence(existence_probs, p_eject)
        self.assertAllClose(np.array([[0.9, 0.1], [0, 1], [1, 0]]), existence_probs_outp)

    def test_measure_position(self):
        with self.subTest(name='Test for 1D motion'):
            motion_state_mean = np.array([[0.0, 1.0], [2.0, 4.0], [3.0, 5.0]])
            motion_state_cov = np.stack(3 * [np.eye(2) * 0.001], axis=0)

            mean_outp, cov_outp = self.particle_model_1d.measure_position(motion_state_mean, motion_state_cov)
            self.assertAllClose(np.array([[0.0], [2.0], [3.0]]), mean_outp, atol=0.3)
            self.assertAllEqual((3, 1, 1), cov_outp.shape)

        with self.subTest(name='Test for 2D motion'):
            motion_state_mean = np.array([[0.0, 1.0, 4.0, 0.0], [2.0, 4.0, 3.0, 0.0], [3.0, 5.0, 2.0, 2.0]])
            motion_state_cov = np.stack(3 * [np.eye(4) * 0.001], axis=0)

            mean_outp, cov_outp = self.particle_model_2d.measure_position(motion_state_mean, motion_state_cov)
            self.assertAllClose(np.array([[0.0, 4.0], [2.0, 3.0], [3.0, 2.0]]), mean_outp, atol=0.3)
            self.assertAllEqual((3, 2, 2), cov_outp.shape)

    def test_measure_existence(self):
        b_outp = self.particle_model_1d.measure_existence(np.array([[0, 1]], dtype=float))
        self.assertAllEqual(np.array([[0, 1]]), b_outp)

        b_outp = self.particle_model_1d.measure_existence(np.array([[1, 0]], dtype=float))
        self.assertAllEqual(np.array([[1, 0]]), b_outp)

        b_outp = self.particle_model_1d.measure_existence(np.array([[0, 1], [1, 0], [1, 0]], dtype=float))
        self.assertAllEqual(np.array([[0, 1], [1, 0], [1, 0]]), b_outp)

    def test_sample_motion_transition(self):
        with self.subTest(name='Test for 1D motion'):
            sampled_motion_state = np.array([[0.0, 1.0], [2.0, 4.0], [3.0, 5.0]])
            sampled_motion_state_outp = self.particle_model_1d.sample_motion_transition(sampled_motion_state)
            self.assertAllClose(np.array([[1.0, 1.0], [6.0, 4.0], [8.0, 5.0]]), sampled_motion_state_outp, atol=0.3)

        with self.subTest(name='Test for 2D motion'):
            sampled_motion_state = np.array([[0.0, 1.0, 4.0, 0.0], [2.0, 4.0, 3.0, 0.0], [3.0, 5.0, 2.0, 2.0]])
            sampled_motion_state_outp = self.particle_model_2d.sample_motion_transition(sampled_motion_state)
            self.assertAllClose(np.array([[1.0, 1.0, 4.0, 0.0],
                                          [6.0, 4.0, 3.0, 0.0],
                                          [8.0, 5.0, 4.0, 2.0]]), sampled_motion_state_outp, atol=0.3)

    def test_sample_existence_transition(self):
        sampled_existence_state = np.array([[0, 1]])
        sampled_ejections = np.array([0])
        sampled_existence_state_outp = self.particle_model_1d.sample_existence_transition(sampled_existence_state,
                                                                                          sampled_ejections)
        self.assertAllEqual(np.array([[0, 1]]), sampled_existence_state_outp)

        sampled_ejections = np.array([1])
        sampled_existence_state_outp = self.particle_model_1d.sample_existence_transition(sampled_existence_state,
                                                                                          sampled_ejections)
        self.assertAllEqual(np.array([[1, 0]]), sampled_existence_state_outp)

        sampled_existence_state = np.array([[0, 1], [0, 1], [1, 0]])
        sampled_ejections = np.array([1, 0, 1])
        sampled_existence_state_outp = self.particle_model_1d.sample_existence_transition(sampled_existence_state,
                                                                                          sampled_ejections)
        self.assertAllEqual(np.array([[1, 0], [0, 1], [1, 0]]), sampled_existence_state_outp)

    def test_sample_position_measurement(self):
        with self.subTest(name='Test for 1D motion'):
            sampled_motion_state = np.array([[0.0, 1.0], [2.0, 4.0], [3.0, 5.0]])
            p_meas = self.particle_model_1d.sample_position_measurement(sampled_motion_state)
            self.assertAllClose(np.array([[0.0], [2.0], [3.0]]), p_meas, atol=0.3)

        with self.subTest(name='Test for 2D motion'):
            sampled_motion_state = np.array([[0.0, 1.0, 4.0, 0.0], [2.0, 4.0, 3.0, 0.0], [3.0, 5.0, 2.0, 2.0]])
            p_meas = self.particle_model_2d.sample_position_measurement(sampled_motion_state)
            self.assertAllClose(np.array([[0.0, 4.0], [2.0, 3.0], [3.0, 2.0]]), p_meas, atol=0.3)

    def test_sample_existence_measurement(self):
        b_meas = self.particle_model_1d.sample_existence_measurement(np.array([[0, 1]], dtype=float))
        self.assertAllEqual(np.array([1]), b_meas)

        b_meas = self.particle_model_1d.sample_existence_measurement(np.array([[1, 0]], dtype=float))
        self.assertAllEqual(np.array([0]), b_meas)

        b_meas = self.particle_model_1d.sample_existence_measurement(np.array([[0, 1], [1, 0], [1, 0]],
                                                                              dtype=float))
        self.assertAllEqual(np.array([1, 0, 0]), b_meas)

    def test_calculate_motion_transition_log_likelihood_test(self):
        with self.subTest(name='Test for 1D motion'):
            motion_state = np.array([[0.0, 1.0], [2.0, 4.0], [3.0, 5.0]])
            next_motion_state = np.array([[1.0, 1.0], [6.0, 4.0], [8.0, 5.0]])

            log_likelihood = self.particle_model_1d.calculate_motion_transition_log_likelihood(
                next_motion_state, motion_state)
            exp_log_l = - np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(self.particle_model_1d._C_w))
            self.assertAllClose(exp_log_l * np.ones(3), log_likelihood)

        with self.subTest(name='Test for 2D motion'):
            motion_state = np.array([[0.0, 1.0, 4.0, 0.0],
                                     [2.0, 4.0, 3.0, 0.0],
                                     [3.0, 5.0, 2.0, 2.0]])
            next_motion_state = np.array([[1.0, 1.0, 4.0, 0.0],
                                          [6.0, 4.0, 3.0, 0.0],
                                          [8.0, 5.0, 4.0, 2.0]])

            log_likelihood = self.particle_model_2d.calculate_motion_transition_log_likelihood(
                next_motion_state, motion_state)
            exp_log_l = - 2 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(self.particle_model_2d._C_w))
            self.assertAllClose(exp_log_l * np.ones(3), log_likelihood)

    def test_calculate_existence_transition_log_likelihood(self):
        existence_state = np.array([[0, 1], [0, 1],
                                    [1, 0], [1, 0]])
        next_existence_state = np.array([[0, 1], [1, 0],
                                         [0, 1], [1, 0]])
        p_eject = np.array([0.9, 0.7, 0, 0.6])

        log_likelihood = self.particle_model_1d.calculate_existence_transition_log_likelihood(
            next_existence_state, existence_state, p_eject)
        with np.errstate(divide='ignore'):
            exp_log_l = np.log(np.array([0.1, 0.7, 0, 1]))
        self.assertAllClose(exp_log_l, log_likelihood)


if __name__ == "__main__":
    tf.test.main()
