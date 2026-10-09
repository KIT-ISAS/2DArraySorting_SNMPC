import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import tensorflow as tf
import numpy as np

from ptcr_particle_ordering import ParticleOrderingById, ParticleOrderingByLeavingTime, ParticleOrderingByCurrentPosition, \
    ParticleOrderingByProximityToKeepParticle


class AbstractParticleOrderingTest(tf.test.TestCase):
    """Abstract class for testing particle ordering strategies."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.particle_id = np.array([1, 3, 0, 2, 4])

        motion_state_mean = np.array(
            [
                [6, 1.0],
                [2, 1.0],
                [10, 1.0],
                [3, 1.0],
                [7, 1.0],
            ],
            dtype=float
        )
        # [position_x, velocity_x, position_y, velocity_y]
        motion_state_mean_2d = np.array(
            [
                [6, 1.0, 1, 1.0],
                [2, 1.0, 2, 1.0],
                [10, 1.0, 3, 1.0],
                [3, 1.0, 4, 1.0],
                [7, 1.0, 5, 1.0],
            ],
            dtype=float
        )
        motion_state_cov = np.array(
            [
                0.001 * np.diag([1.0, 1.2]),
                0.001 * np.diag([1.1, 0.9]),
                0.001 * np.diag([1.3, 1.1]),
                0.001 * np.diag([0.8, 1.0]),
                0.001 * np.diag([1.2, 1.4]),
            ],
            dtype=float
        )

        cls.particle_states = [motion_state_mean, motion_state_cov]
        cls.particle_states_2d = [motion_state_mean_2d, motion_state_cov]

        cls.particle_class = np.array([0, 1, 0, 0, 0], dtype=int)


class ParticleOrderingByIdTest(AbstractParticleOrderingTest):
    """Test cases for the ParticleOrderingById class."""

    def test_sort(self):
        sorter = ParticleOrderingById()

        outp_sort_order = sorter.sort(particle_id=self.particle_id)
        expected_sort_order = np.array([2, 0, 3, 1, 4])

        # Check if the particles are sorted correctly
        self.assertAllEqual(outp_sort_order, expected_sort_order)


class ParticleOrderingByLeavingTimeTest(AbstractParticleOrderingTest):
    """Test cases for the ParticleOrderingByLeavingTime class."""

    def test_sort(self):
        sorter = ParticleOrderingByLeavingTime(array_end=10)
        """ motion_state_mean = np.array(
            [
                [6, 1.0],
                [2, 1.0],
                [10, 1.0],
                [3, 1.0],
                [7, 1.0],
            ],
            dtype=float
        ) 
        
        expected_sorted_particle_states_mean = np.array(
            [
            [10, 1.0],
            [7, 1.0],
            [6, 1.0],
            [3, 1.0],
            [2, 1.0],
            ], dtype=float
        )
        """
        outp_sort_order = sorter.sort(particle_states=self.particle_states, )
        expected_sort_order = np.array([2, 4, 0, 3, 1])

        self.assertAllEqual(expected_sort_order, outp_sort_order)


class ParticleOrderingByCurrentPositionTest(AbstractParticleOrderingTest):
    """Test cases for the ParticleOrderingByCurrentPosition class."""
    sorter = ParticleOrderingByCurrentPosition()

    def test_sort(self):
        with self.subTest(name='Test where particle states are 1D'):
            outp_sort_order = self.sorter.sort(particle_states=self.particle_states)
            expected_sort_order = np.array([2, 4, 0, 3, 1])
            self.assertAllEqual(expected_sort_order, outp_sort_order)

        with self.subTest(name='Test where particle states are 2D'):
            outp_sort_order_2d = self.sorter.sort(particle_states=self.particle_states_2d)
            expected_sort_order_2d = np.array([2, 4, 0, 3, 1])
            self.assertAllEqual(expected_sort_order_2d, outp_sort_order_2d)


class ParticleOrderingByProximityToKeepParticleTest(AbstractParticleOrderingTest):
    """Test cases for the ParticleOrderingByProximityToKeepParticle class."""
    sorter = ParticleOrderingByProximityToKeepParticle()

    def test_sort(self):
        with self.subTest(name='Test where particle states are 1D'):
            outp_sort_order = self.sorter.sort(particle_states=self.particle_states,
                                          particle_class=np.array([1, 1, 0, 0, 1], dtype=int))
            expected_sort_order = np.array([1, 0, 4, 2, 3])
            self.assertAllEqual(expected_sort_order, outp_sort_order)

        with self.subTest(name='Test where particle states are 2D'):
            outp_sort_order_2d = self.sorter.sort(particle_states=self.particle_states_2d,
                                             particle_class=np.array([1, 1, 0, 0, 1], dtype=int))
            expected_sort_order_2d = np.array([1, 4, 0, 2, 3])
            self.assertAllEqual(expected_sort_order_2d, outp_sort_order_2d)


if __name__ == "__main__":
    tf.test.main()
