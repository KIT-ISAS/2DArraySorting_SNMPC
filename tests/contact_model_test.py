import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np

import tensorflow as tf

from actor_model import ActorModelVariantB
from contact_model import AbstractContactModel


class _AbstractContactModelStub(AbstractContactModel):
    """Minimal concrete subclass to exercise the retained AbstractContactModel helpers."""

    def predict_ejection(self, motion_state_mean, motion_state_cov, particle_class, *args):
        return np.zeros(motion_state_mean.shape[:-1])


class AbstractContactModelTest(tf.test.TestCase):
    """Test cases for the AbstractContactModel class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        actor_dict = {'t_activate': 1,
                      't_up': 1,
                      't_hit': 2,
                      't_down': 1,
                      't_reset': 4}
        actor_positions = np.array([10, 13, 16, 19])
        length = 0.5

        actor_model = ActorModelVariantB(actor_positions, length=length, T=6, **actor_dict)
        cls.contact_model = _AbstractContactModelStub(actor_model)

    def test_calculate_ejection_probability_by_product(self):
        p_eject_i_at_j_1 = np.array([[0, 0.2, 0.8, 0, 0],
                                     [0.2, 0.8, 0.4, 0, 0],
                                     [1, 0, 0.1, 0.1, 0],
                                     [0, 0, 0, 0, 0]])
        p_eject_i_at_j_2 = np.array([[0, 0.3, 0.8, 0, 0],
                                     [0.2, 0.9, 0.4, 0, 0],
                                     [1, 0, 0.2, 0.1, 0],
                                     [0, 0, 0.8, 0, 0]])

        with self.subTest(name='Test with batch shape.'):
            p_eject_i_at_j = np.stack([p_eject_i_at_j_1, p_eject_i_at_j_2], axis=0)
            p_eject = self.contact_model._calculate_ejection_probability_by_product(p_eject_i_at_j)
            self.assertAllEqual(np.array([2, 5]), p_eject.shape)
            self.assertAllClose(np.array([1, 0.2 + 0.8 * 0.8, 0.8 + 0.2 * 0.4 + (1 - (0.8 + 0.2 * 0.4)) * 0.1, 0.1, 0]),
                                p_eject[0])

        with self.subTest(name='Test without batch shape.'):
            p_eject = self.contact_model._calculate_ejection_probability_by_product(p_eject_i_at_j_1)
            self.assertAllClose(np.array([1, 0.2 + 0.8 * 0.8, 0.8 + 0.2 * 0.4 + (1 - (0.8 + 0.2 * 0.4)) * 0.1, 0.1, 0]),
                                p_eject)

    def test_calculate_ejection_probability_by_sum(self):
        p_eject_i_at_j_1 = np.array([[0, 0.2, 0.8, 0, 0],
                                     [0.2, 0.8, 0.4, 0, 0],
                                     [1, 0, 0.1, 0.1, 0],
                                     [0, 0, 0, 0, 0]])
        p_eject_i_at_j_2 = np.array([[0, 0.3, 0.8, 0, 0],
                                     [0.2, 0.9, 0.4, 0, 0],
                                     [1, 0, 0.2, 0.1, 0],
                                     [0, 0, 0.8, 0, 0]])

        with self.subTest(name='Test without batch shape.'):
            p_eject_i_at_j = np.stack([p_eject_i_at_j_1, p_eject_i_at_j_2], axis=0)
            p_eject = self.contact_model._calculate_ejection_probability_by_sum(p_eject_i_at_j)
            self.assertAllEqual(np.array([2, 5]), p_eject.shape)
            self.assertAllEqual(np.array([1, 1, 1, 0.1, 0]), p_eject[0])

        with self.subTest(name='Test without batch shape.'):
            p_eject = self.contact_model._calculate_ejection_probability_by_sum(p_eject_i_at_j_1)
            self.assertAllEqual(np.array([1, 1, 1, 0.1, 0]), p_eject)

