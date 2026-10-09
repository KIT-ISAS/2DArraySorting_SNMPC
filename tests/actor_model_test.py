import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np

import tensorflow as tf

from actor_model import ActorModelVariantA, ActorModelVariantB, ActorModelVariantB2


class AbstractActorModelTest(tf.test.TestCase):
    """Test cases for the AbstractActorModel class."""

    def test_one_dimensional_array(self):
        actor_positions = np.array([10, 13, 16, 19])
        T = 1.0
        actor_dict = {'t_activate': 1,
                      't_up': 1,
                      't_hit': 2,
                      't_down': 1,
                      't_reset': 4}

        with self.subTest(name='Test with valid values'):
            length = 0.5
            actor_model = ActorModelVariantA(actor_positions,
                                             length=length,
                                             T=T, **actor_dict)
            self.assertEqual(4, actor_model.num_rows)
            self.assertAllEqual(np.ones(4), actor_model.actors_in_row)
            exp_actor_grid = np.array([[[10, 0.5]], [[13, 0.5]], [[16, 0.5]], [[19, 0.5]]])
            self.assertAllEqual(exp_actor_grid, actor_model.actor_grid)
            self.assertAllEqual(actor_positions, actor_model.row_pos)
            self.assertIsNone(actor_model.column_pos)

        with self.subTest(name='Test with invalid values'):
            length = (0.5, 0.5, 3.5, 5)
            with self.assertRaisesWithLiteralMatch(ValueError,
                                                   'The operating ranges of the actors 3 and 4 must not overlap.'):
                actor_model = ActorModelVariantA(actor_positions,
                                                 length=length,
                                                 T=T, **actor_dict)

    def test_two_dimensional_array(self):
        actor_positions = np.array([[45., 13.5], [45., 29.5], [65., 7.5], [65., 23.5]])
        T = 1.0
        actor_dict = {'t_activate': 1,
                      't_up': 1,
                      't_hit': 2,
                      't_down': 1,
                      't_reset': 4}
        with self.subTest(name='Test with valid values'):
            length = 4
            width = 9
            actor_model = ActorModelVariantA(actor_positions,
                                             length=length,
                                             width=width,
                                             T=T, **actor_dict)
            self.assertEqual(2, actor_model.num_rows)
            self.assertAllEqual(2 * np.ones(2), actor_model.actors_in_row)
            self.assertAllEqual((45, 65), actor_model.row_pos)
            for i, exp_column_pos in enumerate(((13.5, 29.5), (7.5, 23.5))):
                self.assertAllEqual(exp_column_pos, actor_model.column_pos[i])
            exp_actor_grid = [np.array([[45., 13.5, 4, 9], [45., 29.5, 4, 9]]),
                              np.array([[65., 7.5, 4, 9], [65., 23.5, 4, 9]])]
            self.assertAllEqual(exp_actor_grid, actor_model.actor_grid)

        with self.subTest(name='Test with invalid values'):
            length = (24, 4, 24, 4)
            width = 9
            with self.assertRaisesWithLiteralMatch(
                    ValueError,
                    'The operating ranges of the actors "row 1, column 1 and row 2, column 1" must not overlap.'):
                actor_model = ActorModelVariantA(actor_positions,
                                                 length=length,
                                                 width=width,
                                                 T=T, **actor_dict)

            length = 4
            width = (17, 17, 9, 9)
            with self.assertRaisesWithLiteralMatch(
                    ValueError,
                    'The operating ranges of the actors "row 1, column 1 and 2" must not overlap.'):
                actor_model = ActorModelVariantA(actor_positions,
                                                 length=length,
                                                 width=width,
                                                 T=T, **actor_dict)


class ActorModelVariantB2Test(tf.test.TestCase):
    """Test cases for the ActorModelVariantB class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        actor_dict = {'t_activate': 1,
                      't_up': 1,
                      't_hit': 2,
                      't_down': 1,
                      't_reset': 4}
        actor_positions = np.array([[10, 0], [13, 0], [16, 0], [19, 0]])
        length = 0.5

        cls.actor_model = ActorModelVariantB2(actor_positions, length=length, width=0, T=6, **actor_dict)

    def test_predict_actor_state(self):
        actors_next_time_ready = np.array([0, 2, 5, 8])
        delta_toa = np.array([[3, 5, 1],
                              [7, 9, 5],
                              [4, 6, 2],
                              [8, 10, 6]])
        actor_indices = np.array([0, 2])

        actors_next_time_ready_outp = self.actor_model.predict_actor_state(actors_next_time_ready, delta_toa,
                                                                           actor_indices=actor_indices,
                                                                           particle_indices=0)
        exp_actors_next_time_ready = np.array([9, 2, 10, 8])
        self.assertAllEqual(exp_actors_next_time_ready, actors_next_time_ready_outp)

        actors_next_time_ready = np.array([0, 2, 5, 8])
        particle_indices = np.array([0, 1])
        actors_next_time_ready_outp = self.actor_model.predict_actor_state(actors_next_time_ready, delta_toa,
                                                                           actor_indices=actor_indices,
                                                                           particle_indices=particle_indices)
        exp_actors_next_time_ready = np.array([9, 2, 12, 8])
        self.assertAllEqual(exp_actors_next_time_ready, actors_next_time_ready_outp)

        actors_next_time_ready = np.array([0, 2, 5, 8])
        delta_toa = np.array([[3, 5],
                              [7, 9],
                              [4, 6],
                              [8, 10]])
        actor_indices = np.array([0, 2])
        particle_indices = np.array([1, 0])

        actors_next_time_ready_outp = self.actor_model.predict_actor_state(actors_next_time_ready, delta_toa,
                                                                           actor_indices=actor_indices,
                                                                           particle_indices=particle_indices)
        exp_actors_next_time_ready = np.array([11, 2, 10, 8])
        self.assertAllEqual(exp_actors_next_time_ready, actors_next_time_ready_outp)


class ActorModelVariantBTest(tf.test.TestCase):
    """Test cases for the ActorModelVariantB class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        actor_dict = {'t_activate': 1,
                      't_up': 1,
                      't_hit': 2,
                      't_down': 1,
                      't_reset': 4}
        actor_positions = np.array([[10, 0], [13, 0], [16, 0], [19, 0]])
        length = 0.5

        cls.actor_model = ActorModelVariantB(actor_positions, length=length, width=0, T=6, **actor_dict)
        cls.actor_model_edge_case = ActorModelVariantB(actor_positions, length=length, width=0, T=9, **actor_dict)
        # this is the edge case where T = t_cycle

    def test_continuous_actor_dynamics(self):
        with self.subTest(name='Test with T < t_cycle'):
            with self.assertRaisesWithLiteralMatch(ValueError, 'The difference time t must be within [0, T].'):
                t_act_k_at_t = self.actor_model._continuous_actor_dynamics(-0.1,
                                                                           t_act_k_minus_one_at_T=np.zeros(4),
                                                                           u_act=np.inf * np.ones(4),
                                                                           )

            with self.assertRaisesWithLiteralMatch(ValueError, 'The difference time t must be within [0, T].'):
                t_act_k_at_t = self.actor_model._continuous_actor_dynamics(6.1,
                                                                           t_act_k_minus_one_at_T=np.zeros(4),
                                                                           u_act=np.inf * np.ones(4),
                                                                           )

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(0,
                                                                       t_act_k_minus_one_at_T=np.array([4, 0, 0, 8]),
                                                                       u_act=np.inf * np.ones(4),
                                                                       )
            self.assertAllEqual(np.array([4, 0, 0, 8]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(0,
                                                                       t_act_k_minus_one_at_T=np.array([4, 0, 0, 8]),
                                                                       u_act=3 + np.array([1, 0, 0, -2]),
                                                                       )
            self.assertAllEqual(np.array([4, 0, 0, 8]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(6,
                                                                       t_act_k_minus_one_at_T=np.zeros(4),
                                                                       u_act=np.inf * np.ones(4),
                                                                       )
            self.assertAllEqual(np.zeros(4), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                       t_act_k_minus_one_at_T=np.zeros(4),
                                                                       u_act=np.inf * np.ones(4),
                                                                       )
            self.assertAllEqual(np.zeros(4), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                       t_act_k_minus_one_at_T=np.zeros(4),
                                                                       u_act=3 * np.ones(4),
                                                                       )
            self.assertAllEqual(np.ones(4), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                       t_act_k_minus_one_at_T=np.ones(4),
                                                                       u_act=np.inf * np.ones(4),
                                                                       )
            self.assertAllEqual(2 * np.ones(4), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                       t_act_k_minus_one_at_T=np.array([4, 0, 0, 8]),
                                                                       u_act=3 + np.array([2, 1, 0, 4]),
                                                                       )
            self.assertAllEqual(np.array([5, 0, 1, 0]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(2,
                                                                       t_act_k_minus_one_at_T=np.array([4, 0, 0, 6]),
                                                                       u_act=3 + np.array([2, 1, 0, 2]),
                                                                       )
            self.assertAllEqual(np.array([6, 1, 2, 8]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(5,
                                                                       t_act_k_minus_one_at_T=np.array([4, 0, 0, 8]),
                                                                       u_act=3 + np.array([2, 1, 0, 4]),
                                                                       )
            self.assertAllEqual(np.array([0, 4, 5, 1]), t_act_k_at_t)

            # a complete series
            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(0,
                                                                       t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                       u_act=3 + np.array([0, 1, 0, 3]),
                                                                       )
            self.assertAllEqual(np.array([5, 0, 0, 6]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                       t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                       u_act=3 + np.array([0, 1, 0, 3]),
                                                                       )
            self.assertAllEqual(np.array([6, 0, 1, 7]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(2,
                                                                       t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                       u_act=3 + np.array([0, 1, 0, 3]),
                                                                       )
            self.assertAllEqual(np.array([7, 1, 2, 8]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(3,
                                                                       t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                       u_act=3 + np.array([0, 1, 0, 3]),
                                                                       )
            self.assertAllEqual(np.array([8, 2, 3, 0]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(4,
                                                                       t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                       u_act=3 + np.array([0, 1, 0, 3]),
                                                                       )
            self.assertAllEqual(np.array([0, 3, 4, 1]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(5,
                                                                       t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                       u_act=3 + np.array([0, 1, 0, 3]),
                                                                       )
            self.assertAllEqual(np.array([0, 4, 5, 2]), t_act_k_at_t)

            t_act_k_at_t = self.actor_model._continuous_actor_dynamics(6,
                                                                       t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                       u_act=3 + np.array([0, 1, 0, 3]),
                                                                       )
            self.assertAllEqual(np.array([0, 5, 6, 3]), t_act_k_at_t)

        with self.subTest(name='Test with T = t_cycle'):
            # a normal case
            t_act_k_at_t = self.actor_model_edge_case._continuous_actor_dynamics(6,
                                                                                 t_act_k_minus_one_at_T=np.array(
                                                                                     [4, 0, 0, 8]),
                                                                                 u_act=3 + np.array([2, 1, 0, 4]),
                                                                                 )
            self.assertAllEqual(np.array([0, 5, 6, 2]), t_act_k_at_t)

            # the edge case, activation and finish in same time step, so that the output is 0 instead of T
            t_act_k_at_t = self.actor_model_edge_case._continuous_actor_dynamics(9,
                                                                                 t_act_k_minus_one_at_T=np.array(
                                                                                     [4, 0, 0, 8]),
                                                                                 u_act=3 + np.array([2, 1, 0, 4]),
                                                                                 )
            self.assertAllEqual(np.array([0, 8, 0, 5]), t_act_k_at_t)

    def test_inverse_continuous_actor_dynamics(self):
        with self.assertRaisesWithLiteralMatch(
                ValueError,
                'The internal actor time t_act_k_at_t must be within (0, self.t_cycle). This is partly because for '
                't_act_k_at_t=0 the function is not bijective, so that there are ambiguous solutions.'):
            t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([1, 0, 1, 1]),
                                                                    t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                    u_act=3 + np.array([2, np.inf, 2, 0]))

        with self.assertRaisesWithLiteralMatch(
                ValueError,
                'The internal actor time t_act_k_at_t must be within (0, self.t_cycle). This is partly because for '
                't_act_k_at_t=0 the function is not bijective, so that there are ambiguous solutions.'):
            t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([1, 9.1, 1, 1]),
                                                                    t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                    u_act=3 + np.array([2, np.inf, 2, 0]))

        with self.assertRaisesWithLiteralMatch(
                ValueError,
                'The internal actor time at the end of the last time step t_act_k_minus_one_at_T t must be '
                'within [0, self.t_cycle).'):
            t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([1, 1, 1, 1]),
                                                                    t_act_k_minus_one_at_T=np.array([-0.1, 4, 8, 4]),
                                                                    u_act=3 + np.array([2, np.inf, 2, 0]))

        with self.assertRaisesWithLiteralMatch(
                ValueError,
                'The internal actor time at the end of the last time step t_act_k_minus_one_at_T t must be '
                'within [0, self.t_cycle).'):
            t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([1, 1, 1, 1]),
                                                                    t_act_k_minus_one_at_T=np.array([9, 4, 8, 4]),
                                                                    u_act=3 + np.array([2, np.inf, 2, 0]))

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([1, 1, 1, 1]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([3, np.inf, 3, np.inf]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([2, 2, 2, 2]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([4, np.inf, 4, np.inf]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([3, 3, 3, 3]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([5, np.inf, 5, np.inf]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([4, 4, 4, 4]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([np.inf, 0, np.inf, 0]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([5, 5, 5, 5]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([np.inf, 1, np.inf, 1]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([6, 6, 6, 6]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([np.inf, 2, np.inf, 2]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([7, 7, 7, 7]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([np.inf, 3, np.inf, 3]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([8, 8, 8, 8]),
                                                                t_act_k_minus_one_at_T=np.array([0, 4, 8, 4]),
                                                                u_act=3 + np.array([2, np.inf, 2, 0]))
        self.assertAllEqual(np.array([np.inf, 4, 0, 4]), t)

        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=np.array([8, 8, 8, 8]),
                                                                t_act_k_minus_one_at_T=np.array([3, 1, 0, 2]),
                                                                u_act=3 + np.array([np.inf, np.inf, 0, np.inf]))
        self.assertAllEqual(np.array([5, np.inf, np.inf, np.inf]), t)

    def test_if_inverse(self):
        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                   t_act_k_minus_one_at_T=np.zeros(4),
                                                                   u_act=3 * np.ones(4),
                                                                   )  # [1, 1, 1, 1]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.zeros(4),
                                                                u_act=3 * np.ones(4),
                                                                )
        self.assertAllEqual(np.ones(4), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                   t_act_k_minus_one_at_T=np.ones(4),
                                                                   u_act=np.inf * np.ones(4),
                                                                   )  # [2, 2, 2, 2]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.ones(4),
                                                                u_act=np.inf * np.ones(4),
                                                                )
        self.assertAllEqual(np.ones(4), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                   t_act_k_minus_one_at_T=np.array([4, 0]),
                                                                   u_act=3 + np.array([2, 0]),
                                                                   )  # [5, 1]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([4, 0]),
                                                                u_act=3 + np.array([2, 0]),
                                                                )
        self.assertAllEqual(np.ones(2), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(2,
                                                                   t_act_k_minus_one_at_T=np.array([4, 0, 0, 6]),
                                                                   u_act=3 + np.array([2, 1, 0, 2]),
                                                                   )  # [6, 1, 2, 8]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([4, 0, 0, 6]),
                                                                u_act=3 + np.array([2, 1, 0, 2]),
                                                                )
        self.assertAllEqual(2 * np.ones(4), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(5,
                                                                   t_act_k_minus_one_at_T=np.array([0, 0, 8]),
                                                                   u_act=3 + np.array([1, 0, 4]),
                                                                   )  # [4, 5, 1])
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([0, 0, 8]),
                                                                u_act=3 + np.array([1, 0, 4]),
                                                                )
        self.assertAllEqual(5 * np.ones(3), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(5,
                                                                   t_act_k_minus_one_at_T=np.array([3, 1, 0, 2]),
                                                                   u_act=3 + np.array([np.inf, np.inf, 0, np.inf]),
                                                                   )  # [8, 6, 5, 7])
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t=t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([3, 1, 0, 2]),
                                                                u_act=3 + np.array([np.inf, np.inf, 0, np.inf]))
        self.assertAllEqual(5 * np.ones(4), t)

        # a complete series
        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(0,
                                                                   t_act_k_minus_one_at_T=np.array([5, 6]),
                                                                   u_act=3 + np.array([0, 3]),
                                                                   )  # [5, 6]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([5, 6]),
                                                                u_act=3 + np.array([0, 3])
                                                                )
        self.assertAllEqual(np.zeros(2), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(1,
                                                                   t_act_k_minus_one_at_T=np.array([5, 0, 6]),
                                                                   u_act=3 + np.array([0, 0, 3]),
                                                                   )  # [6, 1, 7]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([5, 0, 6]),
                                                                u_act=3 + np.array([0, 0, 3]),
                                                                )
        self.assertAllEqual(1 * np.ones(3), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(2,
                                                                   t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                   u_act=3 + np.array([0, 1, 0, 3]),
                                                                   )  # [7, 1, 2, 8]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([5, 0, 0, 6]),
                                                                u_act=3 + np.array([0, 1, 0, 3]),
                                                                )
        self.assertAllEqual(2 * np.ones(4), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(3,
                                                                   t_act_k_minus_one_at_T=np.array([5, 0, 0]),
                                                                   u_act=3 + np.array([0, 1, 0]),
                                                                   )  # [8, 2, 3]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([5, 0, 0]),
                                                                u_act=3 + np.array([0, 1, 0]),
                                                                )
        self.assertAllEqual(3 * np.ones(3), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(4,
                                                                   t_act_k_minus_one_at_T=np.array([0, 0, 6]),
                                                                   u_act=3 + np.array([1, 0, 3]),
                                                                   )  # [3, 4, 1]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([0, 0, 6]),
                                                                u_act=3 + np.array([1, 0, 3]),
                                                                )
        self.assertAllEqual(4 * np.ones(3), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(5,
                                                                   t_act_k_minus_one_at_T=np.array([0, 0, 6]),
                                                                   u_act=3 + np.array([1, 0, 3]),
                                                                   )  # [4, 5, 2]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([0, 0, 6]),
                                                                u_act=3 + np.array([1, 0, 3]),
                                                                )
        self.assertAllEqual(5 * np.ones(3), t)

        t_act_k_at_t = self.actor_model._continuous_actor_dynamics(6,
                                                                   t_act_k_minus_one_at_T=np.array([0, 0, 6]),
                                                                   u_act=3 + np.array([1, 0, 3]),
                                                                   )  # [5, 6, 3]
        t = self.actor_model._inverse_continuous_actor_dynamics(t_act_k_at_t,
                                                                t_act_k_minus_one_at_T=np.array([0, 0, 6]),
                                                                u_act=3 + np.array([1, 0, 3]),
                                                                )
        self.assertAllEqual(np.inf * np.ones(3), t)  # not within the current time step / output not in [0, T)

    def test_output_t_up_interval(self):
        t_up_int_outp = self.actor_model.output_t_up_interval(t_act_k_minus_one_at_T=np.array([1, 3, 0, 0, 0, 4]),
                                                              u_act=np.array([np.inf, np.inf, 6, 7, 10, np.inf]),
                                                              )
        exp_t_up_int = np.array([[0, 1], [np.inf, np.inf], [4, 5], [5, 6], [np.inf, np.inf], [np.inf, np.inf]])
        self.assertAllEqual(exp_t_up_int, t_up_int_outp)

    def test_output_t_hit_interval(self):
        t_up_int_outp = self.actor_model.output_t_hit_interval(t_act_k_minus_one_at_T=np.array([1, 3, 0, 0, 0, 4]),
                                                               u_act=np.array([np.inf, np.inf, 6, 7, 10, np.inf]),
                                                               )
        exp_t_up_int = np.array([[1, 3], [0, 1], [5, 6], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf]])
        self.assertAllEqual(exp_t_up_int, t_up_int_outp)

    def test_output_t_down_interval(self):
        t_up_int_outp = self.actor_model.output_t_down_interval(t_act_k_minus_one_at_T=np.array([1, 3, 0, 0, 0, 4]),
                                                                u_act=np.array([np.inf, np.inf, 6, 7, 10, np.inf]),
                                                                )
        exp_t_up_int = np.array([[3, 4], [1, 2], [np.inf, np.inf], [np.inf, np.inf], [np.inf, np.inf], [0, 1]])
        self.assertAllEqual(exp_t_up_int, t_up_int_outp)


if __name__ == "__main__":
    tf.test.main()
