import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np

import tensorflow as tf

from simulator import ActorSimulator, AreaParticleSimulator
from heuristics import FirstActorFirstHeuristicController


class AbstractHeuristicControllerTest(tf.test.TestCase):
    """Test cases for the AbstractAbstractHeuristicController class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        actor_dict = {'t_activate': 1,
                      't_up': 1,
                      't_hit': 2,
                      't_down': 1,
                      't_reset': 4}
        actor_positions = np.array([[45., 13.5], [45., 29.5], [65., 7.5], [65., 23.5]])
        length = 4
        width = 9
        T = 1
        actors = ActorSimulator(pos=actor_positions,
                                length=length,
                                width=width,
                                T=T,
                                **actor_dict)
        area_width = 40
        v0_x = 1.0
        particle_simulator_dict = {'total_num_particles': 20,
                                   'process_parameters': 2 * ({'type': 'CV',
                                                               'pos_0': (-2.5, area_width / 2),
                                                               'pos0_stddev': (0.0, 3 * area_width),
                                                               'v_0': (v0_x, 0.0),
                                                               'v0_stddev': (v0_x / 10.0, v0_x / 10.0),
                                                               'S_w': (0.0, 0.0),
                                                               },),
                                   'birth_rate_mean_and_stddev': ((0.5, 0.0), (0.5, 0.0)),
                                   'class_labels': (0, 1),
                                   'area_width': area_width,
                                   'particle_r': 3,
                                   }
        particle_simulator = AreaParticleSimulator(T=T,
                                                   **particle_simulator_dict,
                                                   seed=12345,  # fix the seed for reproducibility
                                                   )

        cls.heuristic = FirstActorFirstHeuristicController(
            particle_simulator,
            actors=actors,
            T=T,
            actor_model_dict=actor_dict,
        )

    def test_detect_implicit_associations(self):
        actors_next_time_ready = np.array([0, 2, 5, 8])
        delta_toa = np.array([[2, 5, 10, 10],
                              [7, 10, 5, 5],
                              [9, 13, 11, 2],
                              [12, 6, 6, 6]])
        decision_mask = np.array([[True, False, True, True],
                                  [False, True, False, False],
                                  [True, False, False, False],
                                  [False, True, False, True]])
        # --> particles blocked:
        #   array([[ True, False, False, False],
        #          [False, False, False, False],
        #          [False, False, False,  True],
        #          [False,  True,  True,  True]])
        particle_class = np.array([1, 1, 1, 0])

        implicit_decisions_particles_mask = self.heuristic._detect_implicit_associations(actors_next_time_ready,
                                                                                         decision_mask,
                                                                                         delta_toa,
                                                                                         particle_class)

        exp_implicit_decisions_particles_mask = np.array([True, True, False, False])
        self.assertAllEqual(exp_implicit_decisions_particles_mask, implicit_decisions_particles_mask)

        actors_next_time_ready_outp = self.heuristic._assign_implicit_decisions(
            actors_next_time_ready,
            decision_mask,
            delta_toa,
            implicit_decisions_particles_mask=implicit_decisions_particles_mask)
        exp_actors_next_time_ready = np.array([0, 16, 15, 8])
        self.assertAllEqual(exp_actors_next_time_ready, actors_next_time_ready_outp)

    def test_assign_implicit_decisions(self):
        actors_next_time_ready = np.array([0, 2, 5, 8])
        delta_toa = np.array([[3, 5, 1],
                              [7, 9, 5],
                              [4, 6, 2],
                              [8, 10, 6]])
        decision_mask = np.array([[True, False, True],
                                  [False, False, True],
                                  [False, True, False],
                                  [False, False, False]])

        particle_mask = np.sum(decision_mask, axis=0) == 1  # [True, True, False]
        actors_next_time_ready_outp = self.heuristic._assign_implicit_decisions(
            actors_next_time_ready,
            decision_mask,
            delta_toa,
            implicit_decisions_particles_mask=particle_mask)

        exp_actors_next_time_ready = np.array([9, 2, 12, 8])
        self.assertAllEqual(exp_actors_next_time_ready, actors_next_time_ready_outp)

        actors_next_time_ready = np.array([0, 2, 5, 8])
        delta_toa = np.array([[3, 5],
                              [7, 9],
                              [4, 6],
                              [8, 10]])
        decision_mask = np.array([[False, True],
                                  [False, False],
                                  [True, False],
                                  [False, False]])
        particle_mask = np.sum(decision_mask, axis=0) == 1  # [True, True]

        actors_next_time_ready_outp = self.heuristic._assign_implicit_decisions(
            actors_next_time_ready,
            decision_mask,
            delta_toa,
            implicit_decisions_particles_mask=particle_mask)

        exp_actors_next_time_ready = np.array([11, 2, 10, 8])
        self.assertAllEqual(exp_actors_next_time_ready, actors_next_time_ready_outp)

    def test_detect_if_particle_actor_assignment_is_not_blocked_by_prescribed_activations(self):
        delta_toa = np.array([[0, 1, 2, 3, 12],
                              [3, 4, 5, 6, 15]])
        prescribed_activation_times = np.array([[0], [np.inf]])
        r = self.heuristic._detect_if_particle_actor_assignment_is_not_blocked_by_prescribed_activations(
            delta_toa, prescribed_activation_times
        )
        exp_mask = np.array([[False, False, False, False, True],
                             [True, True, True, True, True]])
        self.assertAllEqual(exp_mask, r)

        delta_toa = np.array([[0, 1, 2, 11, 12, 13, 14],
                              [2, 3, 4, 20, 21, 22, 23]])
        prescribed_activation_times = np.array([[1, np.inf, np.inf],
                                                [np.inf, 9, 27]])
        r = self.heuristic._detect_if_particle_actor_assignment_is_not_blocked_by_prescribed_activations(
            delta_toa, prescribed_activation_times
        )
        exp_mask = np.array([[False, False, False, False, False, True, True],
                             # cycle is finished at time 10 --> the earliest arrival time is 13
                             [True, True, False, False, True, False, False],
                             # cycle must be finished at time 9, started at time 0 --> the latest arrival time is 3
                             # cycle is finished at time 18 --> the earliest arrival time is 21
                             # cycle must be finished at time 27, started at time 18 --> the latest arrival time is 21
                             ])
        self.assertAllEqual(exp_mask, r)


if __name__ == "__main__":
    tf.test.main()
