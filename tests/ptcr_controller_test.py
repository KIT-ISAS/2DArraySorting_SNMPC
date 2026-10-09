import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from absl import logging


import tensorflow as tf
import numpy as np

from unittest.mock import patch

from ptcr_controller import PTCRController, NodeData
from adf_mc_validation.adf_mc_controller_hooks import AdfMcComparisonHooks
from heuristics import FirstActorFirstHeuristicController, AbstractHeuristicController2D
from simulator import ActorSimulator, AreaParticleSimulator, AreaSortingSimulator


class PTCRControllerTest(tf.test.TestCase):
    """Test cases for the PTCRController class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.T = 1
        actor_dict = {"t_activate": 1,
                      "t_up": 1,
                      "t_hit": 2,
                      "t_down": 1,
                      "t_reset": 4}

        # 2D - small grid with 2 actors
        actor_positions = np.array([[10, 20], [13, 20]])
        length = 2
        width = 9

        actors_2d_small = ActorSimulator(pos=actor_positions,
                                         length=length,
                                         width=width,
                                         T=cls.T,
                                         **actor_dict)

        area_width = 40
        v0_x = 1.0

        particle_simulator_dict = {
            "total_num_particles": 20,
            "process_parameters": 2
                                  * (
                                      {
                                          "type": "CV",
                                          "pos_0": (-2.5, area_width / 2),
                                          "pos0_stddev": (0.0, 3 * area_width),
                                          "v_0": (v0_x, 0.0),
                                          "v0_stddev": (v0_x / 10.0, v0_x / 10.0),
                                          "S_w": (0.0001, 0.0001),
                                      },
                                  ),
            "birth_rate_mean_and_stddev": ((0.5, 0.0), (0.5, 0.0)),
            "class_labels": (0, 1),
            "area_width": area_width,
            "particle_r": 3,
        }

        particle_model_dict = {
            "p_detect": 0.95,
            "S_w": (0.0001, 0.0001),
            "S_v": (0.01, 0.01),
        }

        contact_model_dict = {
            "p_eject": [0.0, 0.9, 0.2, 0.1],
            "use_sum_approximation": False,
        }

        particle_simulator_2d = AreaParticleSimulator(
            T=cls.T,
            **particle_simulator_dict,
            seed=1,
        )

        # Time between two control steps
        cls.controller_rate = 5

        cls.small_grid_controller_kwargs = dict(
            mtt_tracker=particle_simulator_2d,
            actors=actors_2d_small,
            T=cls.controller_rate * cls.T,
            N=3,
            particle_model_dict=particle_model_dict,
            actor_model_dict=actor_dict,
            contact_model_dict=contact_model_dict,
            particle_ordering="id",
            N_R=2,
            seed=987654321,
        )

        # 2D - full grid
        v0_x = 1.0  # initial velocity in x-direction
        avg_num_time_steps_particle_in_array = lambda array_end, T: int(np.ceil(array_end / (v0_x * T)))

        actor_positions = np.array(  # in mm, positions correspond to the positions of the ISAS-TableSort with the
            # origin of the coordinate system fixed in the lower left corner of the (physical) actuator array assembly
            [[25., 17.5], [25., 33.5], [25., 49.5], [25., 65.5], [25., 81.5], [25., 97.5], [25., 113.5],
             [45., 11.5], [45., 27.5], [45., 43.5], [45., 59.5], [45., 75.5], [45., 91.5], [45., 107.5],
             [65., 21.5], [65., 37.5], [65., 53.5], [65., 69.5], [65., 85.5], [65., 101.5], [65., 117.5],
             [85., 15.5], [85., 31.5], [85., 47.5], [85., 63.5], [85., 79.5], [85., 95.5], [85., 111.5],
             [105., 9.5], [105., 25.5], [105., 41.5], [105., 57.5], [105., 73.5], [105., 89.5], [105., 105.5],
             [125., 19.5], [125., 35.5], [125., 51.5], [125., 67.5], [125., 83.5], [125., 99.5], [125., 115.5],
             [145., 13.5], [145., 29.5], [145., 45.5], [145., 61.5], [145., 77.5], [145., 93.5], [145., 109.5],
             [165., 7.5], [165., 23.5], [165., 39.5], [165., 55.5], [165., 71.5], [165., 87.5], [165., 103.5]])
        offset = 20  # we need to shift the array to the right for the simulation
        actor_positions[:, 0] += offset
        array_end = 180 + offset
        area_width = 125
        avg_particles_in_array = 24  # average number of particles (visible & invisible, i.e., ejected) that should be
        # in the sorter, defines the occupation density/mass flow of the sorter
        total_birth_rate_mean = avg_particles_in_array / avg_num_time_steps_particle_in_array(array_end, cls.T)
        length = 4
        width = 9

        # Define the actors for the simulator
        cls.actors_2d = ActorSimulator(pos=actor_positions,
                                       length=length,
                                       width=width,
                                       T=cls.T,
                                       **actor_dict)

        particle_simulator_dict_2d = {'total_num_particles': 1000,
                                      'process_parameters': 2 * ({'type': 'CV',
                                                                  'pos_0': (-2.5, area_width / 2),
                                                                  'pos0_stddev': (0.0, 3 * area_width),
                                                                  'v_0': (v0_x, 0.0),
                                                                  'v0_stddev': (v0_x / 10.0, v0_x / 10.0),
                                                                  'S_w': (0.0001, 0.0001),
                                                                  },),
                                      'birth_rate_mean_and_stddev': ((total_birth_rate_mean / 2, 0.3),
                                                                     (total_birth_rate_mean / 2, 0.3)),
                                      'class_labels': (0, 1),
                                      'area_width': area_width,
                                      'particle_r': 3,
                                      }

        particle_model_dict_2d = {
            'p_detect': 0.95,
            'S_w': (0.0001, 0.0001),
            'S_v': (0.01, 0.01),
        }

        particle_simulator_2d = AreaParticleSimulator(T=cls.T,
                                                      **particle_simulator_dict_2d,
                                                      seed=12345,  # fix the seed for reproducibility
                                                      )

        cls.two_d_controller_kwargs = dict(
            mtt_tracker=particle_simulator_2d,
            particle_model_dict=particle_model_dict_2d,
            actor_model_dict=actor_dict,
            contact_model_dict=contact_model_dict,
            particle_ordering="id",
            seed=987654321,
            debug=True
        )

    def test_control_from_states(self):
        # Define particle states: always contains position and speed of particle
        initial_motion_state_mean = np.array([
            # particle 0 arrives in 4s at actor 1 since actor 1 is at position 10 and particle at 6 with speed of
            # 1 and arrives in 7s at actor 2 since actor 2 is at position 13
            [6, 1.0, 20.0, 0.0],
            # particle 1 arrives in 2s at actor 1 and in 5s at actor 2
            [8, 1.0, 20.0, 0.0],
            # particle 2 has already passed actor 1 and arrives in 1s at actor 2.
            [14, 1.0, 20.0, 0.0],
            # particle 3 has already passed actor 1 and arrives in 3s at actor 2
            [11, 1.0, 20.0, 0.0],
        ])

        initial_motion_state_cov = np.stack([0.001 * np.diag([1.0, 1.0, 1.0, 1.0])] * 4)
        particle_states = [initial_motion_state_mean, initial_motion_state_cov, np.ones(4)]

        # Define the particle ids: shoot out all particles here
        particle_classes = np.array([1, 1, 1, 1])

        # Define current actor times
        t_act = np.array([0.0, 0.0])

        # between activating and hitting the actor, we have 4 time steps and between hitting and reactivating the actor
        # we have 5 time steps
        # for actor 1 we only have one option where we can eject one particle (where there is enough time): particle 0
        # for actor 2, we only have two particles that could be ejected (where there is enough time): particle 0 and 1
        # this means, it is optimal to eject particle 0 at actor 1 and particle 1 at actor 2 because this gives us most
        # ejections, thus lowest costs. Therefore,we activate actor 1 in 1s and actor 2 in 2s
        expected_activations = np.array([[1], [2]])

        with self.subTest(name='Test with olf, internal_T = t_cycle, terminal costs'):
            controller = PTCRController(**self.small_grid_controller_kwargs,
                                        use_olf=True,
                                        use_only_terminal_costs=True,
                                        )
            calculated_activations = controller.control_from_states(
                particle_states=particle_states,
                particle_class=particle_classes,
                particle_id=np.array([0, 1, 2, 3]),
                t_act=t_act,
            )

            self.assertAllEqual(expected_activations, calculated_activations)

        with self.subTest(name='Test with clf, internal_T =  t_cycle, terminal costs'):
            controller = PTCRController(**self.small_grid_controller_kwargs,
                                        use_olf=True,
                                        adf={"use_numba": False},
                                        use_only_terminal_costs=True,
                                        )
            calculated_activations = controller.control_from_states(
                particle_states=particle_states,
                particle_class=particle_classes,
                particle_id=np.array([0, 1, 2, 3]),
                t_act=t_act,
            )

            self.assertAllEqual(expected_activations, calculated_activations)

        with self.subTest(name='Test with olf, internal_T != t_cycle, terminal costs'):
            controller = PTCRController(**self.small_grid_controller_kwargs,
                                        use_olf=True,
                                        internal_T=self.controller_rate * self.T,
                                        use_only_terminal_costs=True,
                                        )
            calculated_activations = controller.control_from_states(
                particle_states=particle_states,
                particle_class=particle_classes,
                particle_id=np.array([0, 1, 2, 3]),
                t_act=t_act,
            )

            self.assertAllEqual(expected_activations, calculated_activations)

        with self.subTest(name='Test with olf, internal_T = t_cycle, step costs'):
            controller = PTCRController(**self.small_grid_controller_kwargs,
                                        use_olf=True,
                                        use_only_terminal_costs=False,
                                        )
            calculated_activations = controller.control_from_states(
                particle_states=particle_states,
                particle_class=particle_classes,
                particle_id=np.array([0, 1, 2, 3]),
                t_act=t_act,
            )

            self.assertAllEqual(expected_activations, calculated_activations)

    def test_search_for_possible_activation_sequences(self):
        # tests where N_R = num_particles = 2, therefore no rollout is performed
        with self.subTest(name='Test with only one reachable actor and internal_T = t_cycle'):
            particle_class = np.array([1, 1])  # All particles are to be ejected
            toas = np.array([
                [4, 20],  # actor 0 can eject p_0 if activated at time 1 and p_1 if activated at time 17
                [-np.inf, -np.inf]  # actor 1 cannot eject any particles
            ])
            t_act = np.array([0, 0])

            exp_control_sequences = np.array([
                [[1, 17, np.inf, np.inf, np.inf],  # actor 0 ejects p_0 at time 1 and p_1 at time 17
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[1, np.inf, np.inf, np.inf, np.inf],  # actor 0 ejects p_0 at time 1
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[np.inf, 17, np.inf, np.inf, np.inf],  # actor 0 ejects p_1 at time 17
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf],  # actor 0 does not eject any particles
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
            ])

            controller = PTCRController(N=5,
                                        **{k: v for k, v in self.small_grid_controller_kwargs.items() if k != 'N'})
            outp_control_sequences = controller._search_for_possible_activation_sequences(
                particle_class=particle_class,
                delta_toa=toas,
                t_act=t_act
            )

            self.assertAllEqual(exp_control_sequences, outp_control_sequences)

        with self.subTest(name='Test legacy tree (tree_search.use_fast=False) with only one reachable actor'):
            particle_class = np.array([1, 1])
            toas = np.array([
                [4, 20],
                [-np.inf, -np.inf]
            ])
            t_act = np.array([0, 0])
            exp_control_sequences = np.array([
                [[1, 17, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf]],
                [[1, np.inf, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf]],
                [[np.inf, 17, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf]],
                [[np.inf, np.inf, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]
            ])
            controller = PTCRController(
                N=5,
                tree_search={"use_fast": False},
                **{k: v for k, v in self.small_grid_controller_kwargs.items() if k != 'N'},
            )
            self.assertFalse(controller._use_fast_tree)
            outp_control_sequences = controller._search_for_possible_activation_sequences(
                particle_class=particle_class,
                delta_toa=toas,
                t_act=t_act,
            )
            self.assertAllEqual(exp_control_sequences, outp_control_sequences)

        with self.subTest(name='Test with only one reachable actor and internal_T != t_cycle'):
            controller = PTCRController(**{k: v for k, v in self.small_grid_controller_kwargs.items() if k != 'N'},
                                        N=5,
                                        internal_T=self.controller_rate * self.T,
                                        use_only_terminal_costs=True,
                                        )

            exp_control_sequences = np.array([
                [[1, np.inf, np.inf, 17, np.inf],  # actor 0 ejects p_0 at time 1 and p_1 at time 17
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[1, np.inf, np.inf, np.inf, np.inf],  # actor 0 ejects p_0 at time 1
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[np.inf, np.inf, np.inf, 17, np.inf],  # actor 0 ejects p_1 at time 17
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf],  # actor 0 does not eject any particles
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
            ])

            outp_control_sequences = controller._search_for_possible_activation_sequences(
                particle_class=particle_class,
                delta_toa=toas,
                t_act=t_act
            )

            self.assertAllEqual(exp_control_sequences, outp_control_sequences)

        with self.subTest(name='Test with only one reachable actor, t_act not zero and internal_T = t_cycle'):
            particle_class = np.array([1, 1])  # All particles are to be ejected
            toas = np.array([
                [4, 20],  # actor 0 can eject p_0 if activated at time 1 and p_1 if activated at time 17
                [-np.inf, -np.inf]  # actor 1 cannot eject any particles
            ])
            t_act = np.array([2, 1])

            exp_control_sequences = np.array([
                [[np.inf, 17, np.inf, np.inf, np.inf],  # actor 0 ejects p_1 at time 17
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf],  # actor 0 does not eject any particles
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
            ])

            controller = PTCRController(N=5,
                                        use_only_terminal_costs=True,
                                        **{k: v for k, v in
                                           self.small_grid_controller_kwargs.items() if k != 'N'}
                                        )
            outp_control_sequences = controller._search_for_possible_activation_sequences(
                particle_class=particle_class,
                delta_toa=toas,
                t_act=t_act
            )

            self.assertAllEqual(exp_control_sequences, outp_control_sequences)

        with self.subTest(name='Test with only one reachable actor, t_act not zero and internal_T != t_cycle'):
            controller = PTCRController(N=5,
                                        internal_T=self.controller_rate * self.T,
                                        use_only_terminal_costs=True,
                                        **{k: v for k, v in
                                           self.small_grid_controller_kwargs.items() if k != 'N'}
                                        )

            exp_control_sequences = np.array([
                [[np.inf, np.inf, np.inf, 17, np.inf],  # actor 0 ejects p_1 at time 17
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf],  # actor 0 does not eject any particles
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]  # actor 1 does not eject any particles
            ])

            outp_control_sequences = controller._search_for_possible_activation_sequences(
                particle_class=particle_class,
                delta_toa=toas,
                t_act=t_act
            )

            self.assertAllEqual(exp_control_sequences, outp_control_sequences)

        with self.subTest(name='Test with two reachable actors, t_act not zero and internal_T = t_cycle'):
            particle_class = np.array([1, 1])  # All particles are to be ejected
            toas = np.array([
                [4.0, 20],  # actor 0 can eject p_0 if activated at time 1 and p_1 if activated at time 17
                [7.0, 23]  # actor 1 can eject p_0 if activated at time 4 and p_1 if activated at time 20
            ])
            t_act = np.array([2, 1])

            exp_control_sequences = np.array([
                # particle one cant be ejected at any because of t_act > 0 and t_cycle = 9
                [[np.inf, 17, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, 20, np.inf, np.inf]]
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf]]
            ])

            controller = PTCRController(N=5,
                                        **{k: v for k, v in
                                           self.small_grid_controller_kwargs.items() if k != 'N'})
            outp_control_sequences = controller._search_for_possible_activation_sequences(
                particle_class=particle_class,
                delta_toa=toas,
                t_act=t_act
            )

            self.assertAllEqual(exp_control_sequences, outp_control_sequences)

        with self.subTest(name='Test with two reachable actors, t_act not zero and internal_T != t_cycle'):
            controller = PTCRController(N=6,
                                        internal_T=self.controller_rate * self.T,
                                        use_only_terminal_costs=True,
                                        **{k: v for k, v in
                                           self.small_grid_controller_kwargs.items() if k != 'N'}
                                        )

            exp_control_sequences = np.array([
                # particle one cant be ejected at any because of t_act > 0 and t_cycle = 9
                [[np.inf, np.inf, np.inf, 17, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]]
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, 20, np.inf]]
                ,
                [[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                 [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]]
            ])

            outp_control_sequences = controller._search_for_possible_activation_sequences(
                particle_class=particle_class,
                delta_toa=toas,
                t_act=t_act
            )

            self.assertAllEqual(exp_control_sequences, outp_control_sequences)

    def test_partially_check_ejection_condition_be_an_undisturbed_particle(self):
        delta_toa = np.array([[-1.0, 7.0, -np.inf, -np.inf, 18.0],
                              [2.5, 10.0, -np.inf, 3.5, -np.inf],
                              [4.0, -np.inf, -np.inf, 6.5, 20.0],
                              [3.0, 8.0, -np.inf, -np.inf, 1.5]])
        t_act = np.array([0.0, 1.0, 7.0, 1.5])
        # actor 0 and 2 do not disturb any particles, actor 1 has an influence from time 0 to 4 it therefore will
        # affect particle 0 and particle 3. actor 3 has an influence from time 0 to 3.5 and therefore will affect
        # particle 0 and particle 4.

        exp_undisturbed_mask = np.array([[True, True, True, True, True],
                                         [False, True, True, False, True],
                                         [True, True, True, True, True],
                                         [False, True, True, True, False]])

        controller = PTCRController(actors=self.actors_2d,
                                    T=self.controller_rate * self.T,
                                    N=5,
                                    N_R=2,
                                    adf={"use_numba": False},
                                    **self.two_d_controller_kwargs)
        outp_undisturbed_mask = controller._partially_check_ejection_condition_be_an_undisturbed_particle(
            delta_toa, t_act)

        self.assertAllEqual(exp_undisturbed_mask, outp_undisturbed_mask)

    def test_ejection_condition_be_free(self):
        controller = PTCRController(actors=self.actors_2d,
                                    T=self.controller_rate * self.T,
                                    N=5,
                                    N_R=2,
                                    **self.two_d_controller_kwargs)

        already_assigned_activation_times_for_actor = np.array([-1.5])

        # an example where the new activation falls in the old activation window
        outp_is_free = controller._check_ejection_condition_be_free(
            activation_time=3.5,
            already_assigned_activation_times_for_actor=already_assigned_activation_times_for_actor)
        self.assertFalse(outp_is_free)

        # an example where the new activation is after the old activation window
        outp_is_free = controller._check_ejection_condition_be_free(
            activation_time=11.5,
            already_assigned_activation_times_for_actor=already_assigned_activation_times_for_actor)
        self.assertTrue(outp_is_free)

        # a sequence of examples
        already_assigned_activation_times_for_actor = np.array([-1.5, 3.5, 22.5])
        outp_is_free = controller._check_ejection_condition_be_free(
            activation_time=12.0,
            already_assigned_activation_times_for_actor=already_assigned_activation_times_for_actor)
        self.assertFalse(outp_is_free)
        outp_is_free = controller._check_ejection_condition_be_free(
            activation_time=12.5,
            already_assigned_activation_times_for_actor=already_assigned_activation_times_for_actor)
        self.assertTrue(outp_is_free)
        outp_is_free = controller._check_ejection_condition_be_free(
            activation_time=13.5,
            already_assigned_activation_times_for_actor=already_assigned_activation_times_for_actor)
        self.assertTrue(outp_is_free)
        outp_is_free = controller._check_ejection_condition_be_free(
            activation_time=14.0,
            already_assigned_activation_times_for_actor=already_assigned_activation_times_for_actor)
        self.assertFalse(outp_is_free)

    def test_control_sequence_from_node(self):
        controller = PTCRController(actors=self.actors_2d,
                                    T=self.controller_rate * self.T,
                                    N=5,
                                    N_R=2,
                                    **self.two_d_controller_kwargs)

        delta_toa = np.array([[-1.0, 7.0, -np.inf, -np.inf, 60.0],
                              [2.5, 10.0, -np.inf, 3.5, -np.inf],
                              [4.0, -np.inf, -np.inf, 6.5, 20.0],
                              [3.0, 8.0, -np.inf, 47.5, -np.inf]])

        with self.subTest(name='Test with all particles assigned'):
            assigned_actor_index = [2, 0, -1, 1, 2]
            node = NodeData(5, assigned_actor_index=assigned_actor_index)

            exp_control_sequence = np.array([[4.0, np.inf, np.inf, np.inf, np.inf],
                                             [0.5, np.inf, np.inf, np.inf, np.inf],
                                             [1.0, 17.0, np.inf, np.inf, np.inf],
                                             [np.inf, np.inf, np.inf, np.inf, np.inf]])

            outp_control_sequence = controller._control_sequence_from_node(node, delta_toa)
            self.assertAllEqual(exp_control_sequence, outp_control_sequence)

        with self.subTest(name='Test with only the first particles assigned'):
            assigned_actor_index = [2, 0, -1, 3]
            node = NodeData(4, assigned_actor_index=assigned_actor_index)

            exp_control_sequence = np.array([[4.0, np.inf, np.inf, np.inf, np.inf],
                                             [np.inf, np.inf, np.inf, np.inf, np.inf],
                                             [1.0, np.inf, np.inf, np.inf, np.inf],
                                             [np.inf, np.inf, np.inf, np.inf, 44.5]])

            outp_control_sequence = controller._control_sequence_from_node(node, delta_toa)
            self.assertAllEqual(exp_control_sequence, outp_control_sequence)

        with self.subTest(name='Test with two short control horizon'):
            delta_toa = np.array([[-1.0, 7.0, -np.inf, -np.inf, 18.0],
                                  [2.5, 10.0, -np.inf, 3.5, -np.inf],
                                  [4.0, -np.inf, -np.inf, 6.5, 20.0],
                                  [3.0, 8.0, -np.inf, 48.5, -np.inf]])
            assigned_actor_index = [2, 0, -1, 3]
            node = NodeData(4, assigned_actor_index=assigned_actor_index)

            exp_control_sequence = np.array([[4.0, np.inf, np.inf, np.inf, np.inf],
                                             [np.inf, np.inf, np.inf, np.inf, np.inf],
                                             [1.0, np.inf, np.inf, np.inf, np.inf],
                                             [np.inf, np.inf, np.inf, np.inf, np.inf]])
            exp_msg = ('Activation time step [5] exceeds control horizon of length 5. Truncating to fit. '
                       'Consider increasing N to avoid truncation.')

            with self.assertLogs(logger=logging.get_absl_logger(), level='WARNING') as cm:
                outp_control_sequence = controller._control_sequence_from_node(node, delta_toa)

            self.assertAllEqual(exp_control_sequence, outp_control_sequence)
            self.assertEqual('WARNING:absl:' + exp_msg, cm.output[0])

        with self.subTest(name='Test with two short control horizon and two particles'):
            delta_toa = np.array([[-1.0, 7.0, -np.inf, -np.inf, 60.0],
                                  [2.5, 10.0, -np.inf, 3.5, -np.inf],
                                  [4.0, -np.inf, -np.inf, 6.5, 20.0],
                                  [3.0, 8.0, -np.inf, 48.5, -np.inf]])
            assigned_actor_index = [2, 0, -1, 3, 0]
            node = NodeData(5, assigned_actor_index=assigned_actor_index)

            exp_control_sequence = np.array([[4.0, np.inf, np.inf, np.inf, np.inf],
                                             [np.inf, np.inf, np.inf, np.inf, np.inf],
                                             [1.0, np.inf, np.inf, np.inf, np.inf],
                                             [np.inf, np.inf, np.inf, np.inf, np.inf]])
            exp_msg = ('Activation time step [5 6] exceeds control horizon of length 5. Truncating to fit. '
                       'Consider increasing N to avoid truncation.')

            with self.assertLogs(logger=logging.get_absl_logger(), level='WARNING') as cm:
                outp_control_sequence = controller._control_sequence_from_node(node, delta_toa)

            self.assertAllEqual(exp_control_sequence, outp_control_sequence)
            self.assertEqual('WARNING:absl:' + exp_msg, cm.output[0])

    def test_perform_rollout_for_unassigned_particles(self):
        controller = PTCRController(**self.small_grid_controller_kwargs)

        with self.subTest(name='Test with one pre-assigned particle'):
            control_sequence = np.full((len(controller.actor_model.pos), controller.N), np.inf)
            control_sequence[0, 0] = 0.0  # activate actor 1 at time 0 and actor 2 never

            toa_third_p = 7.0
            first_toa = np.array([controller.actor_model.t_lead, 6.0, toa_third_p, 15.0])
            toas = np.stack((first_toa, first_toa + 3), axis=0)  # second actor is 3s behind first actor
            particle_class = np.array([1, 0, 1, 1])
            t_act = np.array([0, 0])
            control_sequence = controller._perform_rollout_for_unassigned_particles(
                control_sequence=control_sequence,
                assigned_particle_idxs=np.array([0, 2]),
                delta_toa=toas,
                particle_class=particle_class,
                t_act=t_act
            )
            assigned_actor_p3 = 0
            activation_time_p3 = toas[assigned_actor_p3, 3] - controller.actor_model.t_lead
            expected_control_sequence = control_sequence.copy()
            expected_control_sequence[assigned_actor_p3, 1] = activation_time_p3
            self.assertAllEqual(expected_control_sequence, control_sequence)

        with self.subTest(name='Test with two pre-assigned particle'):
            control_sequence = np.full((len(controller.actor_model.pos), controller.N), np.inf)
            control_sequence[:, 0] = np.array([3.0, 0.0])

            first_toa = np.array([0., 6., 7., 15.])  # toas for first actor
            toas = np.stack((first_toa, first_toa + 3), axis=0)  # second actor is 3s behind first actor
            particle_class = np.array([1, 1, 1, 1])
            t_act = np.array([0, 0])
            control_sequence = controller._perform_rollout_for_unassigned_particles(
                control_sequence=control_sequence,
                assigned_particle_idxs=np.array([0, 1]),
                delta_toa=toas,
                particle_class=particle_class,
                t_act=t_act
            )
            activation_time_p3 = toas[0, 3] - controller.actor_model.t_lead
            expected_control_sequence = control_sequence.copy()
            expected_control_sequence[0, 1] = activation_time_p3  # actor 0 can eject particle 4
            self.assertAllEqual(expected_control_sequence, control_sequence)

        with self.subTest(name='Test with zero pre-assigned particle'):
            control_sequence = np.full((len(controller.actor_model.pos), controller.N), np.inf)

            first_toa = np.array([1., 2., 3., 4.])
            toas = np.stack((first_toa, first_toa + 3), axis=0)  # second actor is 3s behind first actor
            particle_class = np.array([1, 1, 1, 1])
            t_act = np.array([0, 0])
            control_sequence = controller._perform_rollout_for_unassigned_particles(
                control_sequence=control_sequence,
                assigned_particle_idxs=np.array([]).astype(int),
                delta_toa=toas,
                particle_class=particle_class,
                t_act=t_act
            )

            expected_control_sequence = control_sequence.copy()
            expected_control_sequence[:, 0] = np.array([0, 1])
            self.assertAllEqual(expected_control_sequence, control_sequence)

    def test_merge_control_sequences(self):
        controller = PTCRController(**self.small_grid_controller_kwargs)

        control_sequence = np.array([[1.0, 2.0, np.inf], [np.inf, 3.5, 10.0]])
        faf_control_sequence = np.array([[np.inf, np.inf, np.inf], [60.1, np.inf, np.inf]])
        exp_control_sequence = np.array([[1.0, 2.0, np.inf], [60.1, 3.5, 10.0]])

        outp_control_sequence = controller._merge_control_sequences(faf_control_sequence, control_sequence)
        self.assertAllEqual(exp_control_sequence, outp_control_sequence)

        faf_control_sequence = np.array([[np.inf, np.inf, np.inf], [60.1, np.inf, 10.1]])
        with self.assertRaisesRegex(
                RuntimeError,
                r'The rollout control sequence has finite values where the original control sequence also '
                r'has finite values\. This indicates a severe error in the implementation of the rollout\.'):
            controller._merge_control_sequences(faf_control_sequence, control_sequence)

    def test_adf_mc_hooks_noop_when_disabled(self):
        controller = PTCRController(**self.small_grid_controller_kwargs, use_olf=True)
        hooks = controller._adf_mc_hooks
        self.assertIsInstance(hooks, AdfMcComparisonHooks)
        self.assertFalse(hooks.validate)
        self.assertFalse(hooks.cost_eval)
        self.assertIsNone(hooks.finalize_cost_eval())
        hooks.maybe_cost_eval(
            control_sequences_delta_t=np.ones((1, 2, 3)),
            costs_adf=None,
            estimated_motion_state_mean=np.zeros((1, 2)),
            estimated_motion_state_cov=np.eye(2)[None, ...],
            init_existence_probs=np.ones(1),
            particle_class=np.array([1]),
            t_act=np.zeros(2),
        )
        hooks.maybe_validate(
            u_star=np.ones((1, 2, 3)),
            estimated_motion_state_mean=np.zeros((1, 2)),
            estimated_motion_state_cov=np.eye(2)[None, ...],
            particle_class=np.array([1]),
            particle_id=np.array([0]),
            t_act=np.zeros(2),
        )

    def test_adf_cost_with_use_olf_false_raises(self):
        with self.assertRaises(NotImplementedError):
            PTCRController(
                **self.small_grid_controller_kwargs,
                use_olf=False,
                adf={},
            )


class ControllerDummy(AbstractHeuristicController2D):
    """A dummy controller that returns the activation time for the actors given a mapping from particles to actors.

    Can be used to mimic the behavior of the AbstractController's call function so that the mapping can be used in the
    self.simulator_3x3.
    """

    def __init__(self,
                 mtt_tracker,
                 actors,
                 T,
                 actor_model_dict,
                 actor_assignment):
        """Initializes the controller dummy.

        :param actor_assignment: A list of lists of tuples, where each tuple contains a (particle_id, actor_index) pair
            defining the assignment of particles to actors. The inner list is of length equal to the number of
            activations in this time step, and the outer list is of length equal to the number of time steps. Empty
            inner lists correspond to time steps where no activations should be performed.
        """
        self._actor_assignment = iter(actor_assignment)

        super().__init__(mtt_tracker=mtt_tracker,
                         actors=actors,
                         T=T,
                         actor_model_dict=actor_model_dict)

    def _assign_particle_to_actor(self,
                                  decision_mask,
                                  delta_toa,
                                  particle_class,
                                  particle_idx,
                                  t_actors_next_time_ready,
                                  prescribed_activation_times=None):
        raise NotImplementedError("This method is not implemented in the ControllerDummy.")

    def control_from_states(self, particle_states, particle_class, particle_id, t_act):
        # expand the states
        motion_state_mean = particle_states[0]

        # initialize the array
        activation_delta_t = np.empty((len(self._actor_model.pos), self._max_actions))
        activation_delta_t[:] = np.inf

        # get the time of arrival
        delta_toa = self.predict_time_of_arrival(motion_state=motion_state_mean)  # array of shape
        # [num_actors, num_particles]

        # make a decision
        decision_mask = np.zeros_like(delta_toa, dtype=bool)
        actor_assignment = next(self._actor_assignment)
        if len(actor_assignment) > 0:
            assigned_particle_ids, actor_idxs = zip(*actor_assignment)
            idx_map = {value: idx for idx, value in enumerate(particle_id)}
            particle_idxs = np.array([idx_map[x] for x in assigned_particle_ids])
            decision_mask[actor_idxs, particle_idxs] = True

        # go from hitting times to activation times
        activation_delta_t[:, :delta_toa.shape[1]] = np.where(decision_mask, delta_toa,
                                                              np.inf) - self._actor_model.t_lead

        return activation_delta_t


class PTCRControllerIntegrationTest(tf.test.TestCase):
    """Integration test for the LSTD class on a LQG example."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # common parameters
        cls.T = 1
        cls.T_variant_b = 2.5  # in ms
        cls.controller_rate = 5
        length = 4
        width = 9
        cls.actor_dict = {"t_activate": 1,
                          "t_up": 1,
                          "t_hit": 2,
                          "t_down": 1,
                          "t_reset": 4}
        cls.actor_dict_variant_b = {'t_activate': 10,  # in ms, actually ~10ms (see Master's thesis Leo Schön, pg. 89)
                                    't_up': 0,  # actually ~0-1ms
                                    't_hit': 2.5,  # actually ~3ms
                                    't_down': 5,  # actually ~6ms
                                    't_reset': 5}

        # define the full grid, actors, and particle simulator
        actor_positions = np.array(  # in mm, positions correspond to the positions of the ISAS-TableSort with the
            # origin of the coordinate system fixed in the lower left corner of the (physical) actuator array assembly
            [[25., 17.5], [25., 33.5], [25., 49.5], [25., 65.5], [25., 81.5], [25., 97.5], [25., 113.5],
             [45., 11.5], [45., 27.5], [45., 43.5], [45., 59.5], [45., 75.5], [45., 91.5], [45., 107.5],
             [65., 21.5], [65., 37.5], [65., 53.5], [65., 69.5], [65., 85.5], [65., 101.5], [65., 117.5],
             [85., 15.5], [85., 31.5], [85., 47.5], [85., 63.5], [85., 79.5], [85., 95.5], [85., 111.5],
             [105., 9.5], [105., 25.5], [105., 41.5], [105., 57.5], [105., 73.5], [105., 89.5], [105., 105.5],
             [125., 19.5], [125., 35.5], [125., 51.5], [125., 67.5], [125., 83.5], [125., 99.5], [125., 115.5],
             [145., 13.5], [145., 29.5], [145., 45.5], [145., 61.5], [145., 77.5], [145., 93.5], [145., 109.5],
             [165., 7.5], [165., 23.5], [165., 39.5], [165., 55.5], [165., 71.5], [165., 87.5], [165., 103.5]])
        offset = 20  # we need to shift the array to the right for the simulation
        actor_positions[:, 0] += offset
        area_width = 125

        # define the actors for the simulator
        cls.actors = ActorSimulator(pos=actor_positions,
                                    length=length,
                                    width=width,
                                    T=cls.T,
                                    **cls.actor_dict)

        cls.actors_variant_b = ActorSimulator(pos=actor_positions,
                                              length=length,
                                              width=width,
                                              T=cls.T_variant_b,
                                              **cls.actor_dict_variant_b)

        # define the particle simulator for the full grid
        v0_x = 1.0
        total_birth_rate_mean = 0.12
        particle_simulator_dict = {'total_num_particles': 1000,
                                   'process_parameters': 2 * ({'type': 'CV',
                                                               'pos_0': (-2.5, area_width / 2),
                                                               'pos0_stddev': (0.0, 3 * area_width),
                                                               'v_0': (v0_x, 0.0),
                                                               'v0_stddev': (v0_x / 10.0, v0_x / 10.0),
                                                               'S_w': (0.0001, 0.0001),
                                                               },),
                                   'birth_rate_mean_and_stddev': ((total_birth_rate_mean / 2, 0.3),
                                                                  (total_birth_rate_mean / 2, 0.3)),
                                   'class_labels': (0, 1),
                                   'area_width': area_width,
                                   'particle_r': 3,
                                   }

        cls.particle_simulator = AreaParticleSimulator(T=cls.T,
                                                       **particle_simulator_dict,
                                                       seed=12345,  # fix the seed for reproducibility
                                                       )

        # define the 3x3 grid, actors, and particle simulator
        actor_positions = np.array(  # in mm
            [[45., 19.5], [45., 35.5], [45., 51.5],
             [65., 13.5], [65., 29.5], [65., 45.5],
             [85., 7.5], [85., 23.5], [85., 39.5]])
        offset = 45  # we need to shift the array to the right for the simulation
        actor_positions[:, 0] += offset
        cls.array_end_3x3 = 100 + offset
        area_width = 59
        avg_particles_in_array = 8  # average number of particles (visible & invisible, i.e., ejected) that should be
        # in the sorter, defines the occupation density/mass flow of the sorter

        # define the actors for the simulator
        cls.actors_3x3 = ActorSimulator(pos=actor_positions,
                                        length=length,
                                        width=width,
                                        T=cls.T,
                                        **cls.actor_dict)

        # define the particle simulator for the 3x3 grid
        cls.v0_x_3x3 = 10
        avg_num_time_steps_particle_in_array = lambda array_end, T: int(np.ceil(array_end / (cls.v0_x_3x3 * T)))
        total_birth_rate_mean = avg_particles_in_array / avg_num_time_steps_particle_in_array(cls.array_end_3x3, cls.T)
        particle_simulator_dict = {'total_num_particles': 1000,
                                   'process_parameters': 2 * ({'type': 'CV',
                                                               'pos_0': (-2.5, area_width / 2),
                                                               'pos0_stddev': (0.0, 3 * area_width),
                                                               'v_0': (cls.v0_x_3x3, 0.0),
                                                               'v0_stddev': (cls.v0_x_3x3 / 10.0, cls.v0_x_3x3 / 10.0),
                                                               'S_w': (0.0, 0.0),
                                                               },),
                                   'birth_rate_mean_and_stddev': ((total_birth_rate_mean / 2, 0.3),
                                                                  (total_birth_rate_mean / 2, 0.3)),
                                   'class_labels': (0, 1),
                                   'area_width': area_width,
                                   'particle_r': 3,
                                   }

        cls.particle_simulator_3x3 = AreaParticleSimulator(T=cls.T,
                                                           **particle_simulator_dict,
                                                           seed=12345,  # fix the seed for reproducibility
                                                           )

        disturbance_area_offset = np.array([-6, 0])
        disturbance_area_length = 18
        disturbance_area_width = 10
        spatial_tolerance = cls.v0_x_3x3 * cls.T / 2  # should be ~ v0x * T / 2 when using CV model
        cls.simulator_3x3 = AreaSortingSimulator(
            cls.particle_simulator_3x3,
            cls.actors_3x3,
            contact_simulator_dict={'spatial_tolerance': (spatial_tolerance, 0.0),
                                    'disturbance_area_offset': disturbance_area_offset,
                                    'disturbance_area_extent': (
                                        disturbance_area_length,
                                        disturbance_area_width),
                                    },
            array_end=cls.array_end_3x3,
        )

        # common kwargs for the controllers
        particle_model_dict = {
            'p_detect': 0.95,
            'S_w': (0.0001, 0.0001),
            'S_v': (0.01, 0.01),
        }

        contact_model_dict = {
            "p_eject": [0.0, 0.9, 0.2, 0.1],
            "use_sum_approximation": False,
        }

        cls.common_controller_kwargs = dict(
            particle_model_dict=particle_model_dict,
            contact_model_dict=contact_model_dict,
            particle_ordering="id",
            seed=987654321,
            debug=True
        )

    @staticmethod
    def create_particle_simulator_create_particles_patch(k_to_init_states_map):
        """A function that returns a replacement for the particle simulator's create_particles method.

        :param k_to_init_states_map: A dict mapping time step k to a list of length 2 of initial states for the
            particles to be created at that time step. The first entry of the list corresponds to particles of the
            accept class (i.e., the class that should be kept in the sorter), and the second entry corresponds to
            particles of the reject class (i.e., the class that should be ejected from the sorter). Each entry of
            the list should be a numpy array of shape (n_particles, 4), the initial state of the created particles.

        :returns: A callable, the function used to patch the simulator's create_particles method.
        """

        def particle_simulator_create_particles_patch(self):
            if self._k in k_to_init_states_map.keys():
                init_states = k_to_init_states_map[self._k]
            else:
                return

            for process_no, (process_parameters, init_state) in enumerate(
                    zip(self._process_parameters, init_states)):
                self._sampled_motion_states[process_no] = np.r_[self._sampled_motion_states[process_no], init_state]
                self._particle_id = np.r_[
                    self._particle_id, np.arange(self._counter, self._counter + len(init_state))]
                self._particle_id_in_motion_states[process_no] = np.r_[
                    self._particle_id_in_motion_states[process_no],
                    np.arange(self._counter, self._counter + len(init_state))]
                self._particle_class = np.r_[
                    self._particle_class, np.full(len(init_state),
                                                  fill_value=self._class_labels[process_no])]
                self._existence = np.r_[self._existence, np.full(len(init_state), fill_value=True)]
                self._counter += len(init_state)

        return particle_simulator_create_particles_patch

    def test_higher_N_R_includes_all_previous_control_sequences(self):
        motion_state_mean = np.array([[7.56114402e+00, 1.00611440e+00, 1.14830897e+02, 1.34707776e-01],
                                      [2.59690468e+00, 1.01938094e+00, 1.32366110e+01, -1.39670506e-01],
                                      [2.87065461e+00, 1.07413092e+00, 6.55800196e+01, 2.22215696e-01],
                                      [1.31331319e-01, 1.05253253e+00, 8.90742560e+01, -1.03372303e-02],
                                      [2.76779340e-02, 1.01107117e+00, 6.46055956e+00, 1.24142710e-01],
                                      [-2.50000000e+00, 1.16889961e+00, 2.70224236e+01, 1.88075025e-01]])

        motion_state_cov = np.array([[[1.04166667e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [6.25000000e-05, 5.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 1.04166667e-04, 6.25000000e-05],
                                      [0.00000000e+00, 0.00000000e+00, 6.25000000e-05, 5.00000000e-05]],

                                     [[1.04166667e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [6.25000000e-05, 5.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 1.04166667e-04, 6.25000000e-05],
                                      [0.00000000e+00, 0.00000000e+00, 6.25000000e-05, 5.00000000e-05]],

                                     [[1.04166667e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [6.25000000e-05, 5.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 1.04166667e-04, 6.25000000e-05],
                                      [0.00000000e+00, 0.00000000e+00, 6.25000000e-05, 5.00000000e-05]],

                                     [[1.04166667e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [6.25000000e-05, 5.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 1.04166667e-04, 6.25000000e-05],
                                      [0.00000000e+00, 0.00000000e+00, 6.25000000e-05, 5.00000000e-05]],

                                     [[1.04166667e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [6.25000000e-05, 5.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 1.04166667e-04, 6.25000000e-05],
                                      [0.00000000e+00, 0.00000000e+00, 6.25000000e-05, 5.00000000e-05]],

                                     [[1.04166667e-04, 6.25000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [6.25000000e-05, 5.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                                      [0.00000000e+00, 0.00000000e+00, 1.04166667e-04, 6.25000000e-05],
                                      [0.00000000e+00, 0.00000000e+00, 6.25000000e-05, 5.00000000e-05]]])

        particle_class = np.array([1, 0, 1, 0, 1, 1])

        particle_ids = np.array([0., 1., 2., 3., 4., 5.])

        init_t_act = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0.])

        particle_states = [motion_state_mean, motion_state_cov, np.ones(len(motion_state_mean))]

        controller = PTCRController(
            self.particle_simulator,
            actors=self.actors_variant_b,
            actor_model_dict=self.actor_dict_variant_b,
            T=self.controller_rate * self.T_variant_b,
            N=10,
            N_R=2,
            use_only_terminal_costs=True,
            use_olf=True,
            adf={"use_numba": False},
            **self.common_controller_kwargs,
        )
        controller.control_from_states(
            particle_states=particle_states,
            particle_class=particle_class,
            particle_id=particle_ids,
            t_act=init_t_act
        )
        considered_control_sequences_NR2 = controller.considered_control_sequences
        cost_NR2 = controller.all_costs[-1]

        controller = PTCRController(
            self.particle_simulator,
            actors=self.actors_variant_b,
            actor_model_dict=self.actor_dict_variant_b,
            T=self.controller_rate * self.T_variant_b,
            N=10,
            N_R=3,
            use_only_terminal_costs=True,
            use_olf=True,
            adf={"use_numba": False},
            **self.common_controller_kwargs,
        )
        controller.control_from_states(
            particle_states=particle_states,
            particle_class=particle_class,
            particle_id=particle_ids,
            t_act=init_t_act
        )
        considered_control_sequences_NR3 = controller.considered_control_sequences
        cost_NR3 = controller.all_costs[-1]

        # check if a control sequence of N_R = 2 is not included in the control sequences of N_R = 3
        counter = 0
        for i, cs1 in enumerate(considered_control_sequences_NR2):
            found = False
            for cs2 in considered_control_sequences_NR3:
                if cs1.shape == cs2.shape and np.allclose(cs1, cs2):
                    found = True
            if not found:
                counter += 1
                break

        self.assertEqual(counter, 0)  # all control sequences should be included
        self.assertLessEqual(cost_NR3, cost_NR2)  # in this case, they are equal

    def test_better_than_faf(self):
        controller = PTCRController(
            self.particle_simulator,
            actors=self.actors,
            actor_model_dict=self.actor_dict,
            T=self.controller_rate * self.T,
            N=4,
            N_R=2,
            use_only_terminal_costs=True,
            use_olf=True,
            adf={"use_numba": False},
            **self.common_controller_kwargs,
        )

        t_act = np.array([0.0] * 56)
        particle_classes = np.array([1, 0])
        particle_ids = np.array([0, 1])
        first_actor_pos = controller.actor_model.pos[47]
        sec_actor_pos = controller.actor_model.pos[54]
        third_actor_pos = controller.actor_model.pos[55]
        particle_state_1 = np.array(
            [first_actor_pos - 3 * (sec_actor_pos - first_actor_pos), (sec_actor_pos - first_actor_pos)]).flatten(
            order='F')
        particle_state_2 = np.array(
            [first_actor_pos - 3 * (third_actor_pos - first_actor_pos), (third_actor_pos - first_actor_pos)]).flatten(
            order='F')
        mean = np.array([particle_state_1, particle_state_2])
        cov = np.full((2, 4, 4), 0.0)
        cov[:] = np.diag([1e-15, 1e-15, 1e-15, 1e-15])[None, :, :]
        particle_states = [mean, cov, np.ones(len(mean))]

        expected_activation_delta_t = np.full((56, 1), np.inf)
        expected_activation_delta_t[54, 0] = 1  # the best control sequence is the one that does not eject p1

        activation_delta_t = controller.control_from_states(
            particle_states=particle_states,
            particle_class=particle_classes,
            particle_id=particle_ids,
            t_act=t_act
        )

        self.assertAllEqual(expected_activation_delta_t, activation_delta_t)
        self.assertNotAllEqual(expected_activation_delta_t, controller.faf_control_sequences[0][0, :, 0])

        cost_ptcr = controller.all_costs[-1]
        cost_faf = controller.faf_costs[-1]

        self.assertLess(cost_ptcr, cost_faf)

    def test_retain_faf_candidate(self):
        """With retain_faf_candidate=True, FAF is among candidates and chosen ADF cost is <= FAF cost."""
        t_act = np.array([0.0] * 56)
        particle_classes = np.array([1, 0])
        particle_ids = np.array([0, 1])
        first_actor_pos = self.actors.pos[47]
        sec_actor_pos = self.actors.pos[54]
        third_actor_pos = self.actors.pos[55]
        particle_state_1 = np.array(
            [first_actor_pos - 3 * (sec_actor_pos - first_actor_pos), (sec_actor_pos - first_actor_pos)]).flatten(
            order='F')
        particle_state_2 = np.array(
            [first_actor_pos - 3 * (third_actor_pos - first_actor_pos), (third_actor_pos - first_actor_pos)]).flatten(
            order='F')
        mean = np.array([particle_state_1, particle_state_2])
        cov = np.full((2, 4, 4), 0.0)
        cov[:] = np.diag([1e-15, 1e-15, 1e-15, 1e-15])[None, :, :]
        particle_states = [mean, cov, np.ones(len(mean))]

        controller_retain = PTCRController(
            self.particle_simulator,
            actors=self.actors,
            actor_model_dict=self.actor_dict,
            T=self.controller_rate * self.T,
            N=4,
            N_R=2,
            use_only_terminal_costs=True,
            use_olf=True,
            adf={"use_numba": False},
            max_branches=1,
            retain_faf_candidate=True,
            **self.common_controller_kwargs,
        )
        controller_retain.control_from_states(
            particle_states=particle_states,
            particle_class=particle_classes,
            particle_id=particle_ids,
            t_act=t_act,
        )

        faf_seq = controller_retain.faf_control_sequences[-1]
        candidates = controller_retain.considered_control_sequences
        self.assertTrue(
            np.any([np.allclose(seq, faf_seq[0], atol=1e-6) for seq in candidates]),
            msg='FAF sequence must be among candidates when retain_faf_candidate=True',
        )
        self.assertLessEqual(controller_retain.all_costs[-1], controller_retain.faf_costs[-1] + 1e-6)

        # Default (retain_faf_candidate=False): behavior still runs; cost comparison uses debug FAF path
        controller_default = PTCRController(
            self.particle_simulator,
            actors=self.actors,
            actor_model_dict=self.actor_dict,
            T=self.controller_rate * self.T,
            N=4,
            N_R=2,
            use_only_terminal_costs=True,
            use_olf=True,
            adf={"use_numba": False},
            max_branches=1,
            retain_faf_candidate=False,
            **self.common_controller_kwargs,
        )
        controller_default.control_from_states(
            particle_states=particle_states,
            particle_class=particle_classes,
            particle_id=particle_ids,
            t_act=t_act,
        )
        self.assertGreater(len(controller_default.faf_control_sequences), 0)
        self.assertGreater(len(controller_default.all_costs), 0)

    def test_ensure_faf_in_candidates_appends_when_missing(self):
        """_ensure_faf_in_candidates appends FAF when it is not already in the set."""
        controller = PTCRController(
            self.particle_simulator,
            actors=self.actors,
            actor_model_dict=self.actor_dict,
            T=self.controller_rate * self.T,
            N=4,
            N_R=2,
            use_only_terminal_costs=True,
            use_olf=True,
            adf={"use_numba": False},
            **self.common_controller_kwargs,
        )
        t_act = np.zeros(56)
        particle_classes = np.array([1, 1])
        particle_ids = np.array([0, 1])
        # two reject particles upstream of different actuators
        pos0 = controller.actor_model.pos[0]
        pos1 = controller.actor_model.pos[1]
        mean = np.array([
            np.array([pos0 - np.array([30.0, 0.0]), np.array([1.0, 0.0])]).flatten(order='F'),
            np.array([pos1 - np.array([30.0, 0.0]), np.array([1.0, 0.0])]).flatten(order='F'),
        ])
        faf_seq = controller._faf_control_from_states(mean, particle_classes, particle_ids, t_act)
        # Candidate set that deliberately excludes FAF (all-inf sequence)
        dummy = np.full_like(faf_seq, np.inf)
        out, returned_faf = controller._ensure_faf_in_candidates(
            dummy, mean, particle_classes, particle_ids, t_act)
        self.assertEqual(out.shape[0], 2)
        self.assertTrue(np.allclose(out[-1], returned_faf[0], atol=1e-6))
        self.assertTrue(np.allclose(returned_faf, faf_seq, atol=1e-6))

        # Idempotent: already present → no second append
        out2, _ = controller._ensure_faf_in_candidates(
            out, mean, particle_classes, particle_ids, t_act)
        self.assertEqual(out2.shape[0], 2)

    def test_on_toy_examples_where_faf_fails(self):

        controller = PTCRController(
            mtt_tracker=self.particle_simulator_3x3,
            actors=self.actors_3x3,
            actor_model_dict=self.actor_dict,
            T=self.controller_rate * self.T,
            N=6,
            N_R=2,
            use_only_terminal_costs=True,
            use_olf=True,
            adf={"use_numba": False},
            **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},
        )
        faf_controller = FirstActorFirstHeuristicController(
            self.particle_simulator_3x3,
            actors=self.actors_3x3,
            T=self.controller_rate * self.T,
            actor_model_dict=self.actor_dict,
        )

        with self.subTest(name='Test with one eject and one keep particle'):
            init_state_eject = np.array([[-2.5, self.v0_x_3x3, -10, 5]])  # should be ejected from the sorter
            init_state_keep = np.array([[-2.5, self.v0_x_3x3, 35, 0]])  # should be kept in the sorter
            k_to_init_states_map = {1: [np.empty((0, 4)), init_state_eject],  # only eject particle at time step 1
                                    2: [init_state_keep, np.empty((0, 4))],  # only keep particle at time step 2
                                    }

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 2
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    faf_controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                no_disturbed = no_disturbed_ns + no_disturbed_ps
                self.assertAllEqual([1, 0, 1, 0, 0], [no_tns, no_tps, no_fns, no_fps, no_disturbed])

                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 2
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                no_disturbed = no_disturbed_ns + no_disturbed_ps
                self.assertAllEqual([1, 1, 0, 0, 0], [no_tns, no_tps, no_fns, no_fps, no_disturbed])

        with self.subTest(name='Test with three eject particles and different speeds'):
            init_state_one = np.array([[-2.5, self.v0_x_3x3 + 1.0, 33, 0]])
            init_state_two = np.array([[-2.5, self.v0_x_3x3 - 1.0, 33, 0]])
            init_state_three = np.array([[-2.5, self.v0_x_3x3 - 2.0, 33, 0]])
            k_to_init_states_map = {1: [np.empty((0, 4)), init_state_one],
                                    2: [np.empty((0, 4)), init_state_two],
                                    6: [np.empty((0, 4)), init_state_three],
                                    }

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 3
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    faf_controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                no_disturbed = no_disturbed_ns + no_disturbed_ps
                self.assertAllEqual([1, 0, 0, 1, 1], [no_tns, no_tps, no_fns, no_fps, no_disturbed])

                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 3
                # although faf fails, there exists a perfect separation, see below
                particle_assignment_0 = []
                particle_assignment_1 = [(0, 4), (1, 1)]
                particle_assignment_2 = []
                particle_assignment_3 = [(2, 4)]
                dummy_controller = ControllerDummy(self.particle_simulator_3x3,
                                                   actors=self.actors_3x3,
                                                   T=self.controller_rate * self.T,
                                                   actor_model_dict=self.actor_dict,
                                                   actor_assignment=(particle_assignment_0,
                                                                     particle_assignment_1,
                                                                     particle_assignment_2,
                                                                     particle_assignment_3,
                                                                     particle_assignment_0,
                                                                     particle_assignment_0,
                                                                     particle_assignment_0,
                                                                     particle_assignment_0))

                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    dummy_controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                no_disturbed = no_disturbed_ns + no_disturbed_ps
                self.assertAllEqual([3, 0, 0, 0, 0], [no_tns, no_tps, no_fns, no_fps, no_disturbed])

                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 3
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                no_disturbed = no_disturbed_ns + no_disturbed_ps
                self.assertAllEqual([3, 0, 0, 0, 0], [no_tns, no_tps, no_fns, no_fps, no_disturbed])

    def test_perfect_separation(self):

        init_state_a = np.array([[-2.5, self.v0_x_3x3, 20, 1.5]])
        init_state_b = np.array([[-2.5, self.v0_x_3x3, 5, 1.75]])
        init_state_c = np.array([[-2.5, self.v0_x_3x3 - 1, 50, -0.5]])
        init_state_d = np.array([[-2.5, self.v0_x_3x3, 12, 0]])
        init_state_e = np.array([[-2.5, self.v0_x_3x3, 25, -1]])
        init_state_f = np.array([[-2.5, self.v0_x_3x3 - 1, 30, -0.5]])
        init_state_g = np.array([[-2.5, self.v0_x_3x3, 5, 1.5]])

        init_state_one = np.array([[-2.5, self.v0_x_3x3, 33, 0]])
        init_state_two = np.array([[-2.5, self.v0_x_3x3 - 1, 38, 0]])

        init_state_eject = np.array([[-2.5, self.v0_x_3x3, -10, 5]])
        init_state_keep = np.array([[-2.5, self.v0_x_3x3, 35, 0]])

        k_to_init_states_map = {1: [np.empty((0, 4)), init_state_b],
                                2: [init_state_c, init_state_d],
                                3: [np.empty((0, 4)), init_state_one],
                                4: [np.empty((0, 4)), init_state_two],
                                6: [init_state_e, np.empty((0, 4))],
                                8: [np.zeros((0, 4)), init_state_g],
                                10: [init_state_f, init_state_a],
                                12: [np.empty((0, 4)), np.concatenate((init_state_eject, init_state_b))],
                                14: [init_state_keep, np.empty((0, 4))],
                                15: [init_state_c, np.empty((0, 4))],
                                }

        # although faf will fail, there exists a perfect separation, see below
        particle_assignment_0 = []
        particle_assignment_1 = [(0, 0)]
        particle_assignment_2 = [(2, 3), (3, 4), (4, 1)]
        particle_assignment_3 = [(6, 7), (10, 0)]
        particle_assignment_4 = [(8, 8), (9, 5)]
        dummy_controller = ControllerDummy(self.particle_simulator_3x3,
                                           actors=self.actors_3x3,
                                           T=self.controller_rate * self.T,
                                           actor_model_dict=self.actor_dict,
                                           actor_assignment=(particle_assignment_0,
                                                             particle_assignment_1,
                                                             particle_assignment_2,
                                                             particle_assignment_3,
                                                             particle_assignment_4,
                                                             particle_assignment_0,
                                                             particle_assignment_0,
                                                             particle_assignment_0,
                                                             particle_assignment_0))

        with self.subTest(name='Test that dummy controller achieves perfect separation and faf fails'):
            faf_controller = FirstActorFirstHeuristicController(
                self.particle_simulator_3x3,
                actors=self.actors_3x3,
                T=self.controller_rate * self.T,
                actor_model_dict=self.actor_dict,
            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    faf_controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertLess(tnr, 1.0)

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13

                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    dummy_controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])

        with self.subTest(name='Test with multiple particles and N_R=2'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                N=6,
                N_R=2,
                use_only_terminal_costs=True,
                use_olf=True,
                adf={"use_numba": False},
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},
            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertGreaterEqual(tnr, 0.75)
                self.assertGreaterEqual(tpr, 0.8)

        with self.subTest(name='Test with multiple particles and N_R=4'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                N=6,
                N_R=4,
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},
            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])

        with self.subTest(name='Test with multiple particles and N_R=6'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                N=6,
                N_R=6,
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},
            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])

        with self.subTest(name='Test with multiple particles and N_R=10'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                N=6,
                N_R=10,
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},
            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])

        with self.subTest(name='Test with multiple particles for olf, step costs'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                N=6,
                N_R=10,
                use_only_terminal_costs=False,
                use_olf=True,
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},
            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])
                all_costs_internal_T_equals_T_cycle = controller.all_costs.copy()

        with self.subTest(name='Test result and costs is same for olf, step costs and different internal_T'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                internal_T=self.controller_rate * self.T,
                N=6,
                N_R=10,
                use_only_terminal_costs=False,
                use_olf=True,
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},

            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])
                # ADF costs can differ slightly across internal_T discretizations.
                self.assertAllClose(
                    all_costs_internal_T_equals_T_cycle,
                    controller.all_costs,
                    rtol=1e-3,
                    atol=1e-3,
                )

        with self.subTest(name='Test for olf, terminal costs'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                internal_T=self.controller_rate * self.T,
                N=6,
                N_R=10,
                use_only_terminal_costs=True,
                use_olf=True,
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},

            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])
                all_costs_internal_T_equals_T_cycle = controller.all_costs.copy()

        with self.subTest(name='Test cost is same for olf, terminal costs and different internal_T'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                internal_T=self.controller_rate * self.T,
                N=6,
                N_R=10,
                use_only_terminal_costs=True,
                use_olf=True,
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},

            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])
                self.assertAllClose(
                    all_costs_internal_T_equals_T_cycle,
                    controller.all_costs,
                    rtol=1e-3,
                    atol=1e-3,
                )

        with self.subTest(name='Test for clf and step costs'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                internal_T=self.controller_rate * self.T,
                N=6,
                N_R=10,
                use_only_terminal_costs=False,
                use_olf=True,
                adf={"use_numba": False},
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},

            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])

        with self.subTest(name='Test for clf and terminal costs'):
            controller = PTCRController(
                mtt_tracker=self.particle_simulator_3x3,
                actors=self.actors_3x3,
                actor_model_dict=self.actor_dict,
                T=self.controller_rate * self.T,
                internal_T=self.controller_rate * self.T,
                N=6,
                N_R=10,
                use_only_terminal_costs=True,
                use_olf=True,
                adf={"use_numba": False},
                **{k: v for k, v in self.common_controller_kwargs.items() if k not in ['mtt_tracker']},

            )

            with patch.object(AreaParticleSimulator, 'create_particles',
                              self.create_particle_simulator_create_particles_patch(k_to_init_states_map)):
                self.simulator_3x3.reset(restore_original_scenario=True)
                self.particle_simulator_3x3.total_num_particles = 13
                no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, _ = self.simulator_3x3.evaluate(
                    controller,
                    time_out=10,
                    no_show_animation=True,
                    save_animation=False,
                )
                self.assertAllEqual([1.0, 1.0], [tnr, tpr])


if __name__ == "__main__":
    tf.test.main()
