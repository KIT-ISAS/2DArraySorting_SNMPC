import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np

import tensorflow as tf

from simulator import ActorSimulator, AreaParticleSimulator, AreaSortingSimulator
from heuristics import FirstActorFirstHeuristicController


class AbstractParticleSimulatorTest(tf.test.TestCase):
    """Test cases for the AbstractParticleSimulator class."""

    @staticmethod
    def _create_area_particle_simulator(seed=1):
        area_width = 37
        particle_simulator_dict = {'total_num_particles': 100,
                                   'process_parameters': 2 * ({'type': 'CV',
                                                               'pos_0': (-2.5, area_width / 2),
                                                               'pos0_stddev': (0.0, 3 * area_width),
                                                               'v_0': (10.0, 0.0),
                                                               'v0_stddev': (0.01, 0.1),
                                                               'S_w': (0.000001, 0.000001),
                                                               },),
                                   'birth_rate_mean_and_stddev': ((0.1, 0.3), (0.1, 0.3)),
                                   'class_labels': (0, 1),
                                   'area_width': area_width,
                                   'particle_r': 3,
                                   }
        return AreaParticleSimulator(T=1.0, seed=seed, **particle_simulator_dict)

    def test_define_func(self):
        simulator = self._create_area_particle_simulator(seed=1)

        # raise a Value error if not in list of predefined functions and no lambda function
        with self.assertRaises(ValueError):
            value = "bla([0, 0], [1, 0], 0.2, 0.8)"
            func = simulator._define_func(value)

        # raise a Value error if not of supported type
        with self.assertRaises(ValueError):
            value = {1, 2}
            func = simulator._define_func(value)

        # raise a Value error if signature of functions is wrong
        with self.assertRaises(TypeError):
            value = "ramp([0, 0], [1, 0], bla=0.2, end=0.8)"
            func = simulator._define_func(value)

        # raise a Value error if list is of wrong shape
        with self.assertRaises(ValueError):
            value = [1, 1, 1]
            func = simulator._define_func(value)

        # for float or int
        value = 1
        func = simulator._define_func(value)
        self.assertEqual(func(0), value)
        self.assertEqual(func(-1111), value)
        value = 1.0
        func = simulator._define_func(value)
        self.assertEqual(func(0), value)
        self.assertEqual(func(-1111), value)

        # for list or tuple
        value = [1, 0]
        func = simulator._define_func(value)
        self.assertAllEqual(func(0), value)
        self.assertAllEqual(func(-1111), value)
        value = (1,)
        func = simulator._define_func(value)
        self.assertAllEqual(func(0), value)

        # for function
        value = lambda k: (2 * k, 1)
        func = simulator._define_func(value)
        self.assertAllEqual(func(0), np.array([0, 1]))
        self.assertAllEqual(func(100), np.array([200, 1]))

        # for lambda string
        value = "lambda k: (2 * k, 1)"
        func = simulator._define_func(value)
        self.assertAllEqual(func(0), np.array([0, 1]))
        self.assertAllEqual(func(100), np.array([200, 1]))

        # for predefined function
        value = "ramp([0, 0], [1, 0], 0.2, 0.8)"
        func = simulator._define_func(value)
        self.assertAllEqual(func(0), np.array([0, 0]))
        self.assertAllEqual(func(1), np.array([1, 0]))
        self.assertAllClose(func(0.5), np.array([0.5, 0]))

        value = "ramp([0, 0], [1, 0], start=0.2, end=0.8)"
        func = simulator._define_func(value)
        self.assertAllEqual(func(0), np.array([0, 0]))
        self.assertAllEqual(func(1), np.array([1, 0]))
        self.assertAllClose(func(0.5), np.array([0.5, 0]))

        value = "ramp((0, 0), (1, 0), start=0.2, end=0.8)"
        func = simulator._define_func(value)
        self.assertAllEqual(func(0), np.array([0, 0]))
        self.assertAllEqual(func(1), np.array([1, 0]))
        self.assertAllClose(func(0.5), np.array([0.5, 0]))

    def test_shallow_copy(self):
        particle_simulator = self._create_area_particle_simulator(seed=1)

        with self.subTest(name="Test with given seed equal to original seed and freeze_scenario=True"):
            particle_simulator_frozen_copy = particle_simulator.shallow_copy(seed=1, freeze_scenario=True)
            self.assertEqual(1, particle_simulator_frozen_copy._original_seed)
            particle_simulator_frozen_copy.reset()
            self.assertEqual(1, particle_simulator_frozen_copy._original_seed)
            particle_simulator_copy_unfrozen_copy = particle_simulator_frozen_copy.shallow_copy(freeze_scenario=False)
            self.assertIsNone(particle_simulator_copy_unfrozen_copy._original_seed)

        with self.subTest(name="Test with given seed not equal to original seed and freeze_scenario=True"):
            particle_simulator_frozen_copy = particle_simulator.shallow_copy(seed=2, freeze_scenario=True)
            self.assertEqual(2, particle_simulator_frozen_copy._original_seed)
            particle_simulator_frozen_copy.reset()
            self.assertEqual(2, particle_simulator_frozen_copy._original_seed)
            particle_simulator_copy_unfrozen_copy = particle_simulator_frozen_copy.shallow_copy(seed=3,
                                                                                                freeze_scenario=False)
            self.assertEqual(3, particle_simulator_copy_unfrozen_copy._original_seed)

        with self.subTest(name="Test without given seed and freeze_scenario=True"):
            particle_simulator_frozen_copy = particle_simulator.shallow_copy(freeze_scenario=True)
            self.assertEqual(1, particle_simulator_frozen_copy._original_seed)
            particle_simulator_frozen_copy.reset()
            self.assertEqual(1, particle_simulator_frozen_copy._original_seed)

        with self.subTest(
                name="Test where and original particle simulator has no seed with given seed and freeze_scenario=True"):
            particle_simulator = self._create_area_particle_simulator(seed=None)
            particle_simulator_frozen_copy = particle_simulator.shallow_copy(seed=1, freeze_scenario=True)
            self.assertEqual(1, particle_simulator_frozen_copy._original_seed)

        with self.subTest(name="Test where and original particle simulator has no seed "
                               "without given seed and freeze_scenario=True"):
            with self.assertRaisesWithLiteralMatch(
                    ValueError,
                    'If freeze_scenario is True, a seed for the random number generator must be given.'):
                particle_simulator.shallow_copy(freeze_scenario=True)


class AreaParticleSimulatorTest(tf.test.TestCase):
    """Test cases for the AreaParticleSimulator class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        area_width = 37
        total_birth_rate_mean = 0.2

        particle_simulator_dict = {'total_num_particles': 100,
                                   'process_parameters': 2 * ({'type': 'CV',
                                                               'pos_0': (-2.5, area_width / 2),
                                                               'pos0_stddev': (0.0, 3 * area_width),
                                                               'v_0': (10.0, 0.0),
                                                               'v0_stddev': (0.01, 0.1),
                                                               'S_w': (0.000001, 0.000001),
                                                               },),
                                   'birth_rate_mean_and_stddev': ((total_birth_rate_mean / 2, 0.3),
                                                                  (total_birth_rate_mean / 2, 0.3)),
                                   'class_labels': (0, 1),
                                   'area_width': area_width,
                                   'particle_r': 3,
                                   }

        cls.particle_simulator = AreaParticleSimulator(T=0.5,
                                                       seed=1,
                                                       **particle_simulator_dict)

    def test_set_total_num_particles(self):
        self.particle_simulator.total_num_particles = 200
        self.assertEqual(self.particle_simulator.total_num_particles, 200)

        self.particle_simulator.reset()
        self.assertEqual(self.particle_simulator.total_num_particles, 200)

        self.particle_simulator.reset(restore_original_scenario=True)
        self.assertEqual(self.particle_simulator.total_num_particles, 100)

        self.particle_simulator.total_num_particles = 200
        self.particle_simulator.reset(seed=15)
        self.assertEqual(self.particle_simulator.total_num_particles, 200)

        self.particle_simulator.reset(seed=16)
        self.particle_simulator.total_num_particles = 300
        self.assertEqual(self.particle_simulator.total_num_particles, 300)

        particle_simulator = self.particle_simulator.shallow_copy(freeze_scenario=True)
        self.assertEqual(particle_simulator.total_num_particles, 300)

        with self.assertRaisesWithLiteralMatch(
                ValueError,
                'The scenario is frozen, the total number of particles cannot be changed.'):
            particle_simulator.total_num_particles = 400

        particle_simulator.reset()
        self.assertEqual(particle_simulator.total_num_particles, 300)

        particle_simulator.reset(restore_original_scenario=True)
        self.assertEqual(particle_simulator.total_num_particles, 300)

        with self.assertRaisesWithLiteralMatch(
                ValueError,
                'The seed cannot be set if restore_original_scenario or self._freeze_scenario is True.'):
            particle_simulator.reset(seed=2)


class AbstractSortingSimulatorTest(tf.test.TestCase):
    """Test cases for the AbstractSortingSimulator class."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.T = 1
        cls.actor_dict = {'t_activate': 1,
                          't_up': 1,
                          't_hit': 2,
                          't_down': 1,
                          't_reset': 4}
        cls.T = 0.5
        cls.actor_positions = np.array([[45., 13.5], [45., 29.5],
                                        [65., 7.5], [65., 23.5]])
        array_end = 80
        area_width = 37
        total_birth_rate_mean = 0.2
        cls.length = 4
        cls.width = 9
        v0_x = 10.0
        disturbance_area_offset = np.array([-6, 0])
        disturbance_area_length = 18
        disturbance_area_width = 10

        cls.actors = ActorSimulator(cls.actor_positions,
                                    length=cls.length,
                                    width=cls.width,
                                    T=cls.T,
                                    **cls.actor_dict)

        cls.particle_simulator_dict = {'total_num_particles': 100,
                                       'process_parameters': 2 * ({'type': 'CV',
                                                                   'pos_0': (-2.5, area_width / 2),
                                                                   'pos0_stddev': (0.0, 3 * area_width),
                                                                   'v_0': (v0_x, 0.0),
                                                                   'v0_stddev': (0.01, 0.1),
                                                                   'S_w': (0.000001, 0.000001),
                                                                   },),
                                       'birth_rate_mean_and_stddev': ((total_birth_rate_mean / 2, 0.3),
                                                                      (total_birth_rate_mean / 2, 0.3)),
                                       'class_labels': (0, 1),
                                       'area_width': area_width,
                                       'particle_r': 3,
                                       }

        cls.particle_simulator = AreaParticleSimulator(T=cls.T,
                                                       seed=1,
                                                       **cls.particle_simulator_dict)

        spatial_tolerance = v0_x * cls.T / 2  # should be ~ v0x * T / 2 when using CV model
        cls.simulator = AreaSortingSimulator(
            particle_simulator=cls.particle_simulator,
            actors=cls.actors,
            contact_simulator_dict={'spatial_tolerance': (spatial_tolerance, 0.0),
                                    'disturbance_area_offset': disturbance_area_offset,
                                    'disturbance_area_extent': (
                                        disturbance_area_length,
                                        disturbance_area_width),
                                    },
            array_end=array_end,
        )
        cls.time_out = 30

    def test_create_initial_state_estimates_iterator(self):

        class BehaviorPolicyWrapper:

            def __init__(self, behavior_policy):
                self.behavior_policy = behavior_policy
                self.estimated_motion_state_mean_ls = []
                self.estimated_motion_state_cov_ls = []
                self.particle_class_ls = []
                self.particle_id_ls = []
                self.existence_ls = []
                self.controls_ls = []

            @property
            def T(self):
                return self.behavior_policy.T

            def control_from_measurements(self, measurements, mtt_tracker, actors):
                controls = self.behavior_policy.control_from_measurements(measurements=None,
                                                                          mtt_tracker=mtt_tracker,
                                                                          actors=actors)
                (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id,
                 existence) = mtt_tracker.track_particles(
                    measurements)
                if particle_class.size != 0:
                    self.estimated_motion_state_mean_ls.append(estimated_motion_state_mean.copy())
                    self.estimated_motion_state_cov_ls.append(estimated_motion_state_cov.copy())
                    self.particle_class_ls.append(particle_class.copy())
                    self.particle_id_ls.append(particle_id.copy())
                    self.existence_ls.append(existence.copy())
                    self.controls_ls.append(controls.copy())
                return controls

        with self.subTest(name="Test behavior policy controller rate = 1"):
            behavior_policy = FirstActorFirstHeuristicController(self.particle_simulator,
                                                                 actors=self.actors,
                                                                 T=1 * self.T,
                                                                 actor_model_dict=self.actor_dict)

            behavior_policy_wrapper = BehaviorPolicyWrapper(behavior_policy)

            it = self.simulator._create_initial_state_estimates_iterator(behavior_policy_wrapper,
                                                                         time_out=self.time_out)

            for i, (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                    t_act, controls) in enumerate(it):
                self.assertDTypeEqual(estimated_motion_state_mean, float)
                self.assertDTypeEqual(estimated_motion_state_cov, float)
                self.assertDTypeEqual(particle_class, int)
                self.assertDTypeEqual(particle_id, int)
                self.assertDTypeEqual(existence, int)
                self.assertDTypeEqual(t_act, float)
                self.assertDTypeEqual(controls, float)

                self.assertAllClose(behavior_policy_wrapper.estimated_motion_state_mean_ls[i],
                                    estimated_motion_state_mean)
                self.assertAllClose(behavior_policy_wrapper.estimated_motion_state_cov_ls[i],
                                    estimated_motion_state_cov)
                self.assertAllEqual(behavior_policy_wrapper.particle_class_ls[i], particle_class)
                self.assertAllEqual(behavior_policy_wrapper.particle_id_ls[i], particle_id)
                self.assertAllEqual(behavior_policy_wrapper.existence_ls[i], existence)
                self.assertAllClose(behavior_policy_wrapper.controls_ls[i], controls)

        with self.subTest(name="Test behavior policy controller rate > 1"):
            behavior_policy = FirstActorFirstHeuristicController(self.particle_simulator,
                                                                 actors=self.actors,
                                                                 T=2 * self.T,
                                                                 actor_model_dict=self.actor_dict)

            behavior_policy_wrapper = BehaviorPolicyWrapper(behavior_policy)

            it = self.simulator._create_initial_state_estimates_iterator(behavior_policy_wrapper,
                                                                         time_out=self.time_out)

            idx = 0
            for i, (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                    t_act, controls) in enumerate(it):
                if i % 2 == 0:
                    self.assertDTypeEqual(estimated_motion_state_mean, float)
                    self.assertDTypeEqual(estimated_motion_state_cov, float)
                    self.assertDTypeEqual(particle_class, int)
                    self.assertDTypeEqual(particle_id, int)
                    self.assertDTypeEqual(existence, int)
                    self.assertDTypeEqual(t_act, float)
                    self.assertDTypeEqual(controls, float)

                    self.assertAllClose(behavior_policy_wrapper.estimated_motion_state_mean_ls[idx],
                                        estimated_motion_state_mean)
                    self.assertAllClose(behavior_policy_wrapper.estimated_motion_state_cov_ls[idx],
                                        estimated_motion_state_cov)
                    self.assertAllEqual(behavior_policy_wrapper.particle_class_ls[idx], particle_class)
                    self.assertAllEqual(behavior_policy_wrapper.particle_id_ls[idx], particle_id)
                    self.assertAllEqual(behavior_policy_wrapper.existence_ls[idx], existence)
                    self.assertAllClose(behavior_policy_wrapper.controls_ls[idx], controls)
                else:
                    self.assertAllClose(behavior_policy_wrapper.controls_ls[idx] - self.T, controls)
                    idx += 1

    def test_reset(self):

        actors = ActorSimulator(pos=self.actor_positions,
                                length=self.length,
                                width=self.width,
                                T=self.T,
                                **self.actor_dict)

        particle_simulator = AreaParticleSimulator(
            T=self.T,
            seed=1,
            **self.particle_simulator_dict)

        simulator = AreaSortingSimulator(
            particle_simulator=particle_simulator,
            actors=actors,
            contact_simulator_dict={'spatial_tolerance': (0.5, 0.0),
                                    'disturbance_area_offset': np.array([-6, 0]),
                                    'disturbance_area_extent': (18, 10),
                                    },
            array_end=20,
        )

        behavior_policy = FirstActorFirstHeuristicController(self.particle_simulator,
                                                             actors=self.actors,
                                                             T=1 * self.T,
                                                             actor_model_dict=self.actor_dict)

        with self.subTest(name="Test with evaluate and two different seeds"):

            no_tns_0, no_tps_0, no_fns_0, no_fps_0, no_disturbed_ns_0, no_disturbed_ps_0, _, _, _ = simulator.evaluate(
                behavior_policy,
                time_out=self.time_out,
                no_show_animation=True)

            simulator.reset(seed=4)
            no_tns_1, no_tps_1, no_fns_1, no_fps_1, no_disturbed_ns_1, no_disturbed_ps_1, _, _, _ = simulator.evaluate(
                behavior_policy,
                time_out=self.time_out,
                no_show_animation=True)

            no_disturbed_0 = no_disturbed_ns_0 + no_disturbed_ps_0
            no_disturbed_1 = no_disturbed_ns_1 + no_disturbed_ps_1
            self.assertFalse(np.allclose([no_tns_0, no_tps_0, no_fns_0, no_fps_0, no_disturbed_0],
                                         [no_tns_1, no_tps_1, no_fns_1, no_fps_1, no_disturbed_1]))

        with self.subTest(name="Test with evaluate and two consecutive resets"):

            simulator.reset(seed=None)
            no_tns_2, no_tps_2, no_fns_2, no_fps_2, no_disturbed_ns_2, no_disturbed_ps_2, _, _, _ = simulator.evaluate(
                behavior_policy,
                time_out=self.time_out,
                no_show_animation=True)

            no_disturbed_2 = no_disturbed_ns_2 + no_disturbed_ps_2
            self.assertFalse(np.allclose([no_tns_1, no_tps_1, no_fns_1, no_fps_1, no_disturbed_1],
                                         [no_tns_2, no_tps_2, no_fns_2, no_fps_2, no_disturbed_2]))

        with self.subTest(name="Test with evaluate and restore_original_scenario=True"):

            simulator.reset(restore_original_scenario=True)
            no_tns_3, no_tps_3, no_fns_3, no_fps_3, no_disturbed_ns_3, no_disturbed_ps_3, _, _, _ = simulator.evaluate(
                behavior_policy,
                time_out=self.time_out,
                no_show_animation=True)
            no_disturbed_3 = no_disturbed_ns_3 + no_disturbed_ps_3
            self.assertAllEqual([no_tns_0, no_tps_0, no_fns_0, no_fps_0, no_disturbed_0],
                                [no_tns_3, no_tps_3, no_fns_3, no_fps_3, no_disturbed_3])

        with self.subTest(name="Test with evaluate and a seed used before"):

            simulator.reset(seed=4)
            no_tns_4, no_tps_4, no_fns_4, no_fps_4, no_disturbed_ns_4, no_disturbed_ps_4, _, _, _ = simulator.evaluate(
                behavior_policy,
                time_out=self.time_out,
                no_show_animation=True)
            no_disturbed_4 = no_disturbed_ns_4 + no_disturbed_ps_4
            self.assertAllEqual([no_tns_1, no_tps_1, no_fns_1, no_fps_1, no_disturbed_1],
                                [no_tns_4, no_tps_4, no_fns_4, no_fps_4, no_disturbed_4])

        with self.subTest(name="Test with create_initial_state_estimates_iterator and two different seeds"):

            simulator.reset(restore_original_scenario=True)
            ds_0 = list(simulator._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))

            simulator.reset(seed=4)
            ds_1 = list(simulator._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))

            allequal = True
            for (estimated_motion_state_mean_0, estimated_motion_state_cov_0, particle_class_0, particle_id_0,
                 existence_0, t_act_0, controls_0), (estimated_motion_state_mean_1, estimated_motion_state_cov_1,
                                                     particle_class_1, particle_id_1, existence_1, t_act_1,
                                                     controls_1) in zip(ds_0, ds_1):
                allequal = allequal and np.allclose(estimated_motion_state_mean_0, estimated_motion_state_mean_1)
            self.assertFalse(allequal)

        with self.subTest(name="Test with create_initial_state_estimates_iterator and two consecutive resets"):

            simulator.reset(seed=None)
            ds_2 = list(simulator._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))

            allequal = True
            for (estimated_motion_state_mean_0, estimated_motion_state_cov_0, particle_class_0, particle_id_0,
                 existence_0, t_act_0, controls_0), (estimated_motion_state_mean_1, estimated_motion_state_cov_1,
                                                     particle_class_1, particle_id_1, existence_1, t_act_1,
                                                     controls_1) in zip(ds_1, ds_2):
                allequal = allequal and np.allclose(estimated_motion_state_mean_0, estimated_motion_state_mean_1)
            self.assertFalse(allequal)

        with self.subTest(name="Test with create_initial_state_estimates_iterator and restore_original_scenario=True"):

            simulator.reset(restore_original_scenario=True)
            ds_3 = list(simulator._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))

            for (estimated_motion_state_mean_0, estimated_motion_state_cov_0, particle_class_0, particle_id_0,
                 existence_0, t_act_0, controls_0), (estimated_motion_state_mean_1, estimated_motion_state_cov_1,
                                                     particle_class_1, particle_id_1, existence_1, t_act_1,
                                                     controls_1) in zip(ds_0, ds_3):
                self.assertAllEqual(estimated_motion_state_mean_0, estimated_motion_state_mean_1)
                self.assertAllEqual(estimated_motion_state_cov_0, estimated_motion_state_cov_1)
                self.assertAllEqual(particle_class_0, particle_class_1)
                self.assertAllEqual(particle_id_0, particle_id_1)
                self.assertAllEqual(existence_0, existence_1)
                self.assertAllEqual(t_act_0, t_act_1)
                self.assertAllEqual(controls_0, controls_1)

        with self.subTest(name="Test with create_initial_state_estimates_iterator and a seed used before"):

            simulator.reset(seed=4)
            ds_4 = list(simulator._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))

            for (estimated_motion_state_mean_0, estimated_motion_state_cov_0, particle_class_0, particle_id_0,
                 existence_0, t_act_0, controls_0), (estimated_motion_state_mean_1, estimated_motion_state_cov_1,
                                                     particle_class_1, particle_id_1, existence_1, t_act_1,
                                                     controls_1) in zip(ds_1, ds_4):
                self.assertAllEqual(estimated_motion_state_mean_0, estimated_motion_state_mean_1)
                self.assertAllEqual(estimated_motion_state_cov_0, estimated_motion_state_cov_1)
                self.assertAllEqual(particle_class_0, particle_class_1)
                self.assertAllEqual(particle_id_0, particle_id_1)
                self.assertAllEqual(existence_0, existence_1)
                self.assertAllEqual(t_act_0, t_act_1)
                self.assertAllEqual(controls_0, controls_1)

    def test_shallow_copy(self):

        behavior_policy = FirstActorFirstHeuristicController(self.particle_simulator,
                                                             actors=self.actors,
                                                             T=1 * self.T,
                                                             actor_model_dict=self.actor_dict)

        self.simulator.reset(restore_original_scenario=True)
        simulator_copy_1 = self.simulator.shallow_copy(seed=1, freeze_scenario=True)  # copy the original scenario
        simulator_copy_2 = self.simulator.shallow_copy(seed=3)

        with self.subTest(name="Test with create_initial_state_estimates_iterator and with freeze_scenario=True"):

            ds_0 = list(
                self.simulator._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))
            ds_1 = list(
                simulator_copy_1._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))

            for (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                 t_act, controls), (estimated_motion_state_mean_1, estimated_motion_state_cov_1, particle_class_1,
                                    particle_id_1, existence_1, t_act_1, controls_1) in zip(ds_0, ds_1):
                self.assertAllEqual(estimated_motion_state_mean, estimated_motion_state_mean_1)
                self.assertAllEqual(estimated_motion_state_cov, estimated_motion_state_cov_1)
                self.assertAllEqual(particle_class, particle_class_1)
                self.assertAllEqual(particle_id, particle_id_1)
                self.assertAllEqual(existence, existence_1)
                self.assertAllEqual(t_act, t_act_1)
                self.assertAllEqual(controls, controls_1)

            # test the fix seed
            simulator_copy_1.reset()
            ds_1 = list(
                simulator_copy_1._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))
            for (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence, t_act,
                 controls), (estimated_motion_state_mean_1, estimated_motion_state_cov_1, particle_class_1,
                             particle_id_1, existence_1, t_act_1, controls_1) in zip(ds_0, ds_1):
                self.assertAllEqual(estimated_motion_state_mean, estimated_motion_state_mean_1)
                self.assertAllEqual(estimated_motion_state_cov, estimated_motion_state_cov_1)
                self.assertAllEqual(particle_class, particle_class_1)
                self.assertAllEqual(particle_id, particle_id_1)
                self.assertAllEqual(existence, existence_1)
                self.assertAllEqual(t_act, t_act_1)
                self.assertAllEqual(controls, controls_1)

        with self.subTest(name="Test with create_initial_state_estimates_iterator and without freeze_scenario"):

            ds_2 = list(
                simulator_copy_2._create_initial_state_estimates_iterator(behavior_policy, time_out=self.time_out))

            allequal = True
            for (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence, t_act,
                 controls), (estimated_motion_state_mean_1, estimated_motion_state_cov_1, particle_class_1,
                             particle_id_1, existence_1, t_act_1, controls_1) in zip(ds_0, ds_2):
                allequal = allequal and np.allclose(estimated_motion_state_mean, estimated_motion_state_mean_1)
            self.assertFalse(allequal)

        with self.subTest(name="Test with evaluate and with restore_original_scenario=True"):

            self.simulator.reset(restore_original_scenario=True)
            no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, _, _, _ = self.simulator.evaluate(
                behavior_policy,
                time_out=self.time_out,
                no_show_animation=True,
                save_animation=False)

            simulator_copy_1.reset(restore_original_scenario=True)
            no_tns_1, no_tps_1, no_fns_1, no_fps_1, no_disturbed_ns_1, no_disturbed_ps_1, _, _, _ = simulator_copy_1.evaluate(
                behavior_policy,
                time_out=self.time_out,
                no_show_animation=True,
                save_animation=False)

            no_disturbed = no_disturbed_ns + no_disturbed_ps
            no_disturbed_1 = no_disturbed_ns_1 + no_disturbed_ps_1
            self.assertAllEqual([no_tns, no_tps, no_fns, no_fps, no_disturbed],
                                [no_tns_1, no_tps_1, no_fns_1, no_fps_1, no_disturbed_1])

    def test_create_buffered_initial_state_estimates_data_set_iterator(self):
        behavior_policy_small_grid = FirstActorFirstHeuristicController(self.particle_simulator,
                                                                        actors=self.actors,
                                                                        T=1 * self.T,
                                                                        actor_model_dict=self.actor_dict)
        behavior_policy = FirstActorFirstHeuristicController(self.particle_simulator,
                                                             actors=self.actors,
                                                             T=1 * self.T,
                                                             actor_model_dict=self.actor_dict)

        with self.subTest(name="Test with given num_scenarios"):
            self.simulator.reset(seed=5)

            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy_small_grid,
                num_scenarios=5,
                num_particles_per_scenario=19,
                scenario_time_out=10,
            )

            self.assertEqual(5, len(scenario_len_list))
            self.assertTrue(len(np.unique(scenario_len_list)) > 1)
            for i in range(4):
                # compare the motion state means at the third time step per scenario (first is always equal because of
                # line)
                self.assertNotAllEqual(it[2][0], it[2 + scenario_len_list[i]][0])

                # check the dtypes
                (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                 t_act, controls) = it[2]
                self.assertDTypeEqual(estimated_motion_state_mean, float)
                self.assertDTypeEqual(estimated_motion_state_cov, float)
                self.assertDTypeEqual(particle_class, int)
                self.assertDTypeEqual(particle_id, int)
                self.assertDTypeEqual(existence, int)
                self.assertDTypeEqual(t_act, float)
                self.assertDTypeEqual(controls, float)

            # now the for area
            self.simulator.reset(seed=5)
            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_scenarios=5,
            )
            l = list(it)
            self.assertGreater(len(l), 0)

            # test that a new iterator does not have the same elements
            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_scenarios=5,
            )
            l_new = list(it)
            self.assertFalse(np.allclose(l_new[-1][0], l[-1][0]))

            # now reset the iterator and check that it is the same as the first one
            self.simulator.reset(seed=5)
            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_scenarios=5,
            )
            l_res = list(it)
            self.assertEqual(len(l), len(l_res))
            for elem_org, elem_res in zip(l, l_res):
                for i in range(len(elem_org)):
                    self.assertAllEqual(elem_org[i], elem_res[i])

        with self.subTest(name="Test with given num_examples"):
            self.simulator.reset(seed=5)

            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy_small_grid,
                num_examples=650,
                num_particles_per_scenario=19,
                scenario_time_out=10,
            )

            self.assertEqual(650, num_examples)
            self.assertEqual(650, len(list(it)))
            self.assertTrue(len(np.unique(scenario_len_list)) > 1)
            it.reset()
            for i in range(4):
                # compare the motion state means at the third time step per scenario (first is always equal because of
                # line)
                self.assertNotAllEqual(it[2][0], it[2 + scenario_len_list[i]][0])

            # now the for area
            self.simulator.reset(seed=5)
            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_examples=235,
            )
            l = list(it)
            self.assertEqual(235, num_examples)
            self.assertEqual(235, len(l))

            # now reset the iterator and check that it is the same as the first one
            self.simulator.reset(seed=5)
            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_examples=235,
            )
            l_res = list(it)
            self.assertEqual(len(l), len(l_res))
            for elem_org, elem_res in zip(l, l_res):
                for i in range(len(elem_org)):
                    self.assertAllEqual(elem_org[i], elem_res[i])

            # now do not reset the simulator and check that it works
            (it, num_examples,
             scenario_len_list) = self.simulator.create_buffered_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_examples=235,
            )
            self.assertEqual(235, num_examples)
            self.assertEqual(235, len(list(it)))

    def test_create_initial_state_estimates_data_set_iterator(self):
        behavior_policy = FirstActorFirstHeuristicController(self.particle_simulator,
                                                             actors=self.actors,
                                                             T=1 * self.T,
                                                             actor_model_dict=self.actor_dict)

        with self.subTest(name="Test with given num_scenarios"):
            self.simulator.reset(seed=5)

            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_scenarios=5,
            )
            l = list(it)
            self.assertGreater(len(l), 0)

            # check the dtypes
            (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
             t_act, controls) = l[0]
            self.assertDTypeEqual(estimated_motion_state_mean, float)
            self.assertDTypeEqual(estimated_motion_state_cov, float)
            self.assertDTypeEqual(particle_class, int)
            self.assertDTypeEqual(particle_id, int)
            self.assertDTypeEqual(existence, int)
            self.assertDTypeEqual(t_act, float)
            self.assertDTypeEqual(controls, float)

            # test that a new iterator does not have the same elements
            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_scenarios=5,
            )
            l_new = list(it)
            self.assertFalse(np.allclose(l_new[-1][0], l[-1][0]))

            # now reset the iterator and check that it is the same as the first one
            self.simulator.reset(seed=5)
            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_scenarios=5,
            )
            l_res = list(it)
            self.assertEqual(len(l), len(l_res))
            for elem_org, elem_res in zip(l, l_res):
                for i in range(len(elem_org)):
                    self.assertAllEqual(elem_org[i], elem_res[i])

        with self.subTest(name="Test with given num_examples"):
            self.simulator.reset(seed=5)

            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_examples=235,
            )
            l = list(it)
            self.assertEqual(235, len(l))

            # now reset the iterator and check that it is the same as the first one
            self.simulator.reset(seed=5)
            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_examples=235,
            )
            l_res = list(it)
            self.assertEqual(len(l), len(l_res))
            for elem_org, elem_res in zip(l, l_res):
                for i in range(len(elem_org)):
                    self.assertAllEqual(elem_org[i], elem_res[i])

            # now do not reset the simulator and check that it works
            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
                num_examples=235,
            )
            self.assertEqual(235, len(list(it)))

        with self.subTest(name="Test with no given num_scenarios or num_examples"):
            self.simulator.reset(seed=5)

            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
            )
            l = []
            for i, elem in enumerate(it):
                if i >= 200:
                    break
                l.append(elem)
            self.assertEqual(200, len(l))

            # now reset the iterator and check that it is the same as the first one
            self.simulator.reset(seed=5)
            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
            )
            l_res = []
            for i, elem in enumerate(it):
                if i >= 200:
                    break
                l_res.append(elem)
            self.assertEqual(len(l), len(l_res))
            for elem_org, elem_res in zip(l, l_res):
                for i in range(len(elem_org)):
                    self.assertAllEqual(elem_org[i], elem_res[i])

            # now do not reset the simulator and check that it works
            it = self.simulator.create_initial_state_estimates_data_set_iterator(
                behavior_policy,
                num_particles_per_scenario=19,
                scenario_time_out=self.time_out,
            )
            for i, elem in enumerate(it):
                if i >= 200:
                    break


if __name__ == "__main__":
    tf.test.main()
