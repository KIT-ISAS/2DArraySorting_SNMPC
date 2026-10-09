import os

from absl import logging

from abc import ABC, abstractmethod

import time

import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from actor_model import ActorModelVariantA, ActorStatus
from plot_utils import flush_figure_generator


class PredefinedFunctions:
    """Predefined functions that can be called for particle simulation.

    Currently, only one type of function, namely a ramp function, is supported.
    """

    def __init__(self):
        """Initializes the predefined functions."""
        # for __call__
        self._functions_dict = {'ramp': self._generate_ramp}

    def __call__(self):
        """Returns a dict of key-function-pairs containing all predefined functions.

        :returns: A dict of key-function-pairs containing all predefined functions.
        """
        return self._functions_dict

    def __contains__(self, item):
        return item in self._functions_dict.keys()

    def __getitem__(self, item):
        if not item in self:
            raise ValueError('{} not in the list of supported functions. Choose on of {}.'.format(
                item, self._functions_dict.keys()))
        return self._functions_dict[item]

    @staticmethod
    def _generate_ramp(low, high, start=None, end=None):
        """Generates a function of the time step that ramps up from low to high.

                  /\
                   |                  _____  high
                   |                /
                   |              /
                   |  low ______/
                   -------------|----|-------------> time step
                              start end

        :param low: A list or a np.array of arbitrary shape representing the low level of the ramp.
        :param high: A list or a np.array of arbitrary shape representing the high level of the ramp.
        :param start: None or an integer, the time step when to start the ramp.
        :param end: None or an integer, the time step when to stop the ramp.

        :returns: A function that describes the ramp. The function output is a np.array.
        """
        low = np.asarray(low)
        high = np.asarray(high)

        def ramp_func(k):
            if k < start:
                param = low
            elif k < end:
                param = low + (k - start) * (high - low) / (end - start)
            else:
                param = high
            return param

        return ramp_func

class ActorSimulator(ActorModelVariantA):
    """A simulator for the actors in discrete time. Builds upon ActorModelVariantA."""

    def __init__(self,
                 pos,
                 length,
                 T,
                 t_activate,
                 t_up,
                 t_hit,
                 t_down,
                 t_reset,
                 width=None,
                 init_t_act=0.0,
                 init_k=0,
                 suppress_warnings=False,
                 ):
        """Initializes the simulator.

        :param pos: A np.array of shape [num_actors, 2] representing the positions of the actors. Each position is
            modeled as [position_x, position_y], with the x-axis pointing in the direction of particle movement and the
            y-axis being orthogonal to the direction of particle movement. Note that this needs to be a sorted list
            array from smallest x-position to largest x-position. Actors with the same x-position need to be sorted from
            smallest y-position to largest y-position.
        :param length: A float or a np.array of shape [num_actors] representing the lengths (operating ranges in
            x-direction) of the actors.
        :param T: A float representing the time interval between two consecutive time steps.
        :param t_activate: A float, the time interval after an activation until the actor reaches the surface.
        :param t_up: A float, the time interval above the surface until reaching the optimal hitting time. The particle
            is not ejected but disturbed or only ejected with a reduced probability when hit during this time period.
        :param t_hit: A float, the time interval when a hitting is optimal, i.e., the particles is ejected.
        :param t_down: A float, the time interval above the surface after t_hit until reaching the surface again.
            The particle is not ejected but disturbed or only ejected with a reduced probability when hit during this
            time period.
        :param t_reset: A float, the reset time. The time interval after an activation before a new activation is
            possible.
        :param width: None or a float or a np.array of shape [num_actors] representing the with (operating ranges in
            y-direction) of the actors. If None, an 1D optical sorter (groove sorter) is assumed. If given, a 2D
            optical sorter (belt or chute) is assumed.
        :param init_t_act: A float or a np.array of shape [num_actors] representing the initial actors' internal time
            at time step k.
        :param init_k: An integer, the initial time step.
        :param suppress_warnings: A Boolean, if True, warnings if an actor is activated when it is not in a READY status
            are suppressed. If False, warnings are shown.
        """
        if not np.isscalar(init_t_act) and len(pos) != len(np.asarray(init_t_act)):
            raise ValueError('If init_t_act is an array it must have num_actors elements')
        if not isinstance(init_k, int) or init_k < 0:
            raise ValueError('The initial time step must be must be an integer greater than or equal to 0.')

        super().__init__(pos=pos,
                         length=length,
                         T=T,
                         t_activate=t_activate,
                         t_up=t_up,
                         t_hit=t_hit,
                         t_down=t_down,
                         t_reset=t_reset,
                         width=width,
                         )

        self.suppress_warnings = suppress_warnings

        # initialize the arrays/states
        self._t_act = init_t_act * np.ones(pos.shape[0]) if np.isscalar(
            init_t_act) else np.asarray(init_t_act)  # np.array of length num_actors
        self._s_act = self.output_actor_status(self._t_act)
        self._k = init_k

        # save the initial values for later reuse
        self._init_t_act = self._t_act.copy()
        self._init_k = init_k

    @property
    def t_act(self):
        """The internal time of the actors.

        :returns: A np.array of shape [num_actors] representing the actors' current internal time.
        """
        return self._t_act

    @property
    def s_act(self):
        """The status of the actors.

        :returns: An object np.array of shape [num_actors] representing the actors' current status.
        """
        return self._s_act

    @property
    def initial_time_step(self):
        """The initial time step.

        :returns: An integer representing the initial time step.
        """
        return self._init_k

    def predict_own_actor_state(self, u_act):
        """Predicts the actors' internal times and status for the next time step.

        This function predicts the state managed by the ActorSimulator class.

        :param u_act: A Boolean np.array of shape [num_actors] representing the control inputs u_k at time step k, where
            True indicates that the actor should be activated in the current time step.
        """
        invalid_activation = np.logical_and(u_act,
                                            np.logical_not(np.logical_or(self._s_act == ActorStatus.READY,
                                                                         np.logical_and(
                                                                             self._s_act == ActorStatus.RESET,
                                                                             self._t_act == self.t_cycle))))

        assert np.logical_not(np.any(np.logical_and(self._s_act == ActorStatus.RESET, self.t_act == 0.0))), (
            'Actor in RESET position with t_act = 0. '
            'This indicates that your controller is accidentally manipulating the actor states.')

        if not self.suppress_warnings and np.any(invalid_activation):
            logging.warning(
                'Actor at index {} can only be activated if it is in ready position.'.format(
                    np.nonzero(invalid_activation)[0]))

        self._k += 1
        self._t_act, self._s_act = self.predict_actor_state(self._t_act, u_act)

    def reset(self):
        """Resets the actors to the initial state."""
        self._t_act = self._init_t_act.copy()
        self._s_act = self.output_actor_status(self._t_act)
        self._k = self._init_k

    def shallow_copy(self):
        """Creates a shallow copy of the actor simulator, i.e., a new instance of the actor simulator with the same
        settings but independent, freshly-initialized states.

        :returns: An ActorSimulator instance, a shallow copy of the actor simulator.
        """
        return ActorSimulator(pos=self._pos,
                              length=self._length,
                              T=self._T,
                              t_activate=self._t_activate,
                              t_up=self._t_up,
                              t_hit=self._t_hit,
                              t_down=self._t_down,
                              t_reset=self._t_reset,
                              width=self._width,
                              init_t_act=self._init_t_act,
                              init_k=self._init_k,
                              suppress_warnings=self.suppress_warnings,
                              )


class AbstractParticleSimulator(ABC):
    """Abstract class for discrete-time simulators of the particles on a sorting machine, either on the groove (1D
    movement) or on a belt or chute (2D movement), including tracks births and deaths.

    Internally uses constant-velocity (CV) and/or constant-acceleration motion models for simulation of the particle
    movement.

    The class supports particle-specific and time-variant motion models, i.e., the motion models can be different for
    each particle class and can change over time. This can be used to model different materials (particle types) on the
    sorting machine as well as changing conditions over time.
    """

    def __init__(self,
                 total_num_particles,
                 T,
                 process_parameters=({"type": "CV"},
                                     {"type": "CV"}),
                 init_k=0,
                 seed=None,
                 freeze_scenario=False,
                 S_v=0.0,
                 ):
        """Initializes the particle simulator.

        :param total_num_particles: An integer, the total number of particles to generate during the simulation.
        :param T: A float representing the time interval between two consecutive time steps.
        :param process_parameters: A tuple of length num_processes containing dicts, where each dict has at least
            a key 'type' describing the type of the process, e.g., a CV-model, and the remaining fields contain the
            parameters of the process. Each process may describe a material in the mass flow. Note that time-variant
            process parameters are supported. See self._define_processes for a detailed documentation of supported
            types.
        :param init_k: An integer, the initial time step.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        :param freeze_scenario: A Boolean, if True, the scenario, including the total number of particles and the seed
            for the random number generator are fixed to the initial value, i.e., each time after resetting the
            simulator, the same settings are restored and the scenario is generated. If False, the scenario is not
            frozen and a new run after resetting is allowed to be different from the ones before.
        :param S_v: A float or a sequence of two floats, the measurement-noise variance used by measure_particles(...).
            A float is used for 1D simulators; a sequence of length 2 for 2D simulators (x- and y-direction). Defaults
            to 0.0 (noise-free measurements).
        """
        if not isinstance(total_num_particles, int) or total_num_particles <= 0:
            raise ValueError(
                'The total number of particles to generate total_num_particles must be a positive integer.')
        if not np.isscalar(T) or T <= 0:
            raise ValueError('The time interval between two consecutive time steps T must be a positive scalar.')
        if not isinstance(init_k, int) or init_k < 0:
            raise ValueError('The initial time step must be must be an integer greater than or equal to 0.')
        if freeze_scenario and seed is None:
            raise ValueError('If freeze_scenario is True, a seed for the random number generator must be given.')
        if np.any(np.asarray(S_v) < 0):
            raise ValueError('The measurement-noise variance S_v must be greater than or equal to zero.')

        self._total_num_particles = total_num_particles
        self._T = float(T)
        self._S_v = S_v

        # get the predefined functions
        self._predefined_functions = PredefinedFunctions()

        # define the processes
        self._process_parameters = len(process_parameters) * [{}]
        self._process_type = len(process_parameters) * [None]
        self._define_processes(process_parameters)  # will manipulate self._process_parameters and self._process_type

        # set the seed and create the random number generators, we need two generators, one for the particle motion
        # prediction and one for the particle initialization to make sure that particle initialization is independent of
        # the subsequent particle motion and thus the controller and therefore ensure reproducibility of the scenarios
        self._rng_for_particle_prediction = np.random.default_rng(seed)
        self._seed_offset = 456789123
        self._rng_for_particle_initialization = np.random.default_rng(
            seed + self._seed_offset if seed is not None else None)
        self._freeze_scenario = freeze_scenario

        # initialize the arrays/states
        self._particle_id = np.empty(0, dtype=int)
        self._particle_class = np.empty(0, dtype=int)
        self._sampled_motion_states = [np.empty((0, process_param["F"](0).shape[0]),
                                                dtype=float) for process_param in self._process_parameters]
        self._particle_id_in_motion_states = [np.empty(0, dtype=int) for _ in self._process_parameters]
        self._existence = np.empty(0, dtype=bool)
        self._counter = 0

        # internal time step for simulation
        self._k = init_k

        # save the initial values for later reuse
        self._init_k = init_k
        self._original_seed = seed
        self._original_total_num_particles = total_num_particles
        self._original_process_parameters = process_parameters

    @property
    @abstractmethod
    def C_w_pos_velo(self):
        """The system noise covariance for the (position, velocity)-components of the particles.

        Note that the acceleration components in case of CA models are not considered here.

        :returns: A np.array of shape [num_particle, ...], the system noise covariance of the particles.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def T(self):
        """The sampling time difference.

        :returns: A float representing the time interval between two consecutive time steps.
        """
        return self._T

    @property
    def position(self):
        """The current positions of the particles.

        :returns: A np.array of shape [num_particle, ...], the positions of the particles.
        """
        flattened_positions = np.concatenate(
            [self._state_to_pos(motion_state) for motion_state in self._sampled_motion_states], axis=0)
        return self._associate_to_particle_id(flattened_positions)

    @property
    def velocity(self):
        """The current velocities of the particles.

        :returns: A np.array of shape [num_particle, ...], the current velocities of the particles.
        """
        flattened_velocities = np.concatenate(
            [self._state_to_velo(motion_state) for motion_state in self._sampled_motion_states], axis=0)
        return self._associate_to_particle_id(flattened_velocities)

    @property
    def particle_existence(self):
        """The existence of the particles.

        :returns: A Boolean np.array of shape [num_particles], the existence of the particles. True indicates that a
            particle exists.
        """
        return self._existence

    @property
    def particle_class(self):
        """The class of the particles.

        :returns: An integer np.array of shape [num_particles] where each component is either 0 ("keep
            the particle") or 1 ("eject the particle") representing the current particle classes.
        """
        return self._particle_class

    @property
    def particle_id(self):
        """The particle IDs.

        :returns: An integer np.array of shape [num_particles], the ids of the particles.
        """
        return self._particle_id

    @property
    def total_num_particles(self):
        """The total number of particles to generate during the simulation.

        :returns: An integer representing the total number of particles to generate during the simulation.
        """
        return self._total_num_particles

    @total_num_particles.setter
    def total_num_particles(self, value):
        """Resets the particle simulator and sets a new total number of particles.

        This option is only available if the scenario is not frozen. If additionally a new seed should be set, call
        reset(seed=<new_seed>) before or afterward.

        :param value: An integer representing the total number of particles to generate during the simulation.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                'The total number of particles to generate total_num_particles must be a positive integer.')
        if self._freeze_scenario:
            raise ValueError('The scenario is frozen, the total number of particles cannot be changed.')

        self._total_num_particles = value
        self.reset(seed=None, restore_original_scenario=False)  # do not change the seed; restore original scenario
        # does not make sense here

    @property
    def initial_time_step(self):
        """The initial time step.

        :returns: An integer representing the initial time step.
        """
        return self._init_k

    @property
    def is_frozen(self):
        """Whether the scenario is frozen.

        :returns: A Boolean, whether the scenario is frozen.
        """
        return self._freeze_scenario

    @property
    def num_particle_seen_so_far(self):
        """The number of already generated particles.

        Can be used to check/assert against self.total_num_particles to verify the implementation.

        :returns: An integer representing the number of already generated particles.
        """
        return self._counter

    @staticmethod
    @abstractmethod
    def _state_to_pos(motion_state):
        """Transforms the motion state to the position of the particles.

        :param motion_state: A np.array of shape [num_particles, state_length], the motion state.

        :returns: A np.array of shape [num_particle, ...], the current positions of the particles.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @staticmethod
    @abstractmethod
    def _state_to_velo(motion_state):
        """Transforms the motion state to the velocities of the particles.

        :param motion_state: A np.array of shape [num_particles, state_length], the motion state.

        :returns: A np.array of shape [num_particle, ...], the current velocities of the particles.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _define_cv_model(self, T, *args, **kwargs):
        """Defines a (potentially time-variant) Constant-Velocity (CV) motion model (also known als white-noise
        acceleration model).

        Note that the units of the parameters must be consistent, but are not restricted to pixels or DEM coordinate
        unit and time steps. For example, also normalized values can be used.

        :param T: A float representing the time interval between two consecutive time steps.

        :returns:
            F: A function f(time_step) -> np.array representing the (potentially time-variant) transition matrix of the
                model.
            C_w: A function f(time_step) -> np.array representing the (potentially time-variant) system noise covariance
                of the model.
            mean_0: A function f(time_step) -> np.array representing the (potentially time-variant) expected value of
                the initial state.
            cov_0: A function f(time_step) -> np.array representing the (potentially time-variant) covariance of
                 the initial state.
            S_w: A function f(time_step) -> (float, float) representing the (potentially time-variant) power spectral
                density of the system noise in x- and y-direction.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _define_ca_model(self, T, *args, **kwargs):
        """Defines a (potentially time-variant) Constant-Velocity (CV) motion model (also known als white-noise
        jerk model).

        Note that the units of the parameters must be consistent, but are not restricted to pixels or DEM coordinate
        unit and time steps. For example, also normalized values can be used.

        :param T: A float representing the time interval between two consecutive time steps.

        :returns:
            F: A function f(time_step) -> np.array representing the (potentially time-variant) transition matrix of the
                model.
            C_w: A function f(time_step) -> np.array representing the (potentially time-variant) system noise covariance
                of the model.
            mean_0: A function f(time_step) -> np.array representing the (potentially time-variant) expected value of
                the initial state.
            cov_0: A function f(time_step) -> np.array representing the (potentially time-variant) covariance of
                 the initial state.
            S_w: A function f(time_step) -> (float, float) representing the (potentially time-variant) power spectral
                density of the system noise in x- and y-direction.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def create_particles(self):
        """Creates new particles in the current time step, i.e., initializes new tracks."""
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def track_particles(self, measurements):
        """This is a dummy/mock for a tracker for the particles and constitutes the (only) interface for data exchange
        to the controllers.

        Instead of really tracking the particle, we assume here that we measure the mean of the motion state perfectly
        and add an artificial noise for the motion states' covariance.

        This mocks outputs the motion state mean and covariance as for a constant velocity model, even if internally
        a constant acceleration model is used. This is to be consistent with the controllers, that currently
        only support constant velocity motion states.

         Format measurements:

            [x, label]

            with label either 0 ("keep the particle") or 1 ("eject the particle").

        :param measurements: A np.array of shape [num_measurements, measurement_vector_length], the measurements
            (including x-position and class label) for the current time step. Used as dummy input here.

        :returns:
             estimated_motion_state_mean: A np.array of shape [num_particles, state_length], the estimated motion state
                mean.
             estimated_motion_state_cov: A np.array of shape [num_particles, state_length, state_length], the estimated
                motion state covariance.
             particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
                particle") or 1 ("eject the particle") representing the particle classes.
             particle_id: An integer np.array of shape [num_particles], the ids of the particles.
             existence: A Boolean np.array of shape [num_particles], the existence of the particles. True indicates that
                a particle exists.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def measure_particles(self):
        """Returns noisy position measurements of currently existing particles for a real multitarget tracker.

        Unlike track_particles(...), this method does not return motion-state estimates. It only provides position
        measurements (plus class labels) that can be passed to a callable MTT tracker.

         Format measurements (1D):

            [x, label]

         Format measurements (2D):

            [x, y, label]

            with label either 0 ("keep the particle") or 1 ("eject the particle").

        Additive Gaussian measurement noise with variance S_v is applied to the position components. Existing particles
        only are included.

        :returns: A np.array of shape [num_existing_particles, measurement_vector_length], the measurements for the
            current time step. If no particle exists, an empty array with the appropriate second dimension is returned.
        """
        existence = self.particle_existence
        position = self.position[existence]
        particle_class = self.particle_class[existence]

        if position.ndim == 1:
            # 1D groove sorter
            if not np.isscalar(self._S_v):
                raise ValueError('For 1D particle simulators, S_v must be a scalar.')
            if position.shape[0] == 0:
                return np.empty((0, 2))
            noise = self._rng_for_particle_prediction.normal(
                loc=0.0, scale=np.sqrt(self._S_v), size=position.shape)
            return np.column_stack((position + noise, particle_class))

        # 2D area sorter
        s_v = np.asarray(self._S_v, dtype=float)
        if s_v.size == 1:
            s_v = np.repeat(s_v, 2)
        if s_v.size != 2:
            raise ValueError('For 2D particle simulators, S_v must be a scalar or a sequence of length 2.')
        if position.shape[0] == 0:
            return np.empty((0, 3))
        noise = self._rng_for_particle_prediction.normal(loc=0.0, scale=1.0, size=position.shape)
        noise *= np.sqrt(s_v)
        return np.column_stack((position + noise, particle_class))

    @abstractmethod
    def _shallow_copy(self, seed=None, freeze_scenario=False):
        """Creates a shallow copy of the particle simulator, i.e., a new instance of the particle simulator with the
        same settings but independent, freshly-initialized states.

        Note that the copy will be initialized with the current self.total_num_particles particles, i.e., after
        resetting the copy with reset(restore_original_scenario=True), the copy will by default work with
        self.total_num_particles particles.

        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        :param freeze_scenario: A Boolean, if True, the scenario, including the total number of particles and the seed
            for the random number generator are fixed to the initial value, i.e., each time after resetting the
            simulator, the same settings are restored and the scenario is generated. If False, the scenario is not
            frozen and a new run after resetting is allowed to be different from the ones before.

        :returns: An AbstractParticleSimulator instance, a shallow copy of the particle simulator.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def _define_processes(self, process_parameters_ls):
        """Defines the (potentially time-variant) processes to be used for data generation, given a list of dicts
        describing the processes.

        Format dicts:

          {'type': process_type,
            ... further key-value pairs}],

          with process_type a string indicating the type of the process and further key-value pairs describing fields
          for its parameters.

        Processes can be defined in two ways (we have two process types):

             i) CV-model (process_type = 'CV'):
                    Fields with the keys
                        - S_w, pos_0, v_0,
                        - and for wo-dimensional motion additionally: pos0_stddev, v0_stddev
             ii) CA-model (process_type = 'CA'):
                    Fields with the keys
                        - S_w, S_v, pos_0, v_0, a_0,
                        - and for wo-dimensional motion additionally: pos0_stddev, v0_stddev, a0_stddev

        For the value of each of the keys either
            * a float, if one-dimensional motion, or
                a tuple of floats with length 2, if two-dimensional motion,
            * a function f(times) --> float, if one-dimensional motion, or
                f(time_step) -> (float, float), if two-dimensional motion,
            * a string representing a lambda function f(time_step) -> (float), if one-dimensional motion, or
                f(time_step) -> (float, float), if two-dimensional motion,
            * a string representing a function call (with args and/or kwargs)) of predefined
                function f(time_step) -> float, if one-dimensional motion, or
                    f(time_step) -> (float, float), if two-dimensional motion,
        can be passed. If a key is not given, default values will be used.

        Please refer to _define_cv_model, and _define_ca_model for further documentation.

         Example (for one-dimensional motion):

            process_parameters_ls = [
                 {'type': 'CV'},
                 {'type': 'CA', 'S_w': 600.0, 'v_0': 0.0},
                 {'type': 'CV', 'S_w': lambda k: 2 * k + 400.0},
                 {'type': 'CV', 'S_w': 'lambda k: 2 * k + 400.0'},
                 {'type': 'CV', 'S_w': 'ramp(400, 40000)'},
                 ]

         Example (for two-dimensional motion):

            process_parameters_ls = [
                 {'type': 'CV'},
                 {'type': 'CA', 'S_w': (600.0, 400.0) , 'v_0': (1.0, 0.0)},
                 {'type': 'CV', 'S_w': lambda k: 2 * k + 400.0, 200},
                 {'type': 'CV', 'S_w': 'lambda k: 2 * k + 400.0, 200'},
                 {'type': 'CV', 'S_w': 'ramp([400, 400], [40000, 400])'},
                 ]

        :param process_parameters_ls: A list of length num_processes containing dicts, where each dict has at least a
            key 'type' describing the type of the process, e.g., a CV-model, and the remaining fields contain the
            parameters of the process. Valid values for the key 'type' are 'CV' or 'CA'. The further fields (if given)
            may have key-values pairs as explained above.
        """
        for process_no, process_dict in enumerate(process_parameters_ls):
            dict = process_dict.copy()
            process_type = dict.pop('type')

            self._process_parameters[process_no] = {}

            if process_type == "CV":
                F, C_w, mean_0, cov_0, S_w = self._define_cv_model(T=self._T, **dict)
                self._process_parameters[process_no]["S_w"] = S_w
            elif process_type == "CA":
                F, C_w, mean_0, cov_0, S_w = self._define_ca_model(T=self._T, **dict)
                self._process_parameters[process_no]["S_w"] = S_w
            else:
                raise ValueError(
                    "Type {} not defined for process_parameters_ls. Key must be be one of ['CV', 'CA']".format(
                        process_type))

            self._process_parameters[process_no]["F"], self._process_parameters[process_no]["C_w"], \
                self._process_parameters[process_no]["mean_0"], self._process_parameters[process_no][
                "cov_0"] = F, C_w, mean_0, cov_0
            self._process_type[process_no] = process_type

    def _define_func(self, value):
        """Wraps the value into a function of desired format, if not already done.

        Value can be passed in five ways:

                i) as a static float or int.
               ii) as a (static) tuple of floats or integers,
              iii) as function f(time_step) -> (float/int, float/int), or f(time_step) -> float/int,
               iv) as a string representing a lambda function f(time_step) -> (float/int, float/int),
                    or f(time_step) -> float/int,
                v) as a string representing a function call (with args and/or kwargs) of predefined function
                    f(time_step) -> (float/int, float/int), or f(time_step) -> float/int,

        Examples:

            value = 1.0
            value = (1.0, 2.0)
            value = lambda k: (1, 2*k) or lambda k: 2*k,
            value = 'lambda k: (1, 2*k)' or 'lambda k: 2*k',
            value = 'ramp((0, 1), (1, 1), start=0, end=1)', or 'ramp(1, start=0, end=1)'

        :param value: A value to be wrapped into a function. Type and format as described above.

        :returns: A function f(time_step) -> np.array of shape [1 or 2].
        """
        if not callable(value):
            if isinstance(value, str):
                if value[:6] == "lambda":
                    fn = eval(value)
                    func = lambda k: np.asarray(fn(k))
                else:
                    func_name = value.rsplit('(')[0]
                    if func_name in self._predefined_functions:
                        func = eval(value, {'__builtins__': None}, self._predefined_functions())
                    else:
                        raise ValueError(
                            'Function {} is not a predefined function of simulator.py.'.format(func_name))
            elif isinstance(value, (list, tuple, np.ndarray)):
                if len(value) != 1 and len(value) != 2:
                    raise ValueError(
                        'If using is a list or tuple for parameter specification, it needs to have a length of 1 or 2.')
                func = lambda k: np.array([*value])
            elif isinstance(value, (float, int)):
                func = lambda k: np.array([value])
            else:
                raise ValueError(
                    'For parameter specification, pass either a scalar, list, tuple, a function, or a function call of '
                    'a predefined function, not {}.'.format(value))
        else:
            func = lambda k: np.asarray(value(k))

        if func(0).ndim != 1 and len(func(0)) != 2:
            raise ValueError(
                'If using a function for parameter specification, its return value needs to be transferable to a '
                'one-dimensional np.array of length 1 or 2.')

        return func

    def move_particles(self):
        """Moves the particles, i.e., predicts the next state of the particles according to the defined motion models.

        Note that this function supports particle-specific and time-variant motion models.
        """
        for process_no, process_param in enumerate(self._process_parameters):
            F, C_w = process_param["F"], process_param["C_w"]
            self._sampled_motion_states[process_no] = np.matmul(F(self._k),
                                                                self._sampled_motion_states[process_no][:, :,
                                                                None]).squeeze(-1)
            w_t = self._rng_for_particle_prediction.multivariate_normal(mean=np.zeros(len(C_w(self._k))),
                                                                        cov=C_w(self._k),
                                                                        size=len(
                                                                            self._sampled_motion_states[process_no]))
            self._sampled_motion_states[process_no] += w_t

        self._k += 1

    def remove_particles(self, remove_mask):
        """Removes particles from the simulation.

        A particle is typically removed if it is ejected from the sorting machine or if it leaves its operating area.

        :param remove_mask: A Boolean np.array of shape [num_particles] used as a mask to remove particles. True
            indicates that a particle should be removed.
        """
        particles_to_keep = np.logical_not(remove_mask)
        self._particle_id = self._particle_id[particles_to_keep]
        for process_no, particle_id_from_process in enumerate(self._particle_id_in_motion_states):
            still_in_mask = np.isin(particle_id_from_process, self._particle_id)
            self._sampled_motion_states[process_no] = self._sampled_motion_states[process_no][still_in_mask]
            self._particle_id_in_motion_states[process_no] = particle_id_from_process[still_in_mask]
        self._existence = self._existence[particles_to_keep]
        self._particle_class = self._particle_class[particles_to_keep]

    def negate_existence(self, hit_mask):
        """Sets the existence of particles to False.

        A particle is typically negated if it is ejected from the sorting machine.

        :param hit_mask: A Boolean np.array of shape [num_particles] used as a mask to negate the existence of
            particles. True indicates that a particle should be negated.
        """
        self._existence[hit_mask] = False

    def reset(self, seed=None, restore_original_scenario=False):
        """Resets the particle simulator to the initial state.

        :param seed: None or an integer, the seed for the random number generator. If given, the random number
            generator is reset to seed. If None, the random number generator is not reset. This option is only available
            if restore_original_scenario and self._freeze_scenario are False.
        :param restore_original_scenario: A Boolean, if True, the scenario, including the total number of particles and
            the seed for the random number generator are reset to the initial value, i.e., the run after resetting the
            simulator will be the same as the first run. If False and self._freeze_scenario is False, the scenario is
            not fixed and the run will be different from the ones before. This option is only available if seed is None.
        """
        if seed is not None and (restore_original_scenario or self._freeze_scenario):
            raise ValueError('The seed cannot be set if restore_original_scenario or self._freeze_scenario is True.')

        # initialize the arrays/states
        self._particle_id = np.empty(0, dtype=int)
        self._particle_class = np.empty(0, dtype=int)
        self._sampled_motion_states = [np.empty((0, process_param['F'](0).shape[0]), dtype=float) for process_param in
                                       self._process_parameters]
        self._particle_id_in_motion_states = [np.empty(0, dtype=int) for _ in self._process_parameters]
        self._existence = np.empty(0, dtype=bool)

        self._counter = 0
        self._k = self._init_k

        if seed is not None:
            self._rng_for_particle_prediction = np.random.default_rng(seed)
            self._rng_for_particle_initialization = np.random.default_rng(seed + self._seed_offset)
        elif restore_original_scenario or self._freeze_scenario:
            # use the same seed as the first, run will be reproducible
            if self._original_seed is None:
                raise ValueError('The original seed is None, the scenario cannot be restored to the original one.')
            self._rng_for_particle_prediction = np.random.default_rng(self._original_seed)
            self._rng_for_particle_initialization = np.random.default_rng(self._original_seed + self._seed_offset)
            self._total_num_particles = self._original_total_num_particles

    def shallow_copy(self, seed=None, freeze_scenario=False):
        """Creates a shallow copy of the particle simulator, i.e., a new instance of the particle simulator with the
        same settings but independent, freshly-initialized states.

        Note that the copy will be initialized with the current self.total_num_particles particles, i.e., after
        resetting the copy with reset(restore_original_scenario=True), the copy will by default work with
        self.total_num_particles particles.

        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        :param freeze_scenario: A Boolean, if True, the scenario, including the total number of particles and the seed
            for the random number generator are fixed to the initial value, i.e., each time after resetting the
            simulator, the same settings are restored and the scenario is generated. If False, the scenario is not
            frozen and a new run after resetting is allowed to be different from the ones before.

        :returns: An AbstractParticleSimulator instance, a shallow copy of the particle simulator.
        """
        return self._shallow_copy(seed=self._original_seed if freeze_scenario and seed is None else seed,
                                  freeze_scenario=freeze_scenario)

    def _associate_to_particle_id(self, flattened_array):
        """Associates the flattened array, indexed with self._particle_id_in_motion_states, to the self._particle_id.

         Example:

            self._particle_id = [1, 2, 3] , self._particle_id_in_motion_states = [[3, 2], [1]]

                                       [[0, 0, 1]
            --> association_matrix =    [0, 1, 0]     in particular, association_matrix @ [3, 2, 1] = [1, 2, 3]
                                        [1, 0, 0]]

        :param flattened_array: A np.array of shape [num_particles, ...] to be associated to the self._particle_id.

        :returns: A np.array of shape [num_particles, ...] with the same shape as flattened_array, but associated to
            self._particle_id.
        """
        flattened_ids = np.hstack([particle_id for particle_id in self._particle_id_in_motion_states])
        association_matrix = np.equal(self._particle_id[:, None], flattened_ids[None, :]).astype(int)
        return association_matrix @ flattened_array


class AreaParticleSimulator(AbstractParticleSimulator):
    """A discrete-time simulator for the particles on a belt or chute (2D movement), including tracks births and deaths.
.
    Internally uses constant-velocity (CV) and/or constant-acceleration motion models for simulation of the particle
    movement.

    The class supports particle-specific and time-variant motion models, i.e., the motion models can be different for
    each particle class and can change over time. This can be used to model different materials (particle types) on the
    sorting machine as well as changing conditions over time.

    This class uses a process/motion model specific birth_rate_mean_and_stddev to define the number of particles of each
    process that spawn in each time step. The processes are mapped to the eject and the accept class via class_labels.
    Therefore, it is possible to defined arbitrary many processes/motion models, as long as there is at least one for
    the reject and one for the accept class). In contrast to a 1D groove sorter, it is not possible to
    predefine the class of each particle. The ratio of accept and eject particles depends on the
    birth_rate_mean_and_stddev of all processes. The simulation terminates, if total_num_particles particles have left
    the sorting machine.
    """

    def __init__(self,
                 total_num_particles,
                 T,
                 process_parameters,
                 birth_rate_mean_and_stddev,
                 class_labels,
                 area_width,
                 particle_r=0.5,
                 init_k=0,
                 seed=None,
                 freeze_scenario=False,
                 S_v=(0.0, 0.0),
                 ):
        """Initializes the particle simulator.

        Note that process parameters should be such that there is enough probability mass to spawn particles in
        [particle_r and area_width - particle_r], orthogonal to the transport direction and [-particle_x_velocity * T,
        0] along the transport direction (note that the field of view of the simulation is [0, array_end] x
        [0, area_width], spawning particles at x < 0 therefore is visually more appealing).

        :param total_num_particles: An integer, the total number of particles to generate during the simulation.
        :param T: A float representing the time interval between two consecutive time steps.
        :param process_parameters: A tuple of length 2 containing dicts, where each dict has at least
            a key 'type' describing the type of the process, e.g., a CV-model, and the remaining fields contain the
            parameters of the process. Each process may describe a material in the mass flow. Note that time-variant
            process parameters are supported. See self._define_processes for a detailed documentation of supported
            types.
        :param birth_rate_mean_and_stddev: A list of length num_processes containing (mean, stddev)-tuples describing
            the mass flows, i.e., the number of track births per time steps. Note that time-variant mean and stddevs are
            supported. See self._define_mass_flows for a detailed documentation of supported types.
        :param class_labels: A list of length num_processes where each component is either 0 ("keep the particle") or
            1 ("eject the particle") representing the particle class for each process.
        :param area_width: A float representing the total width of the belt or chute.
        :param particle_r: A float representing the radius of the simulated particles.
        :param init_k: An integer, the initial time step.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        :param freeze_scenario: A Boolean, if True, the scenario, including the total number of particles and the seed
            for the random number generator are fixed to the initial value, i.e., each time after resetting the
            simulator, the same settings are restored and the scenario is generated. If False, the scenario is not
            frozen and a new run after resetting is allowed to be different from the ones before.
        :param S_v: A float or a sequence of two floats, the measurement-noise variance used by measure_particles(...)
            in x- and y-direction. Defaults to (0.0, 0.0) (noise-free measurements).
        """
        if not np.isscalar(area_width) or area_width <= 0:
            raise ValueError('The width of the belt or chute area_width must be a positive scalar.')
        if not np.isscalar(particle_r) or particle_r <= 0:
            raise ValueError('The particle radius particle_r must be a positive scalar.')

        if len(birth_rate_mean_and_stddev) != len(process_parameters) or len(birth_rate_mean_and_stddev) != len(
                class_labels):
            raise ValueError(
                'The process_parameters, birth_rate_mean_and_stddev, and class_labels must be of same length.')
        if np.all(np.unique(class_labels) != np.array([0, 1])):
            raise ValueError('The class labels must be either 0 or 1 and both labels must be present.')
        s_v = np.asarray(S_v, dtype=float)
        if s_v.size == 1:
            S_v = (float(s_v), float(s_v))
        elif s_v.size == 2:
            S_v = (float(s_v[0]), float(s_v[1]))
        else:
            raise ValueError('For AreaParticleSimulator, S_v must be a scalar or a sequence of length 2.')

        self._area_width = area_width
        self._particle_r = particle_r

        super().__init__(total_num_particles=total_num_particles,
                         T=T,
                         process_parameters=process_parameters,
                         init_k=init_k,
                         seed=seed,
                         freeze_scenario=freeze_scenario,
                         S_v=S_v)

        # define the mass flows
        self._birth_rate_mean_and_stddev = len(birth_rate_mean_and_stddev) * [None]
        self._define_mass_flows(birth_rate_mean_and_stddev)  # will manipulate self._birth_rate_mean_and_stddev
        self._class_labels = class_labels

        # save the values for later reuse
        self._original_birth_rate_mean_and_stddev = self._birth_rate_mean_and_stddev.copy()

    @property
    def C_w_pos_velo(self):
        """The system noise covariance for the (position, velocity)-components of the particles.

        Note that the acceleration components in case of CA models are not considered here.

        :returns: A np.array of shape [num_particles, 4, 4], the system noise covariance of the particles.
        """
        C_w = np.empty((len(self._particle_id), 4, 4), dtype=float)
        i = 0
        for process_type, process_parameters, motion_state in zip(self._process_type, self._process_parameters,
                                                                  self._sampled_motion_states):
            C_w_process = process_parameters["C_w"](self._k)
            if np.all(C_w_process == 0):
                if process_type == "CV":
                    C_w_process = 0.00002 * np.array([[self._T ** 3 / 3, self._T ** 2 / 2, 0, 0],
                                                      [self._T ** 2 / 2, self._T, 0, 0],
                                                      [0, 0, self._T ** 3 / 3, self._T ** 2 / 2],
                                                      [0, 0, self._T ** 2 / 2, self._T]])
                if process_type == "CA":
                    C_w_process = 0.00002 * np.array([[self._T ** 5 / 20, self._T ** 4 / 8, self._T ** 3 / 6, 0, 0, 0],
                                                      [self._T ** 4 / 8, self._T ** 3 / 3, self._T ** 2 / 2, 0, 0, 0],
                                                      [self._T ** 3 / 6, self._T ** 2 / 2, self._T, 0, 0, 0],
                                                      [0, 0, 0, self._T ** 5 / 20, self._T ** 4 / 8, self._T ** 3 / 6],
                                                      [0, 0, 0, self._T ** 4 / 8, self._T ** 3 / 3, self._T ** 2 / 2],
                                                      [0, 0, 0, self._T ** 3 / 6, self._T ** 2 / 2, self._T]])

            if process_type == "CA":
                C_w_process = C_w_process[[0, 1, 3, 4]][:, [0, 1, 3, 4]]

            C_w[i:i + len(motion_state)] = C_w_process
            i += len(motion_state)

        return C_w

    @property
    def area_width(self):
        """The total width of the belt or chute (maximum y position).

        :returns: A float representing the total width of the belt or chute.
        """
        return self._area_width

    @property
    def particle_r(self):
        """The radius of the particles.

        :returns: A float representing the radius of the particles.
        """
        return self._particle_r

    @staticmethod
    def _state_to_pos(motion_state):
        return motion_state[:, [0, 2]] if motion_state.shape[1] == 4 else motion_state[:, [0, 3]]

    @staticmethod
    def _state_to_velo(motion_state):
        return motion_state[:, [1, 3]] if motion_state.shape[1] == 4 else motion_state[:, [1, 4]]

    def _define_cv_model(self,
                         T,
                         S_w=(8E7, 8E7),
                         # System noise power spectral density in px² / sec³, variance v = S_w * T
                         # -> Stddev. of v approx. 200 px / sec * 0.05 mm / px = 10 mm / sec,
                         pos_0=(160.0, 1160),  # px,
                         v_0=(20000, 0),  # px / sec., 100 px / frame, approx. 5 mm / frame or 1 m / sec.,
                         pos0_stddev=(80, 640),  # px, approx. [4, 16] mm
                         v0_stddev=(2000, 2000),  # px / sec., approx. 0.5 mm / frame.
                         ):
        """Defines a (potentially time-variant) Constant-Velocity (CV) motion model (also known als white-noise
        acceleration model).

         Format State:

            [x, v_x, y, v_y],

            where v_x and v_y are the velocities in x- and y-direction.

        Note that the units of the parameters must be consistent, but are not restricted to pixels or DEM coordinate
        unit and time steps. For example, also normalized values can be used.

        Apart from T, all parameters can be defined in four ways:

                i) as a (static) tuple of floats or integers,
               ii) as function f(time_step) -> (float, float),
              iii) as a string representing a lambda function f(time_step) -> (float, float), or
               iv) as a string representing a function call (with args and/or kwargs)) of predefined function
                    f(time_step) -> (float, float).

        :param T: A float representing the time interval between two consecutive time steps.
        :param S_w: A list of floats with length 2, a function f(time_step) -> (float, float) or a string representing
            either a lambda function or function call of a predefined function (with args and/or kwargs), all
            representing the power spectral density of the system noise in x- and y-direction.
        :param pos_0: A list of floats with length 2, a function f(time_step) -> (float, float) or a string representing
            either a lambda function or function call of a predefined function (with args and/or kwargs), all
            representing the initial position mean of the tracks.
        :param v_0: A list of floats with length 2, a function f(time_step) -> (float, float) or a string representing
            either a lambda function or function call of a predefined function (with args and/or kwargs), all
            representing the initial velocity mean of the tracks.
        :param pos0_stddev: A list of floats with length 2, a function f(time_step) -> (float, float) or a string
            representing either a lambda function or function call of a predefined function (with args and/or kwargs),
            all representing the initial position standard deviation of the tracks.
        :param v0_stddev: A list of floats with length 2, a function f(time_step) -> (float, float) or a string
            representing either a lambda function or function call of a predefined function (with args and/or kwargs),
            all representing the initial velocity standard deviation of the tracks.

        :returns:
            F: A function f(time_step) -> np.array representing the (potentially time-variant) transition matrix of the
                model.
            C_w: A function f(time_step) -> np.array representing the (potentially time-variant) system noise covariance
                of the model.
            mean_0: A function f(time_step) -> np.array representing the (potentially time-variant) expected value of
                the initial state.
            cov_0: A function f(time_step) -> np.array representing the (potentially time-variant) covariance of
                 the initial state.
            S_w: A function f(time_step) -> (float, float) representing the (potentially time-variant) power spectral
                density of the system noise in x- and y-direction.
        """
        # convert arguments to functions
        S_w = self._define_func(S_w)
        pos_0 = self._define_func(pos_0)
        v_0 = self._define_func(v_0)
        pos0_stddev = self._define_func(pos0_stddev)
        v0_stddev = self._define_func(v0_stddev)

        # some checks on the input
        if S_w(0).shape != (2,):
            raise ValueError('The power spectral density S_w must be of length 2.')
        if pos_0(0).shape != (2,):
            raise ValueError('The initial velocity v_0 must be of length 2.')
        if v_0(0).shape != (2,):
            raise ValueError('The initial velocity v_0 must be of length 2.')
        if pos0_stddev(0).shape != (2,):
            raise ValueError('The initial velocity stddev v0_stddev must be of length 2.')
        if v0_stddev(0).shape != (2,):
            raise ValueError('The initial velocity stddev v0_stddev must be of length 2.')

        # Transition matrix
        F = lambda k: np.array([[1, T, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, T],
                                [0, 0, 0, 1]], dtype=float)

        # System noise covariance
        C_w = lambda k: np.matmul(np.array([[T ** 3 / 3, T ** 2 / 2, 0, 0],
                                            [T ** 2 / 2, T, 0, 0],
                                            [0, 0, T ** 3 / 3, T ** 2 / 2],
                                            [0, 0, T ** 2 / 2, T]], dtype=float),
                                  np.diag([S_w(k)[0], S_w(k)[0], S_w(k)[1], S_w(k)[1]]))

        # Initial values
        mean_0 = lambda k: np.array([pos_0(k)[0], v_0(k)[0], pos_0(k)[1], v_0(k)[1]], dtype=float)
        cov_0 = lambda k: np.diag(
            np.array([pos0_stddev(k)[0], v0_stddev(k)[0], pos0_stddev(k)[1], v0_stddev(k)[1]], dtype=float) ** 2)

        return F, C_w, mean_0, cov_0, S_w

    def _define_ca_model(self,
                         T,
                         S_w=(9.6E11, 9.6E11),
                         # System noise power spectral density in px² / sec⁵, variance v = S_w * T³/3
                         # -> Stddev. of v approx. 200 px / sec * 0.05 mm / px = 10 mm / sec,
                         pos_0=(160.0, 1160),  # px,
                         v_0=(20000, 0),  # px / sec., 100 px / frame, approx. 5 mm / frame or 1 m / sec.,
                         a_0=(500, 0),
                         pos0_stddev=(80, 640),  # px, approx. [4, 16] mm
                         v0_stddev=(2000, 2000),  # px / sec., approx. 0.5 mm / frame.
                         a0_stddev=(2000, 2000),
                         ):
        """Defines a (potentially time-variant) Constant-Velocity (CV) motion model (also known als white-noise
        jerk model).

         Format State:

            [x, v_x, a_x y, v_y, a_y]

            where v_x and v_y are the velocities in x- and y-direction and a_x and a_y are the accelerations in x- and
            y-direction.

        Note that the units of the parameters must be consistent, but are not restricted to pixels or DEM coordinate
        unit and time steps. For example, also normalized values can be used.

        Apart from T, all parameters can be defined in four ways:

                i) as a (static) tuple of floats or integers,
               ii) as function f(time_step) -> (float, float),
              iii) as a string representing a lambda function f(time_step) -> (float, float), or
               iv) as a string representing a function call (with args and/or kwargs)) of predefined function
                    f(time_step) -> (float, float).

        :param T: A float representing the time interval between two consecutive time steps.
        :param S_w: A list of floats with length 2, a function f(time_step) -> (float, float) or a string representing
            either a lambda function or function call of a predefined function (with args and/or kwargs), all
            representing the power spectral density of the system noise in x- and y-direction.
        :param pos_0: A list of floats with length 2, a function f(time_step) -> (float, float) or a string representing
            either a lambda function or function call of a predefined function (with args and/or kwargs), all
            representing the initial position mean of the tracks.
        :param v_0: A list of floats with length 2, a function f(time_step) -> (float, float) or a string representing
            either a lambda function or function call of a predefined function (with args and/or kwargs), all
            representing the initial velocity mean of the tracks.
         :param a_0: A list of floats with length 2, a function f(time_step) -> (float, float) or a string representing
            either a lambda function or function call of a predefined function (with args and/or kwargs), all
            representing the initial acceleration mean of the tracks.
        :param pos0_stddev: A list of floats with length 2, a function f(time_step) -> (float, float) or a string
            representing either a lambda function or function call of a predefined function (with args and/or kwargs),
            all representing the initial position standard deviation of the tracks.
        :param v0_stddev: A list of floats with length 2, a function f(time_step) -> (float, float) or a string
            representing either a lambda function or function call of a predefined function (with args and/or kwargs),
            all representing the initial velocity standard deviation of the tracks.
        :param a0_stddev: A list of floats with length 2, a function f(time_step) -> (float, float) or a string
            representing either a lambda function or function call of a predefined function (with args and/or kwargs),
            all representing the initial acceleration standard deviation of the tracks.

        :returns:
            F: A function f(time_step) -> np.array representing the (potentially time-variant) transition matrix of the
                model.
            C_w: A function f(time_step) -> np.array representing the (potentially time-variant) system noise covariance
                of the model.
            mean_0: A function f(time_step) -> np.array representing the (potentially time-variant) expected value of
                the initial state.
            cov_0: A function f(time_step) -> np.array representing the (potentially time-variant) covariance of
                 the initial state.
            S_w: A function f(time_step) -> (float, float) representing the (potentially time-variant) power spectral
                density of the system noise in x- and y-direction.
        """
        # convert arguments to functions
        S_w = self._define_func(S_w)
        pos_0 = self._define_func(pos_0)
        v_0 = self._define_func(v_0)
        a_0 = self._define_func(a_0)
        pos0_stddev = self._define_func(pos0_stddev)
        v0_stddev = self._define_func(v0_stddev)
        a0_stddev = self._define_func(a0_stddev)

        # some checks on the input
        if S_w(0).shape != (2,):
            raise ValueError('The power spectral density S_w must be of length 2.')
        if pos_0(0).shape != (2,):
            raise ValueError('The initial velocity v_0 must be of length 2.')
        if v_0(0).shape != (2,):
            raise ValueError('The initial velocity v_0 must be of length 2.')
        if a_0(0).shape != (2,):
            raise ValueError('The initial acceleration a_0 must be of length 2.')
        if pos0_stddev(0).shape != (2,):
            raise ValueError('The initial velocity stddev v0_stddev must be of length 2.')
        if v0_stddev(0).shape != (2,):
            raise ValueError('The initial velocity stddev v0_stddev must be of length 2.')
        if a0_stddev(0).shape != (2,):
            raise ValueError('The initial acceleration stddev a0_stddev must be of length 2.')

        # Transition matrix
        F = lambda k: np.array([[1, T, T ** 2 / 2, 0, 0, 0],
                                [0, 1, T, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, T, T ** 2 / 2],
                                [0, 0, 0, 0, 1, T],
                                [0, 0, 0, 0, 0, 1]], dtype=float)

        # System noise covariance
        C_w = lambda k: np.matmul(np.array([[T ** 5 / 20, T ** 4 / 8, T ** 3 / 6, 0, 0, 0],
                                            [T ** 4 / 8, T ** 3 / 3, T ** 2 / 2, 0, 0, 0],
                                            [T ** 3 / 6, T ** 2 / 2, T, 0, 0, 0],
                                            [0, 0, 0, T ** 5 / 20, T ** 4 / 8, T ** 3 / 6],
                                            [0, 0, 0, T ** 4 / 8, T ** 3 / 3, T ** 2 / 2],
                                            [0, 0, 0, T ** 3 / 6, T ** 2 / 2, T]], dtype=float),
                                  np.diag([S_w(k)[0], S_w(k)[0], S_w(k)[0], S_w(k)[1], S_w(k)[1], S_w(k)[1]]))
        # Initial values
        mean_0 = lambda k: np.array([pos_0(k)[0], v_0(k)[0], a_0(k)[0],
                                     pos_0(k)[1], v_0(k)[1], a_0(k)[1]], dtype=float)
        cov_0 = lambda k: np.diag(
            np.array([pos0_stddev(k)[0], v0_stddev(k)[0], a0_stddev(k)[0],
                      pos0_stddev(k)[1], v0_stddev(k)[1], a0_stddev(k)[1]], dtype=float) ** 2)

        return F, C_w, mean_0, cov_0, S_w

    def create_particles(self):
        """Creates new particles in the current time step, i.e., initializes new tracks.

        The number of particles per process are sampled from the birth_rate_mean_and_stddev of the processes.
        """
        # sample the number of new tracks for each process
        n_new_tracks = np.zeros(len(self._process_parameters), dtype=int)
        for process_no, process_birth_rate_mean_and_stddev in enumerate(self._birth_rate_mean_and_stddev):
            process_n_new_tracks = self._rng_for_particle_initialization.normal(
                *process_birth_rate_mean_and_stddev(self._k))
            n_new_tracks[process_no] = max(round(process_n_new_tracks), 0)

        # check if we have too many tracks, if so cut them to the total number of particles
        if self._counter + np.sum(n_new_tracks) > self._total_num_particles:
            # choose self._total_num_particles - self._counter elements at random from len(n_new_tracks) classes with
            # n_new_tracks[i] elements each without replacement --> sample from multivariate hypergeometric distribution
            n_new_tracks = self._rng_for_particle_initialization.multivariate_hypergeometric(
                n_new_tracks,
                self._total_num_particles - self._counter)
            assert np.sum(n_new_tracks) == self._total_num_particles - self._counter

        # sample the initial states and update motion states, particle class, particle id, and existence
        for process_no, process_parameters in enumerate(self._process_parameters):
            mean_0, cov_0 = process_parameters["mean_0"], process_parameters["cov_0"]
            init_state = self._start_tracks(mean_0(self._k), cov_0(self._k),
                                            num_tracks_in_time_step=n_new_tracks[process_no])
            self._sampled_motion_states[process_no] = np.r_[self._sampled_motion_states[process_no], init_state]
            self._particle_id = np.r_[
                self._particle_id, np.arange(self._counter, self._counter + n_new_tracks[process_no])]
            self._particle_id_in_motion_states[process_no] = np.r_[self._particle_id_in_motion_states[process_no],
            np.arange(self._counter, self._counter + n_new_tracks[process_no])]
            self._particle_class = np.r_[
                self._particle_class, np.full(n_new_tracks[process_no], fill_value=self._class_labels[process_no])]
            self._existence = np.r_[self._existence, np.full(n_new_tracks[process_no], fill_value=True)]
            self._counter += n_new_tracks[process_no]

    def _start_tracks(self,
                      mean_0,
                      cov_0,
                      num_tracks_in_time_step,
                      num_tracks_in_time_step_start=None,
                      start_tracks_recursive_counter=0):
        """Starts num_tracks_in_time_step new tracks.

        Note that this function checks if the first created position is a valid state. If not, the track is
        ignored. To guarantee creating exactly num_tracks_in_time_step valid tracks, we create some more than
        num_tracks_in_time_step tracks and delete unnecessary tracks if we have too much valid tracks and call the
        function recursively if we have too few valid tracks.

        :param mean_0: A np.array representing the expected value of the initial state at the current
            time step.
        :param cov_0: A np.array representing the covariance of the initial state.
        :param num_tracks_in_time_step: An integer, the number of tracks to return in the current time step.
        :param num_tracks_in_time_step_start: An integer, the number of tracks to create in the current time step.
            If num_tracks_in_time_step_start > num_tracks_in_time_step, only the first num_tracks_in_time_step
            tracks will be used. If None, num_tracks_in_time_step will be used
        :param start_tracks_recursive_counter: An integer, the number of recursive calls of this function. If this
            number exceeds 10, a ValueError will be raised.

        :returns:
            state_0: A np.array of shape [num_tracks_in_time_step, state_length] representing the first state of the
                created tracks.
        """
        # draw random samples
        num_samples = 40 * int(  # safety factor 40
            num_tracks_in_time_step_start if num_tracks_in_time_step_start is not None else num_tracks_in_time_step)
        state_0 = self._rng_for_particle_initialization.multivariate_normal(mean_0, cov=cov_0, size=num_samples)

        # check initial state
        pos_0 = self._state_to_pos(state_0)
        velo_0 = self._state_to_velo(state_0)
        initial_invalid_mask = np.logical_or.reduce((pos_0[:, 0] > 0,
                                                     pos_0[:, 1] > self._area_width - self._particle_r,
                                                     pos_0[:, 0] < -velo_0[:, 0] * self._T,
                                                     pos_0[:, 1] < self._particle_r))
        num_valid_tracks = np.sum(np.logical_not(initial_invalid_mask))

        # if not enough measurements try again
        if num_valid_tracks < num_tracks_in_time_step:
            start_tracks_recursive_counter += 1
            if start_tracks_recursive_counter >= 10:
                raise ValueError('Parameters for track generation are poorly chosen. '
                                 'Even after 10 recursive attempts not enough valid tracks were generated.')
            return self._start_tracks(mean_0,
                                      cov_0,
                                      num_tracks_in_time_step=num_tracks_in_time_step,
                                      num_tracks_in_time_step_start=int(round(
                                          num_tracks_in_time_step ** 2 / np.where(num_valid_tracks,
                                                                                  num_valid_tracks != 0,
                                                                                  1),
                                          0)),
                                      start_tracks_recursive_counter=start_tracks_recursive_counter)

        # shorten created tracks to exactly num_tracks_in_time_step
        state_0 = state_0[np.logical_not(initial_invalid_mask)][:num_tracks_in_time_step]
        return state_0

    def track_particles(self, measurements):
        """This is a dummy/mock for a tracker for the particles and constitutes the (only) interface for data exchange
        to the controllers.

        Instead of really tracking the particle, we assume here that we measure the mean of the motion state perfectly
        and add an artificial noise for the motion states' covariance.

        This mocks outputs the motion state mean and covariance as for a constant velocity model, even if internally
        a constant acceleration model is used. This is to be consistent with the controllers, that currently
        only support constant velocity motion states.

         Format measurements:

            [x, y, label]

            with label either 0 ("keep the particle") or 1 ("eject the particle").

         Format motion_state:

            [x_pos, x_velo, y_pos, y_velo]

        :param measurements: A np.array of shape [num_measurements, measurement_vector_length], the measurements
            (including x-position, y-position and class label) for the current time step. Used as dummy input here.

        :returns:
             estimated_motion_state_mean: A np.array of shape [num_particles, 4], the estimated motion state mean.
             estimated_motion_state_cov: A np.array of shape [num_particles, 4, 4], the estimated motion state
                covariance.
             particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
                particle") or 1 ("eject the particle") representing the particle classes.
             particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        """
        x_pos, y_pos = self.position.T
        x_velo, y_velo = self.velocity.T
        estimated_motion_state_mean = np.stack((x_pos, x_velo, y_pos, y_velo), axis=-1)
        estimated_motion_state_cov = 0.00000001 * self.C_w_pos_velo  # use a fraction of the system noise

        return (estimated_motion_state_mean,
                estimated_motion_state_cov,
                self._particle_class,
                self._particle_id,
                self._existence)

    def move_particles(self):
        super().move_particles()
        self._wall_collision_model()

    def _wall_collision_model(self):
        """Simple collision model for collisions with walls orientated along the transport direction (x).

        The model assumes reflection and energy preservation.

         Format state:

            [x, v_x, ... , y, v_y, ...]
        """
        for process_no, process_param in enumerate(self._process_parameters):
            y_ind = int(self._sampled_motion_states[process_no].shape[1] / 2)

            # collision with upper wall
            collision_mask = self._sampled_motion_states[process_no][:, y_ind] > self._area_width - self._particle_r
            self._sampled_motion_states[process_no][collision_mask, y_ind] = 2 * (
                    self._area_width - self._particle_r) - self._sampled_motion_states[process_no][
                                                                                 collision_mask, y_ind]
            self._sampled_motion_states[process_no][collision_mask, (y_ind + 1):] *= -1

            # collision with lower wall
            collision_mask = self._sampled_motion_states[process_no][:, y_ind] < self._particle_r
            self._sampled_motion_states[process_no][collision_mask, y_ind] = 2 * self._particle_r - \
                                                                             self._sampled_motion_states[process_no][
                                                                                 collision_mask, y_ind]
            self._sampled_motion_states[process_no][collision_mask, (y_ind + 1):] *= -1

    def _define_mass_flows(self, birth_rate_mean_and_stddev_ls):
        """Defines the quantities of mass flows, i.e., the number of tracks births, for the defined processes as a
        function of time steps.

         Format mass_flow_parameters_ls:

            [mean, stddev],

            where mean and standard deviation are in particles per time step.

        The mass flow can be defined in three ways:

                i) as a (static) tuple of integers,
               ii) as function f(time_step) -> (int, int),
              iii) as a string representing a lambda function f(time_step) -> (int, int), or
               iv) as a string representing a function call (with args and/or kwargs)) of predefined function
                    f(time_step) -> (int, int).

         Example:

            mass_flow_parameters_ls = [
                 (10, 0),
                 lambda k: (10 + 2*k, 2),
                 'lambda k: (10 + 2*k, 2)',
                 'ramp((10, 0), (100, 0)',
                 ]

        :param birth_rate_mean_and_stddev_ls: A list of length num_processes representing the mean and standard
            deviation for each mass flow. Type and format as explained above.
        """
        for process_no, mass_flow in enumerate(birth_rate_mean_and_stddev_ls):
            self._birth_rate_mean_and_stddev[process_no] = self._define_func(mass_flow)

    def _shallow_copy(self, seed=None, freeze_scenario=False):
        return AreaParticleSimulator(total_num_particles=self._total_num_particles,
                                     T=self._T,
                                     process_parameters=self._original_process_parameters,
                                     birth_rate_mean_and_stddev=self._original_birth_rate_mean_and_stddev,
                                     class_labels=self._class_labels,
                                     area_width=self._area_width,
                                     particle_r=self._particle_r,
                                     init_k=self._init_k,
                                     seed=seed,
                                     freeze_scenario=freeze_scenario,
                                     S_v=self._S_v)


class AbstractContactSimulator(ABC):
    """Abstract class for simulators of the particle-actor contact in a sorting machine (either 1D or 2D optical
    sorter).

    The contact simulator similar to the models defined in contact_models.py, models the particle-actor contact. However,
    there are three significant differences, which is why the classes are strictly separated:

        i) In contrast to the contact models defined in contact_models.py, the contact simulator introduces an
           additional third outcome of a particle-actor contact. In addition to being ejected or not ejected, a particle
           can also be disturbed. A disturbed particle is a particle that was hit by an actor when it was not supposed
           to be hit, for example because the actor was not in a HIT status, and we therefore do not know the exact
           outcome of the contact.

       ii) The conditions being used to classify a particle as either ejected, not ejected, or disturbed are slightly
           different from the models in contact_models.py. This is because the time-discrete nature of the particle
           simulation and the deterministic modeling used here (we do not have distributions for the particle motion or
           the arrival time in the simulator, which we always have in the models in contact_models.py, even if we call
           their sampling methods). Since we need to make a decision for the single simulated particle (and do not have
           a set of samples, or a distribution describing the particle motion), this required some additional
           adjustments. In particular, this leads to the introduction of the spatial tolerance parameter, which defines
           some tolerance in particle position due to the time-discretization of the particle motion during simulation
           (see the documentation of the subclasses for more details).

      iii) A third difference is that the contact simulator is totally deterministic, i.e., there is no sampling
           involved. If a particle meets the conditions for being ejected, it will always be ejected. This is in
           contrast to the contact models in contact_models.py, where the outcome of a contact is sampled from a
           distribution.
    """

    def __init__(self,
                 actors,
                 spatial_tolerance,
                 ):
        """Initializes the contact simulator.
        
        :param actors: An ActorSimulator instance, the actor simulator to be used for simulating the actors.
        :param spatial_tolerance: The spatial tolerance for the particle position (in either of the two directions).
            Must be in (0, distance between two actors / 2). Note that the optimal value of the spatial tolerance in
            transport direction is half the distance between the particle positions in two consecutive time steps.
            However, one might choose a larger value to account for the fact that actors are not line-like objects and
            do also have certain extent.
        """
        if not isinstance(actors, ActorSimulator):
            raise ValueError('The actors of the simulator must be an ActorSimulator object.')

        self._actors = actors
        self._spatial_tolerance = spatial_tolerance

        # initialize the arrays/states
        self._old_particle_pos = None

    @staticmethod
    @abstractmethod
    def _check_if_in_tolerance_area(particle_pos, particle_existence, actor_pos, tolerance):
        """Calculates which particles tolerance span over a specific actor position.

        :param particle_pos: A np.array of shape [num_particles, ...], the positions of the particles.
        :param particle_existence: A Boolean np.array of shape [num_particles], the existence of the particles. True
            indicates that a particle exists.
        :param actor_pos: A float, the positions of the actor.
        :param tolerance: A float, the spatial tolerance for the particle position (in either of the two directions).

        :returns: A Boolean np.array of shape [num_particles], a mask for which particles are hit by the actor.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def _calculate_contact(self, particle_pos, particle_existence, actor_status):
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def shallow_copy(self, actors):
        """Creates a shallow copy of the contact simulator, i.e., a new instance of the contact simulator with the same
        settings but independent, freshly-initialized states.

        :param actors: An ActorSimulator instance, the actor simulator to be used for simulating the actors.

        :returns: An AbstractContactSimulator instance, a shallow copy of the contact simulator.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def spatial_tolerance(self):
        """The spatial tolerance for the particle position (in either of the two directions).
        
        :returns: A float or a tuple, the spatial tolerance for the particle position. See the documentation of the
            subclasses for more details.
        """
        return self._spatial_tolerance

    def calculate_contact(self, particle_pos, particle_existence, actor_status):
        """Contact model for the discrete-time simulation, i.e., computes the hit and disturbed hits masks for the
        particles.

        If an entry of the mask is True, the particle is hit/disturbed in the current time step.

        :param particle_pos: A np.array of shape [num_particles, ...], the positions of the particles.
        :param particle_existence: A Boolean np.array of shape [num_particles], the existence of the particles. True
            indicates that a particle exists.
        :param actor_status: An object np.array of shape [num_actors] representing the actors' current status.

        :returns:
            hits: A Boolean np.array of shape [num_particles], a mask for which particles are hit by the actors.
            disturbed_hits: A Boolean np.array of shape [num_particles], a mask for which particles are disturbed by the
                actors.
        """
        self._check_spatial_tolerance(particle_pos)
        return self._calculate_contact(particle_pos, particle_existence, actor_status)

    def update_particle_pos(self, particle_pos):
        """Updates the internal state of the simulator with the current particle positions.

        :param particle_pos: A np.array of shape [num_particles, ...], the positions of the particles.
        """
        self._old_particle_pos = particle_pos

    def remove_particle_pos(self, remove_mask):
        """Removes the particles from the internal state of the simulator.

         :param remove_mask: A Boolean np.array of shape [num_particles] used as a mask to remove particles. True
            indicates that a particle should be removed.
        """
        self._old_particle_pos = self._old_particle_pos[np.logical_not(remove_mask)]

    def _check_spatial_tolerance(self, particle_pos):
        """Checks if the spatial tolerance is smaller than the current difference in particle position of two
        consecutive time steps.

        :param particle_pos: A np.array of shape [num_particles, ...], the positions of the particles.
        """
        if np.any(np.abs(particle_pos - self._old_particle_pos) / 2 > self._spatial_tolerance):
            raise ValueError(
                'The spatial tolerance is too small for the current difference in particle position of two '
                'consecutive time steps. Increase the spatial tolerance or decrease the sampling time interval T.')
        self._old_particle_pos = particle_pos

    def reset(self):
        """Resets the contact model to the initial state."""
        self._old_particle_pos = None


class AreaContactSimulator(AbstractContactSimulator):
    """A simulators of the particle-actor contact on the groove (1D optical sorter).

    The contact simulator similar to the models defined in contact_models.py, models the particle-actor contact. However,
    there are three significant differences, which is why the classes are strictly separated:

        i) In contrast to the contact models defined in contact_models.py, the contact simulator introduces an
           additional third outcome of a particle-actor contact. In addition to being ejected or not ejected, a particle
           can also be disturbed. A disturbed particle is a particle that was hit by an actor when it was not supposed
           to be hit, for example because the actor was not in a HIT status, and we therefore do not know the exact
           outcome of the contact.

       ii) The conditions being used to classify a particle as either ejected, not ejected, or disturbed are slightly
           different from the models in contact_models.py. This is because the time-discrete nature of the particle
           simulation and the deterministic modeling used here (we do not have distributions for the particle motion or
           the arrival time in the simulator, which we always have in the models in contact_models.py, even if we call
           their sampling methods). Since we need to make a decision for the single simulated particle (and do not have
           a set of samples, or a distribution describing the particle motion), this required some additional
           adjustments. In particular, this leads to the introduction of the spatial tolerance parameter, which defines
           some tolerance in particle position due to the time-discretization of the particle motion during simulation
           (see the documentation of the subclasses for more details).

      iii) A third difference is that the contact simulator is totally deterministic, i.e., there is no sampling
           involved. If a particle meets the conditions for being ejected, it will always be ejected. This is in
           contrast to the contact models in contact_models.py, where the outcome of a contact is sampled from a
           distribution.

    Contact model:

        We consider an actor to be fully described by its position, width (orthogonal to the transport direction) and
        its status, as defined in the ActorSimulator class. The particle therefore is

            - ejected if it arrives at an actor j within its width when it has status HIT,
            - disturbed if it arrives at an actor j within its width when it has status UP or DOWN, and
            - not ejected or disturbed if it does not arrive within the width of any actor j when it has a status HIT,
                UP, or DOWN,

        We refer to this model as idealized.

        Idealized contact model (2 by 2 grid):

                                    upper wall    actors 3 & 4
                                    ----------------------------
                                            |
                                            O          |  actor
                                                       |   width
                                            |
                                            |        O |
                                                       |
                                    ----------------------------
                                       actors 1 & 2       lower wall


        However, the idealized contact model requires a time-continuous simulation of particle motion. Since the
        simulator uses a time-discrete simulation of the particle motion, a particle will never be exactly at an actor's
        position. This leads to the introduction of the spatial tolerance parameter, which defines some tolerance for
        the particle-actor distance to be still considered as a contact. The particle therefore is

            - ejected if it is located at an actor j within its width and within the spatial tolerance when it has
                status HIT
            - disturbed if it is at an actor j within its width and within the spatial tolerance when it has status UP
                or DOWN, and
            - not ejected or disturbed if it is not within the width or spatial tolerance of any actor j when it has a
                status HIT, UP, or DOWN or is not within the width and spatial tolerance of any actor j.

        Refined contact model (2 by 2 grid):

                                                                     spatial tolerance:  –>, /\ resp. <–, \/

                                    upper wall    actors 3 & 4
                                    ----------------------------
                                           /\
                                          <–O->        |  actor
                                           \/          |   width
                                            |       /\
                                            |      <–O->
                                                    \/ |
                                    -----------------------------
                                       actors 1 & 2       lower wall


        An alternative (but less precise, see the discussion below) view on the spatial tolerance is that it defines the
        operating range of an actor, i.e., only particles that lie within this operating range can be ejected.

        Role of the spatial tolerance:

            Note that the value of the spatial tolerance should be dependent on the sampling time interval T and the
            particle velocity. The smaller T, the closer the particle will come to the actor's position, and in the
            limit T --> 0, also the spatial tolerance should go to zero. If the spatial tolerance is too large,
            particles might be disturbed in the time-step before they actually arrived at the actor (and would be
            ejected). If the spatial tolerance is too small, particles might pass the actor without being ejected,
            although it was in a HIT status. Therefore, the optimal value of the spatial tolerance (in transport
            direction and orthogonal to it) is half the distance between the particle positions in two consecutive time
            steps. However, one might choose a larger value to account for the fact that actors are not line-like
            objects and do also have a certain length and width.

        For the implemented contact model, we extend the refined and idealized contact model by introducing a
        disturbance area. The disturbance area is a rectangular area around the actor's position (not necessarily
        centered at the actor's position) with a certain extent. If a particle is within the disturbance area of an
        actor and the actor is in the HIT, UP, or DOWN status, it can only be ejected by the same actor in the following
        time steps but not by any other actor. This models situations where a particle crosses multiple actors and one
        of the earlier actors accidentally disturbs the particle.

        Implemented contact model (2 by 2 grid):

                                                                     spatial tolerance:  –>, /\ resp. <–, \/
                                                                     disturbance area:  _______
                                    upper wall    actors 3 & 4                          |     |
                                    -----_______ ----------------                       |     |
                                         |   /\|        _______                         –––––––
                                         | <–O->        |   | |  actor
                                         –––-\/––       |   | |   width
                                         _______        –––––––
                                         |   | |      /\ _______
                                         |   | |    <–O->   | |
                                         –––––––      \/|   | |
                                    --------------------–––––––-
                                       actors 1 & 2       lower wall

        The final conditions for a particle to be ejected or disturbed are therefore: A particle is

            - ejected if it is located at an actor j within its width and within the spatial tolerance when it has
                status HIT
            - disturbed
                - if it is at an actor j within its width and within the spatial tolerance when it has status UP or
                        DOWN, or
                - if it was within the disturbance area of an actor j when it had a status HIT, UP, or DOWN but now left
                    the actor without being hit by the same actor
            - not ejected or disturbed, otherwise.

        Note that for the calculation if the particle is in the disturbance area, we do not consider the spatial
        tolerance, as the disturbance area is usually larger than the spatial tolerance and therefore, if discretization
        errors lead to the particle being slightly outside the disturbance area, there a good chance that it will be
        within the disturbance area in the next time step.

        Furthermore, note that because of the second disturbance condition, the model is stateful, i.e., it needs to
        know the potential disturbance of the particles of the previous time step.
    """

    def __init__(self,
                 actors,
                 spatial_tolerance,
                 disturbance_area_offset=(0, 0),
                 disturbance_area_extent=None,
                 ):
        """Initializes the contact simulator.

        :param actors: An ActorSimulator instance, the actor simulator to be used for simulating the actors.
        :param spatial_tolerance: A tuple of length 2 of floats, the spatial tolerance along and orthogonal to the
            transport direction for the particle position (in either of the two directions). Must be in
            (0, distance between two actors (including widths) / 2). Note that the optimal value of the spatial
            tolerance in transport direction is half the distance between the particle positions in two consecutive time
            steps (both along and orthogonal to the transport direction). However, one might choose a larger value to
            account for the fact that actors are not line-like objects and do also have certain extent in transport
            direction.
        :param disturbance_area_offset: A tuple of length 2 of floats representing the offset of the center points of
            the disturbance area along and orthogonal to the transport direction to the actor position. The disturbance
            are may be seen as the physical area of the actor, whereas the actor's position may be seen as the optimal
            point where a particle should be ejected. Therefore, the actor's center does usually not coincide with the
            center of the disturbance area.
        :param disturbance_area_extent: None or a tuple of length 2 of floats representing the length and width of the
            disturbance area. Defaults to the actor length and width.
        """
        if len(spatial_tolerance) != 2:
            raise ValueError('The spatial tolerance spatial_tolerance must be of length 2.')
        if (not np.isscalar(spatial_tolerance[0])) or spatial_tolerance[0] <= 0 or spatial_tolerance[0] >= np.min(
                np.diff(actors.row_pos) / 2):
            raise ValueError(
                'The spatial tolerance in transport direction spatial_tolerance[0] must be a positive scalar smaller'
                ' than half the distance between two actors.')
        if (not np.isscalar(spatial_tolerance[1])) or spatial_tolerance[1] < 0 or np.any(
                spatial_tolerance[1] >= np.array(
                    [np.min((xylw[1:, 1] - xylw[1:, -1] / 2) - (xylw[:-1, 1] + xylw[:-1, -1] / 2)) for xylw in
                     actors.actor_grid]) / 2):
            raise ValueError(
                'The spatial tolerance orthogonal to the transport direction spatial_tolerance[1] must be a positive '
                'scalar smaller than half the distance between two actors (including widths).')

        if len(disturbance_area_offset) != 2:
            raise ValueError('The offset of the center points of the disturbance to the actor position '
                             'disturbance_area_offset must be of length 2.')
        if disturbance_area_extent is not None and len(disturbance_area_extent) != 2 or np.any(
                np.asarray(disturbance_area_extent) <= 0):
            raise ValueError('The length and width of the disturbance area disturbance_area_extent '
                             'must be of length 2 and positive.')
        super().__init__(actors=actors,
                         spatial_tolerance=spatial_tolerance)

        self._disturbance_area_extent = np.ones_like(self._actors.pos) * np.asarray(
            disturbance_area_extent)[None, :] if disturbance_area_extent is not None else np.stack(
            (self._actors.length, self._actors.width), axis=1)
        self._disturbance_area_center = self._actors.pos + np.asarray(disturbance_area_offset)[None, :]

        # check if the disturbance areas are non-overlapping
        self._actors.check_two_dimensional_actor_grid(self._disturbance_area_center,
                                                      self._disturbance_area_extent[:, 0],
                                                      self._disturbance_area_extent[:, 1])

        # initialize the arrays/states
        self._potential_disturbances = np.zeros((len(self._actors.pos), 0), dtype=bool)

        # save the initial values for later reuse
        self._original_disturbance_area_offset = disturbance_area_offset
        self._original_disturbance_area_extent = disturbance_area_extent

    @property
    def disturbance_area_center(self):
        """The centers of the actors' disturbance areas.

        Usually, will coincide with the actor's physical center.

        :returns: A np.array of shape [num_actors, 2], the actors' disturbance areas.
        """
        return self._disturbance_area_center

    @property
    def disturbance_area_extent(self):
        """The extent of the actors' disturbance areas.

        Usually, will coincide with the actor's physical extent.

        :returns: A np.array of shape [num_actors, 2], the actors' disturbance areas.
        """
        return self._disturbance_area_extent

    @staticmethod
    def _check_if_in_tolerance_area(particle_pos, particle_existence, actor_pos, tolerance):
        """Calculates which particles tolerance span over a specific actor position.

        :param particle_pos: A np.array of shape [num_particles, 2], the positions of the particles.
        :param particle_existence: A Boolean np.array of shape [num_particles], the existence of the particles. True
           indicates that a particle exists.
        :param actor_pos: A np.array of shape [2], the positions of the actor.
        :param tolerance: A np.array of shape [2], the spatial tolerance for the particle position (in either of the
           two directions).

        :returns: A Boolean np.array of shape [num_particles], a mask for which particles are hit by the actor.
        """
        x_check = np.logical_and(
            particle_pos[:, 0] + tolerance[0] >= actor_pos[0],
            particle_pos[:, 0] - tolerance[0] <= actor_pos[0])
        y_check = np.logical_and(
            particle_pos[:, 1] + tolerance[1] >= actor_pos[1],
            particle_pos[:, 1] - tolerance[1] <= actor_pos[1])
        return np.logical_and.reduce((x_check, y_check, particle_existence))

    def _calculate_contact(self, particle_pos, particle_existence, actor_status):
        """Contact model for the discrete-time simulation, i.e., computes the hit and disturbed hits masks for the
        particles.

        If an entry of the mask is True, the particle is hit/disturbed in the current time step.

        :param particle_pos: A np.array of shape [num_particles, 2], the positions of the particles.
        :param particle_existence: A Boolean np.array of shape [num_particles], the existence of the particles. True
            indicates that a particle exists.
        :param actor_status: An object np.array of shape [num_actors] representing the actors' current status.

        :returns:
            hits: A Boolean np.array of shape [num_particles], a mask for which particles are hit by the actors.
            disturbed_hits: A Boolean np.array of shape [num_particles], a mask for which particles are disturbed by the
                actors.
        """
        hits = np.zeros(len(particle_pos), dtype=bool)
        disturbed_hits = np.zeros(len(particle_pos), dtype=bool)
        for j, actor_status_j in enumerate(actor_status):
            # TODO: Vectorize/parallelize the implementation
            potential_hits_at_j = self._check_if_in_tolerance_area(
                particle_pos,
                particle_existence,
                actor_pos=self._actors.pos[j],
                tolerance=np.asarray(self._spatial_tolerance) + np.array([0.0, self._actors.width[j] / 2]))
            potential_disturbance_at_j = self._check_if_in_tolerance_area(
                particle_pos,
                particle_existence,
                actor_pos=self._disturbance_area_center[j],
                tolerance=self._disturbance_area_extent[j] / 2,
            )
            if actor_status_j == ActorStatus.HIT:
                hits_at_j = potential_hits_at_j
                hits[potential_hits_at_j] = True
            elif actor_status_j == ActorStatus.UP or actor_status_j == ActorStatus.DOWN:
                hits_at_j = np.zeros_like(particle_existence, dtype=bool)
                disturbed_hits[potential_hits_at_j] = True
            else:
                hits_at_j = np.zeros_like(particle_existence, dtype=bool)

            was_disturbed_but_not_ejected_on_j = np.logical_and.reduce((self._potential_disturbances[j],
                                                                        particle_existence,
                                                                        np.logical_not(potential_disturbance_at_j),
                                                                        np.logical_not(hits_at_j)))
            disturbed_hits[was_disturbed_but_not_ejected_on_j] = True

            if actor_status_j == ActorStatus.HIT or actor_status_j == ActorStatus.UP \
                    or actor_status_j == ActorStatus.DOWN:
                self._potential_disturbances[j] = np.logical_or(self._potential_disturbances[j],
                                                                potential_disturbance_at_j)

        hits[disturbed_hits] = False  # previously disturbed particles by an actor different from the one that ejects
        # the particle are not counted as ejected

        assert np.all(hits.astype(int) + disturbed_hits.astype(
            int) <= 1), 'A particle can only be hit or disturbed by one actor at a time.'

        return hits, disturbed_hits

    def update_particle_pos(self, particle_pos):
        super().update_particle_pos(particle_pos)
        pd = np.zeros((len(self._actors.pos), len(particle_pos)), dtype=bool)
        pd[:, :self._potential_disturbances.shape[1]] = self._potential_disturbances
        self._potential_disturbances = pd

    def remove_particle_pos(self, remove_mask):
        super().remove_particle_pos(remove_mask)
        self._potential_disturbances = self._potential_disturbances[:, np.logical_not(remove_mask)]

    def reset(self):
        super().reset()
        self._potential_disturbances = np.zeros((len(self._actors.pos), 0), dtype=bool)

    def shallow_copy(self, actors):
        return AreaContactSimulator(actors=actors,
                                    spatial_tolerance=self._spatial_tolerance,
                                    disturbance_area_offset=self._original_disturbance_area_offset,
                                    disturbance_area_extent=self._original_disturbance_area_extent)


class AbstractSortingSimulator(ABC):
    """Abstract class for discrete-time simulators for a (potentially controlled) sorting machine (either 1D or 2D
    optical sorter).

    """

    def __init__(self,
                 particle_simulator,
                 actors,
                 contact_simulator,
                 array_end=None,
                 **kwargs,
                 ):
        """Initializes the simulator.

        :param particle_simulator: An AbstractParticleSimulator instance, the particle simulator to be used for
            simulating the particles movement and track birth and death.
        :param actors: An ActorSimulator instance, the actor simulator to be used for simulating the actors.
        :param contact_simulator: An AbstractContactSimulator instance, the contact simulator to be used for simulating
            the contact between the particles and the actors.
        :param array_end: None or a float, the end coordinate of the actuator array. Defaults to the position of the
            last actor.
        """
        if not isinstance(actors, ActorSimulator):
            raise ValueError('The actors of the simulator must be an ActorSimulator object.')
        if actors.T != particle_simulator.T:
            raise ValueError('The sampling time difference T of particle_simulator and actors must be the same.')
        if actors.initial_time_step != particle_simulator.initial_time_step:
            raise ValueError('The initial time step of particle_simulator and actors must be the same.')
        if array_end is not None and (not np.isscalar(array_end) or array_end <= 0):
            raise ValueError('The end coordinate of the actuator array must be a positive scalar.')

        self._particle_simulator = particle_simulator
        self._actors = actors
        self._contact_simulator = contact_simulator

        self._array_end = float(array_end) if array_end is not None else self._actors.end_array

        # save additional parameters as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        # store the kwargs for later reuse
        self._original_kwargs = kwargs

    @property
    def T(self):
        """The sampling time difference.

        :returns: A float representing the time interval between two consecutive time steps.
        """
        return self._actors.T

    @property
    def particle_simulator(self):
        """The particle simulator.

        :returns: An AbstractParticleSimulator object, the particle simulator to be used for the simulation.
        """
        return self._particle_simulator

    @property
    def actors(self):
        """The actors.

        :returns: An ActorSimulator object, the actor simulator to be used for the simulation.
        """
        return self._actors

    @property
    def array_end(self):
        """The end coordinate of the actuator array.

        :returns: A float, the end coordinate of the actuator array.
        """
        return self._array_end

    @abstractmethod
    def show_action_fn_generator(self,
                                 particle_r=None,
                                 no_show_animation=False,
                                 save_animation=False,
                                 animation_dir=None,
                                 remove_particles_from_lists_if_hit=True,
                                 **kwargs):
        """Build a function that can be called in every time step and visualizes the current situation as interactive
         animation of the simulation in a video stream.

        :param particle_r: None or a float, the radius of the particles for visualization.
        :param no_show_animation: A Boolean, whether to show the animation or not.
        :param save_animation: A Boolean, whether to save the animation as video frames or not.
        :param animation_dir: None or a string, the directory where to save the plots for visualization.
        :param remove_particles_from_lists_if_hit: A Boolean, whether to remove particles from the lists if they are hit
            (True) or keep them and set the existence to False (False). If False, visualizes the not existent particles
            in grey. Should be only False for debugging purposes.

        :returns: The create_video_frame function. A function to be called in every time step and visualizes the current
            situation. See the documentation of create_video_frame. Its signature is
            create_video_frame(time_step, particle_pos, particle_class, actor_status), where
                - time_step: An integer, the current time step.
                - particle_pos: A np.array of shape [num_particles] or [num_particles, 2], the positions of the
                    particles.
                - particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep
                    the particle") or 1 ("eject the particle") representing the particle classes.
                - actor_status: An object np.array of shape [num_actors] representing the actors' current status.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def evaluate(self,
                 controller,
                 time_out=10,
                 no_show_animation=False,
                 save_animation=False,
                 animation_dir=None,
                 result_path=None,
                 with_logging=True,
                 ):
        """Evaluates the controller using the simulator.

        Evaluation includes an interactive animation of the controlled sorting system, counting successful and
        unsuccessful sorted particles, and measuring the computational times required by the controller.

        :param controller: A child instance of AbstractController, the controller to be applied to the task of sorting
            using an actuator array.
        :param time_out: An integer, the number of time steps after which the simulation will stop when no more
            particles have appeared.
        :param no_show_animation: A Boolean, whether to show the simulation as an interactive animation or not.
        :param save_animation: A Boolean, whether to save the animation as video frames or not.
        :param animation_dir: None or a string, the directory where the animation is saved.
        :param result_path: None or a string, the directory where the results are saved. If None, the results are not
            saved.
        :param with_logging: A Boolean, whether to log the controller's output or not.

        :returns:
            no_tns: An integer, the number of true negatives (correctly eject particles).
            no_tps: An integer, the number of true positives (correctly not-ejected particles).
            no_fns: An integer, the number of false negatives (falsely eject particles).
            no_fps: An integer, the number of false positives (falsely not-ejected particles).
            no_disturbed_ps: An integer, the number of disturbed positives.
            no_disturbed_ns: An integer, the number of disturbed negatives.
            tnr: A float, the true negative rate (TNR).
            tpr: A float, the true positive rate (TPR).
            computation_time_ls: A list for storing the computational times of the controller calls.
        """
        # TODO: Add a check for the controller, currently results in a circular import.
        # sanity checks
        if not np.isclose(np.mod(controller.T, self.T), 0):
            raise ValueError(
                'Controller sampling time intervals must be multiples of the simulator sampling time interval T.')
        if not isinstance(time_out, int) or time_out <= 0:
            raise ValueError(
                'The number of time steps after which the simulation will stop time_out must be a positive integer.')
        if save_animation and (animation_dir is None or not os.path.isdir(animation_dir)):
            raise ValueError('The animation directory must be a valid directory.')

        # build the animation function
        animation_func = self.show_action_fn_generator(
            particle_r=self._particle_simulator.particle_r if isinstance(self.particle_simulator,
                                                                         AreaParticleSimulator) else None,
            no_show_animation=no_show_animation,
            save_animation=save_animation,
            animation_dir=animation_dir,
        )

        controller_rate = controller.T / self.T

        no_tns = 0  # number of true negatives (correctly eject particles)
        no_tps = 0  # number of true positives (correctly not-ejected particles)
        no_fns = 0  # number of false negatives (falsely eject particles)
        no_fps = 0  # number of false positives (falsely not-ejected particles)
        no_disturbed_ns = 0  # number of disturbed negatives
        no_disturbed_ps = 0  # number of disturbed positives
        no_neg = 0  # number of negative particles
        no_pos = 0  # number of positive particles in this time step
        computation_time_ls = []  # list for computation times

        time_step = 0
        time_to_activate = np.empty(len(self._actors.pos))
        time_to_activate[:] = np.inf
        visible_particle_ids = np.array([], dtype=int)
        time_out_counter = 0
        while True:
            # increase_the time step
            time_step += 1

            # simulate the particles
            (time_to_activate, visible_particle_ids, particle_pos, particle_class, _, hits, disturbed_hits,
             particle_end_of_line, computation_time_ls) = self._simulate_one_step(
                time_step,
                controller=controller,
                controller_rate=controller_rate,
                time_to_activate=time_to_activate,  # this value must be fed back to the simulator
                visible_particle_ids=visible_particle_ids,  # this value must be fed back to the simulator
                computation_time_ls=computation_time_ls,
                animation_func=animation_func,
                with_logging=with_logging,
            )

            no_tns_ts = np.sum(np.logical_and(particle_class, hits))  # number of true negatives (correctly eject
            # particles) in this time step
            no_tps_ts = np.sum(np.logical_and(np.logical_not(particle_class), particle_end_of_line))  # number of true
            # positives (correctly not-ejected particles) in this time step
            no_fns_ts = np.sum(np.logical_and(np.logical_not(particle_class), hits))  # number of false negatives
            # (falsely eject particles) in this time step
            no_fps_ts = np.sum(np.logical_and(particle_class, particle_end_of_line))  # number of false positives
            # (falsely not-ejected particles) in this time step
            no_disturbed_ns_ts = np.sum(np.logical_and(particle_class, disturbed_hits))  # number of disturbed negatives
            # in this time step
            no_disturbed_ps_ts = np.sum(np.logical_and(np.logical_not(particle_class), disturbed_hits))  # number of
            # disturbed positives in this time step
            no_neg_ts = no_tns_ts + no_fps_ts + np.sum(np.logical_and(particle_class, disturbed_hits))  # number of
            # negative particles leaving the array in this time step
            no_pos_ts = no_tps_ts + no_fns_ts + np.sum(np.logical_and(np.logical_not(particle_class), disturbed_hits))
            # number of positive particles leaving the array in this time step

            no_tns += no_tns_ts
            no_tps += no_tps_ts
            no_fns += no_fns_ts
            no_fps += no_fps_ts
            no_disturbed_ns += no_disturbed_ns_ts
            no_disturbed_ps += no_disturbed_ps_ts
            no_neg += no_neg_ts
            no_pos += no_pos_ts

            tnr = float(np.divide(no_tns, no_neg, out=np.array(np.nan), where=no_neg != 0.0))
            tpr = float(np.divide(no_tps, no_pos, out=np.array(np.nan), where=no_pos != 0.0))

            # break conditions
            if particle_pos.size == 0:
                time_out_counter += 1
            else:
                time_out_counter = 0
            if time_out_counter >= time_out:
                logging.info(
                    'Evaluation result: TNs: {}, TPs: {}, FNs: {}, FPs: {}, '
                    'Disturbed Ns: {}, Disturbed Ps: {}, TNR: {}%, TPR: {}%.'.format(
                        no_tns,
                        no_tps,
                        no_fns,
                        no_fps,
                        no_disturbed_ns,
                        no_disturbed_ps,
                        np.round(tnr * 100, 1),
                        np.round(tpr * 100, 1),
                    ))

                computation_time_ls = computation_time_ls[1:]  # some jit compilers may corrupt first run
                med_time = np.round(np.median(computation_time_ls) * 1000, 1)
                third_quant_time = np.round(np.quantile(computation_time_ls, 0.75) * 1000, 1)
                max_time = np.round(np.max(computation_time_ls) * 1000, 1)
                logging.info(
                    'Controller computation times: median: {} ms, third quartile: {} ms, max: {} ms.'.format(
                        med_time,
                        third_quant_time,
                        max_time,
                    ))

                if result_path is not None:
                    # save lists (number of particles, percentage and timings) as combined csv
                    results = {}
                    results[
                        "number_of_particles"] = no_tps + no_tns + no_fps + no_fns + no_disturbed_ns + no_disturbed_ps
                    results["TNR"] = tnr
                    results["TPR"] = tpr
                    results["number_of_true_negatives"] = no_tns
                    results["number_of_true_positives"] = no_tps
                    results["number_of_false_negatives"] = no_fns
                    results["number_of_false_positives"] = no_fps
                    results["number_of_disturbed_negatives"] = no_disturbed_ns
                    results["number_of_disturbed_positives"] = no_disturbed_ps

                    results["median_time"] = med_time
                    results["third_quantile_time"] = third_quant_time
                    results["max_time"] = max_time
                    # results contains only scalar values, so wrap it in a list to build a single-row DataFrame.
                    df = pd.DataFrame([results])

                    dir_path = os.path.dirname(result_path)
                    if not os.path.isdir(dir_path):
                        os.makedirs(dir_path)
                    result_path = result_path if result_path.endswith('.csv') else result_path + '.csv'
                    df.to_csv(result_path)

                assert self._particle_simulator.num_particle_seen_so_far == self._particle_simulator.total_num_particles, \
                    'The number of particles seen during evaluation does not match the total number of particles.'
                assert no_tns + no_tps + no_fns + no_fps + no_disturbed_ns + no_disturbed_ps == self._particle_simulator.total_num_particles, \
                    'The number of particles counted during evaluation does not match the total number of particles.'
                assert no_neg + no_pos == self._particle_simulator.total_num_particles, \
                    ('The number of negative and positive particles counted during evaluation does not match the total '
                     'number of particles.')

                return no_tns, no_tps, no_fns, no_fps, no_disturbed_ns, no_disturbed_ps, tnr, tpr, computation_time_ls

    def create_buffered_initial_state_estimates_data_set_iterator(self,
                                                                  behavior_policy,
                                                                  num_particles_per_scenario,
                                                                  scenario_time_out,
                                                                  num_scenarios=None,
                                                                  num_examples=None,
                                                                  map_fn=None,
                                                                  progbar=True):
        """Returns an iterator that provides random initial state estimates for the particles in the simulation using
        the behavior policy on different scenarios.

        The returned state estimate distributions can be viewed as samples from the hyperparameter distribution of the
        initial state, i.e, samples of the motions state mean & covariance, actor state, and particle class for the
        initial time step (n = 0) collected from different scenarios. Therefore, the scenarios are generated randomly,
        i.e., the seed of the simulator is not reset.

        Note that it remains to the user to reset the initial scenario, including the original total number if particles
        and initial random state of the random number generator after calling this method, if desired, using
        simulator.reset(restore_original_scenario=True).

        If num_examples is not provided, the number of elements in the iterator is not fixed. It depends on the number
        of considered scenarios num_scenarios and, since, each scenario itself can provide a different number of
        samples, the number of particles that are moved and the duration of sampling time interval for each scenario. In
        general, the larger the number of particles and the smaller the sampling time interval, the more elements are in
        the iterator.

        :param behavior_policy: An AbstractController object, the behavior policy to use for generating the data.
        :param num_particles_per_scenario: An integer, the number of particles in each scenario.
        :param scenario_time_out: An integer, the number of time steps after which the simulation for one scenario will
            stop when no more particles have appeared.
        :param num_scenarios: None or an integer, the number of scenarios to consider for learning. Either num_scenarios
            or num_examples must be provided.
        :param num_examples: None or an integer, the number of examples to consider for learning. If provided,
            corresponds to the number of elements in the iterator. Either num_scenarios or num_examples must be provided.
        :param map_fn: None or a callable, a function that maps the elements of the iterator to new elements. Note that
            map_fn is applied before initial_states_iterator.reset(), i.e., unless calling map(some_other_map_fn,
            initial_states_iterator) which breaks the iterator.reset() call, map_fn_can be used in combination with
            initial_states_iterator.reset(). However, map_fn is not allowed to change the number and types of element
            of initial_states_iterator.
        :param progbar: A Boolean, whether to display a progress bar or not.

        :returns:
            initial_states_iterator: An InitialStatesIterator object, an iterator that yields
                estimated_motion_state_mean: A np.array of shape [num_particles, 2 or 4], the estimated motion state
                    mean.
                estimated_motion_state_cov: A np.array of shape [num_particles, 2 or 4, 2 or 4], the estimated motion
                    state covariance.
                particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
                    particle") or 1 ("eject the particle") representing the particle classes.
                particle_id: An integer np.array of shape [num_particles], the ids of the particles.
                existence: An integer np.array of shape [num_particles] where each component is either 0 ("particle does
                    not exist") or 1 ("particle exists") representing the existence of the particles.
                t_act: A np.array of shape [num_actors] representing the actors' current internal time.
                controls: A np.array of shape [num_actors, max_actions], the time to activate the actors as delta w.r.t.
                    the current time step.
            num_examples: An integer, the number of examples in the initial_states_iterator.
            scenario_len_list: A list of integers, the length of each scenario that was considered.
        """
        if self._particle_simulator.is_frozen:
            raise ValueError('Cannot use a frozen simulator for creating initial state estimates, since all created '
                             'state estimate sequence would be equal.')

        if self._particle_simulator.is_frozen:
            raise ValueError('Cannot use a frozen simulator for creating initial state estimates, since all created '
                             'state estimate sequence would be equal.')
        if (num_scenarios is None and num_examples is None) or (num_scenarios is not None and num_examples is not None):
            raise ValueError('Either num_scenarios or num_examples must be provided.')

        target = num_scenarios if num_scenarios is not None else num_examples
        unit_name = "scenario" if num_scenarios is not None else "example"
        logging.info(f'Collecting initial state estimate data set of {target} {unit_name}s.')
        progbar = tf.keras.utils.Progbar(target=target, unit_name=unit_name, verbose=1 if progbar else 0)

        scenario_counter = 0
        examples_counter = 0

        estimated_motion_state_mean_buffer = []
        estimated_motion_state_cov_buffer = []
        particle_class_buffer = []
        particle_id_buffer = []
        existence_buffer = []
        t_act_buffer = []
        controls_buffer = []

        scenario_len_list = []

        cont = True
        while cont:

            if num_scenarios is not None:
                if scenario_counter >= num_scenarios:
                    break
                progbar.add(1)

            self.reset(seed=None, restore_original_scenario=False)  # do not reset the seed; restore original
            # scenario does not make sense here
            self._particle_simulator.total_num_particles = num_particles_per_scenario  # sample a new random class list
            # with number_of_particles particles
            initial_states_iterator = self._create_initial_state_estimates_iterator(
                behavior_policy,
                time_out=scenario_time_out)

            scenario_len = 0
            for (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence, t_act,
                 controls) in initial_states_iterator:

                if num_examples is not None:
                    if examples_counter >= num_examples:
                        cont = False
                        break
                    progbar.add(1)

                if map_fn is not None:
                    (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                     t_act, controls) = map_fn((estimated_motion_state_mean, estimated_motion_state_cov, particle_class,
                                                particle_id, existence, t_act, controls))

                estimated_motion_state_mean_buffer.append(estimated_motion_state_mean)
                estimated_motion_state_cov_buffer.append(estimated_motion_state_cov)
                particle_class_buffer.append(particle_class)
                particle_id_buffer.append(particle_id)
                existence_buffer.append(existence)
                t_act_buffer.append(t_act)
                controls_buffer.append(controls)
                scenario_len += 1
                examples_counter += 1

            if scenario_len == 0:
                logging.warning(
                    f"Scenario {scenario_counter + 1} does not contain any particles. "
                    f"Try to increase the scenario time out.")

            scenario_len_list.append(scenario_len)
            scenario_counter += 1

        num_examples = len(estimated_motion_state_mean_buffer)

        class InitialStatesIterator:

            def __init__(self):
                self._index = -1

            def __iter__(self):
                return self

            def __getitem__(self, index):
                return (estimated_motion_state_mean_buffer[index],
                        estimated_motion_state_cov_buffer[index],
                        particle_class_buffer[index],
                        particle_id_buffer[index],
                        existence_buffer[index],
                        t_act_buffer[index],
                        controls_buffer[index])

            def __next__(self):
                if self._index + 1 >= len(estimated_motion_state_mean_buffer):
                    raise StopIteration
                self._index += 1
                return (estimated_motion_state_mean_buffer[self._index],
                        estimated_motion_state_cov_buffer[self._index],
                        particle_class_buffer[self._index],
                        particle_id_buffer[self._index],
                        existence_buffer[self._index],
                        t_act_buffer[self._index],
                        controls_buffer[self._index])

            def reset(self):
                self._index = -1

        logging.info('Number of examples in the initial state data set: {}'.format(num_examples))

        return InitialStatesIterator(), num_examples, scenario_len_list

    def create_initial_state_estimates_data_set_iterator(self,
                                                         behavior_policy,
                                                         num_particles_per_scenario,
                                                         scenario_time_out,
                                                         num_scenarios=None,
                                                         num_examples=None,
                                                         map_fn=None,
                                                         progbar=True):
        """Returns an iterator that provides random initial state estimates for the particles in the simulation using
        the behavior policy on different scenarios.

        The returned state estimate distributions can be viewed as samples from the hyperparameter distribution of the
        initial state, i.e, samples of the motions state mean & covariance, actor state, and particle class for the
        initial time step (n = 0) collected from different scenarios. Therefore, the scenarios are generated randomly,
        i.e., the seed of the simulator is not reset.

        Note that it remains to the user to reset the initial scenario, including the original total number if particles
        and initial random state of the random number generator after calling this method, if desired, using
        simulator.reset(restore_original_scenario=True).

        If num_examples is not provided, the number of elements in the iterator is not fixed. It depends on the number
        of considered scenarios num_scenarios and, since, each scenario itself can provide a different number of
        samples, the number of particles that are moved and the duration of sampling time interval for each scenario. In
        general, the larger the number of particles and the smaller the sampling time interval, the more elements are in
        the iterator.

        If both num_scenarios and num_examples are None, the iterator will potentially run indefinitely and needs to
        be stopped manually. If both are provided, will raise a ValueError.

        :param behavior_policy: An AbstractController object, the behavior policy to use for generating the data.
        :param num_particles_per_scenario: An integer, the number of particles in each scenario.
        :param scenario_time_out: An integer, the number of time steps after which the simulation for one scenario will
            stop when no more particles have appeared.
        :param num_scenarios: None or an integer, the number of scenarios to consider for learning.
        :param num_examples: None or an integer, the number of examples to consider for learning. If provided,
            corresponds to the number of elements in the iterator.
        :param map_fn: None or a callable, a function that maps the elements of the iterator to new elements. Note that
            map_fn is applied before initial_states_iterator.reset(), i.e., unless calling map(some_other_map_fn,
            initial_states_iterator) which breaks the iterator.reset() call, map_fn_can be used in combination with
            initial_states_iterator.reset(). However, map_fn is not allowed to change the number and types of element
            of initial_states_iterator.
        :param progbar: A Boolean, whether to display a progress bar or not.

        :returns:
            initial_states_iterator: An iterator that yields
                estimated_motion_state_mean: A np.array of shape [num_particles, 2 or 4], the estimated motion state
                    mean.
                estimated_motion_state_cov: A np.array of shape [num_particles, 2 or 4, 2 or 4], the estimated motion
                    state covariance.
                particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
                    particle") or 1 ("eject the particle") representing the particle classes.
                particle_id: An integer np.array of shape [num_particles], the ids of the particles.
                existence: An integer np.array of shape [num_particles] where each component is either 0 ("particle does
                    not exist") or 1 ("particle exists") representing the existence of the particles.
                t_act: A np.array of shape [num_actors] representing the actors' current internal time.
                controls: A np.array of shape [num_actors, max_actions], the time to activate the actors as delta w.r.t.
                    the current time step.
        """
        if self._particle_simulator.is_frozen:
            raise ValueError('Cannot use a frozen simulator for creating initial state estimates, since all created '
                             'state estimate sequence would be equal.')
        if num_scenarios is not None and num_examples is not None:
            raise ValueError('Both num_scenarios and num_examples cannot be provided at the same time.')

        if num_scenarios is not None:
            target, unit_name = num_scenarios, "scenario"
        elif num_examples is not None:
            target, unit_name = num_examples, "example"
        else:
            target, unit_name = None, "undefined"
        if num_scenarios is not None or num_examples is not None:
            logging.info(f'Collecting initial state estimate data set of {target} {unit_name}s.')

        # with target=None, progbar will automatically not be display
        progbar = tf.keras.utils.Progbar(target=target, unit_name=unit_name, verbose=1 if progbar else 0)

        scenario_counter = 0
        examples_counter = 0
        while True:

            if num_scenarios is not None:
                if scenario_counter >= num_scenarios:
                    return  # stop the generator
                progbar.add(1)

            self.reset(seed=None, restore_original_scenario=False)  # do not reset the seed; restore original
            # scenario does not make sense here
            self._particle_simulator.total_num_particles = num_particles_per_scenario  # sample a new random class list
            # with number_of_particles particles
            initial_states_iterator = self._create_initial_state_estimates_iterator(
                behavior_policy,
                time_out=scenario_time_out)

            scenario_len = 0
            for (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence, t_act,
                 controls) in initial_states_iterator:

                if num_examples is not None:
                    if examples_counter >= num_examples:
                        return  # stop the generator
                    progbar.add(1)

                if map_fn is not None:
                    (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                     t_act, controls) = map_fn((estimated_motion_state_mean, estimated_motion_state_cov, particle_class,
                                                particle_id, existence, t_act, controls))

                yield (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                       t_act, controls)

                scenario_len += 1
                examples_counter += 1

            if scenario_len == 0:
                logging.warning(
                    f"Scenario {scenario_counter + 1} does not contain any particles. "
                    f"Try to increase the scenario time out.")

            scenario_counter += 1

    def _simulate_one_step(self,
                           time_step,
                           controller,
                           controller_rate,
                           time_to_activate,
                           visible_particle_ids,
                           computation_time_ls,
                           animation_func,
                           remove_particles_from_lists_if_hit=True,
                           with_logging=True):
        """Simulates the particles and actors, i.e., creates particles, moves and removes them if they are hit or reach
         the end of the line and activates and moves the actors.

        :param time_step: An integer, the current time step. Note that the initial time step (the time step of the first
            call of this method per scenario) should be 1. The method nonetheless calls the controller for calculating
            the controls for time step 0.
        :param controller: A child instance of AbstractController, the controller to be applied to the task of sorting
            using an actuator array.
        :param controller_rate: An integer, the number of simulated time steps after which the controller is called,
            i.e., the rate controller.T / self.T at which the simulator is (time) up-sampled compared with the
            controller.
        :param time_to_activate: A np.array of shape [num_actors, controller._max_actions] containing the time
            at which the actors should be activated. This array is provided by previous function calls.
        :param visible_particle_ids: An integer np.array of shape [num_particles], the ids of the particles that were
            visible. This array is provided by previous function calls.
        :param computation_time_ls: A list for storing the computational times of the controller calls. This array is
            provided by previous function calls.
        :param animation_func: A function to be called in every time step and visualizes the current situation.
        :param remove_particles_from_lists_if_hit: A Boolean, whether to remove particles from the lists if they are hit
            (True) or keep them and set the existence to False (False). Should be True, if the simulation is used for
            visualization purposes and False if the simulation is used for learning a controller that internally uses
            an existence state.
        :param with_logging: A Boolean, whether to log the controller's output or not.

        :returns:
            time_to_activate: A np.array of shape [num_actors, controller._max_actions] containing the time at which the
                actors should be activated.
            visible_particle_ids: An integer np.array of shape [num_particles], the ids of the particles that were
                visible.
            particle_pos: A np.array of shape [num_particles], the positions of the particles.
            particle_class: An integer np.array [num_particles] where each component is either 0 ("keep the particle")
                or 1 ("eject the particle") representing the particle classes.
            particle_existence: A Boolean np.array of shape [num_particles], the existence of the particles. True
                indicates that a particle exists.
            hits: A Boolean np.array [num_particles] where each component is either 0 ("particle was not hit") or
                1 ("particle was hit") representing whether the particles were ejected in time step under consideration.
            disturbed_hits: Boolean np.array [num_particles] where each component is either 0 ("particle was not
                disturbed") or 1 ("particle was disturbed") representing whether the particles were hit by an actor not
                in the intended HIT-status (i.e., it is in the UP- or DOWN-status) in time step under consideration.
            particle_end_of_line: Boolean np.array [num_particles] where each component is either 0 ("particle did not
                cross the end of the line") or 1 ("particle crossed the end of the line) representing whether left the
                actuator array in time step under consideration.
            computation_time_ls: A list for storing the computational times of the controller calls.
        """
        # calculate controls u_{k-1} for going from k-1 to k
        # this still belongs to the former time step k-1

        # calculate new control inputs if the controller rate is reached
        if (time_step - 1) % controller_rate == 0:
            if with_logging:
                logging.info('Calculating controls for time step: {}'.format(time_step - 1))
            start_time = time.time()
            # Controllers without an mtt_tracker attribute (e.g. thin behavior-policy wrappers in tests)
            # are treated as oracle / simulation mode for backward compatibility.
            mtt_tracker = getattr(controller, 'mtt_tracker', None)
            if mtt_tracker is None or isinstance(mtt_tracker, AbstractParticleSimulator):
                # Oracle / simulation mode: use ground-truth particle states from the simulator's particle model.
                time_to_activate = (time_step - 1) * self.T + controller.control_from_measurements(
                    measurements=None,
                    mtt_tracker=self._particle_simulator,
                    actors=self._actors)
            else:
                # Real MTT mode: generate (noisy) measurements and use the controller's configured tracker.
                measurements = self._particle_simulator.measure_particles()
                time_to_activate = (time_step - 1) * self.T + controller.control_from_measurements(
                    measurements=measurements,
                    mtt_tracker=mtt_tracker,
                    actors=self._actors)
            # In oracle mode we do not pass measurements, but use track_particles of the particle simulator as a
            # stand-in for the controller's internal tracker.
            computation_time_ls.append(time.time() - start_time)

            # now make a check to ensure that the combination of settings of the simulator and the controller do not
            # cause overlooking some particles
            particle_id = self._particle_simulator.particle_id
            first_time_seen_particles_mask = np.logical_not(np.isin(particle_id, visible_particle_ids))
            visible_particle_ids = particle_id.copy()
            position = self._particle_simulator.position[first_time_seen_particles_mask]
            velocity = self._particle_simulator.velocity[first_time_seen_particles_mask]
            actor_pos = self._actors.pos
            if isinstance(self._particle_simulator, AreaParticleSimulator):
                position = position[:, 0]  # only consider x position
                velocity = velocity[:, 0]  # only consider x velocity
                actor_pos = actor_pos[:, 0]  # only consider x position
            delta_toa_first_actor = (np.min(
                actor_pos) - position) / velocity  # time to reach the actors for the first time seen particles
            if np.any(delta_toa_first_actor < self._actors.t_lead):
                logging.warning(
                    'Some particles are seen for the first time when its already too late to activate the actors for '
                    'them. '
                    'Consider decreasing the controller rate or shifting the actors in the simulation to the right.')

        # activate actors, this corresponds to u_{k-1}
        actors_to_activate = np.any(
            np.logical_and(time_to_activate >= (time_step - 1) * self.T, time_to_activate < time_step * self.T),
            axis=1)  # if at least one particle needs to be shot out

        # go from k-1 to k, i.e., predict according to the (abstract) system model x_k = a_{k-1}(x_{k-1}, u_{k-1})
        # the results belong to the current time step.
        # move_existing particles
        if time_step != 0:
            self._particle_simulator.move_particles()

        # update actors
        self._actors.predict_own_actor_state(actors_to_activate)

        # create particles
        self._particle_simulator.create_particles()

        particle_pos = self._particle_simulator.position
        particle_class = self._particle_simulator.particle_class
        particle_existence = self._particle_simulator.particle_existence
        actor_status = self._actors.s_act

        self._contact_simulator.update_particle_pos(particle_pos)

        # check if they are hit (intentionally or accidentally)
        hits, disturbed_hits = self._contact_simulator.calculate_contact(
            particle_pos,
            self._particle_simulator.particle_existence,
            actor_status,
        )

        # check if particles reach the end of the line
        particle_end_of_line = np.logical_and.reduce((
            particle_pos[:, 0] >= self._array_end if particle_pos.ndim == 2 else particle_pos >= self._array_end,
            np.logical_not(hits),
            np.logical_not(disturbed_hits)))

        # remove particles
        if remove_particles_from_lists_if_hit:
            # remove particles that are hit (intentionally or accidentally) or reach the end of the line
            remove_mask = np.logical_or.reduce((hits, disturbed_hits, particle_end_of_line))
        else:
            self._particle_simulator.negate_existence(np.logical_or(hits, disturbed_hits))
            remove_mask = particle_end_of_line  # only remove particles that reach the end of the line
        self._particle_simulator.remove_particles(remove_mask)
        self._contact_simulator.remove_particle_pos(remove_mask)

        # update the simulation
        animation_func(time_step, particle_pos, particle_class, particle_existence, disturbed_hits, actor_status)

        return (time_to_activate, visible_particle_ids, particle_pos, particle_class, particle_existence, hits,
                disturbed_hits, particle_end_of_line, computation_time_ls)

    def _create_initial_state_estimates_iterator(self,
                                                 behavior_policy,
                                                 time_out=10,
                                                 no_show_animation=True,
                                                 ):
        """Returns an iterator that provides the initial state estimates for the particles in the simulation using the
        behavior policy.

        The returned state estimate distributions can be viewed as samples from the hyperparameter distribution of the
        initial state, i.e, samples of the motions state mean & covariance, actor state, and particle class for the
        initial time step (n = 0).

        The number of elements in the iterator is not fixed, but depends on the number of particles that are moved and
        the duration of sampling time interval. In general, the larger the number of particles and the smaller the
        sampling time interval, the more elements are in the iterator.

        :param behavior_policy: An AbstractController object, the behavior policy to use for generating the data.
        :param time_out: An integer, the number of time steps after which the simulation will stop when no more
            particles have appeared.
        :param no_show_animation: A Boolean, whether to show the simulation as an interactive animation or not. Should
            be only False for debugging purposes.

        :returns: An iterator that yields
            estimated_motion_state_mean: A np.array of shape [num_particles, 2 or 4], the estimated motion state
                mean.
            estimated_motion_state_cov: A np.array of shape [num_particles, 2 or 4, 2 or 4], the estimated motion
                state covariance.
            particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
                particle") or 1 ("eject the particle") representing the particle classes.
            particle_id: An integer np.array of shape [num_particles], the ids of the particles.
            existence: An integer np.array of shape [num_particles] where each component is either 0 ("particle does
                not exist") or 1 ("particle exists") representing the existence of the particles.
            t_act: A np.array of shape [num_actors] representing the actors' current internal time.
            controls: A np.array of shape [num_actors, max_actions], the time to activate the actors as delta w.r.t. the
                current time step.
        """
        controller_rate = behavior_policy.T / self.T

        time_step = 0
        time_to_activate = np.empty(len(self._actors.pos))
        time_to_activate[:] = np.inf
        visible_particle_ids = np.array([], dtype=int)
        time_out_counter = 0

        # build the animation function
        if not no_show_animation:
            animation_func = self.show_action_fn_generator(
                particle_r=self._particle_simulator.particle_r if isinstance(self.particle_simulator,
                                                                             AreaParticleSimulator) else None,
                no_show_animation=False,
                save_animation=False,
                animation_dir=None,
                remove_particles_from_lists_if_hit=False,
            )
        else:
            animation_func = lambda *args: None

        while True:
            time_step += 1

            # get the states for time step - 1 (the controller is called for time step - 1, see _simulate_one_step)
            (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id,
             existence) = self._particle_simulator.track_particles(measurements=None)
            t_act = self._actors.t_act.copy()  # copy is important here since the next line changes actors.t_act
            # otherwise
            existence = existence.copy()  # copy is important here since the next line changes
            # particle_simulator.particle_existence otherwise

            # simulate the particles and get the controls
            (time_to_activate, visible_particle_ids, _, p_class_after_step, p_ex_after_step, hits, disturbed_hits,
             particle_end_of_line, _) = self._simulate_one_step(
                time_step,
                controller=behavior_policy,
                controller_rate=controller_rate,
                time_to_activate=time_to_activate,  # this value must be fed back to the simulator
                visible_particle_ids=visible_particle_ids,  # this value must be fed back to the simulator
                computation_time_ls=[],
                animation_func=animation_func,
                remove_particles_from_lists_if_hit=False,
                with_logging=False,
            )

            no_fns_ts = np.sum(np.logical_and(np.logical_not(p_class_after_step), hits))  # number of false negatives
            # (falsely eject particles) in this time step
            no_fps_ts = np.sum(np.logical_and.reduce((p_ex_after_step, p_class_after_step, particle_end_of_line)))
            # number of false positives (falsely not-ejected particles) in this time step
            dist_hits = np.sum(disturbed_hits)  # number of disturbed particles in this time step
            if dist_hits > 0:
                logging.debug(f'{dist_hits} particle(s) was/were disturbed.')
            if no_fns_ts > 0:
                logging.debug(f'{no_fns_ts} particle(s) was/were falsely ejected.')
            if no_fps_ts > 0:
                logging.debug(f'{no_fps_ts} particle(s) was/were falsely not ejected.')

            # go back to delta times
            controls = time_to_activate - (time_step - 1) * self.T  # the controller was called for time step - 1
            existence = existence.astype(int)

            # break conditions
            if particle_class.size == 0:
                time_out_counter += 1
            else:
                time_out_counter = 0
                yield (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id, existence,
                       t_act, controls)
            if time_out_counter >= time_out:
                # reset the simulator
                self.reset(seed=None, restore_original_scenario=False)  # do not reset the seed; restore original
                # scenario does not make sense here
                break

    def reset(self, seed=None, restore_original_scenario=False):
        """Resets the actors, particle and contact simulators to their initial state.

        :param seed: None or an integer, the seed for the random number generator. If given, the random number
            generator is reset to seed. If None, the random number generator is not reset. This option is only available
            if restore_original_scenario and self.particle_simulator._freeze_scenario are False.
        :param restore_original_scenario: A Boolean, if True, the scenario, including the total number of particles and
            the seed for the random number generator are reset to the initial value, i.e., the run after resetting the
            simulator will be the same as the first run. If False and self.particle_simulator._freeze_scenario is False,
            the scenario is not fixed and the run will be different from the ones before. This option is only available
            if seed is None.
        """
        self._actors.reset()
        self._particle_simulator.reset(seed=seed, restore_original_scenario=restore_original_scenario)
        self._contact_simulator.reset()

    def shallow_copy(self, seed=None, freeze_scenario=False):
        """Creates a shallow copy of the simulator, i.e., a new instance of the simulator with the same settings but
        independent, freshly-initialized states.

        Note that the copy will be initialized with the current self.particle_simulator.total_num_particles particles,
        i.e., after resetting the copy with reset(restore_original_scenario=True), the copy will by default work with
        self.particle_simulator.total_num_particles particles.

        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        :param freeze_scenario: A Boolean, if True, the scenario, including the total number of particles and the seed
            for the random number generator are fixed to the initial value, i.e., each time after resetting the
            simulator, the same settings are restored and the scenario is generated. If False, the scenario is not
            frozen and a new run after resetting is allowed to be different from the ones before.

        :returns: A shallow copy of the simulator.
        """
        actors = self._actors.shallow_copy()
        cl = type(self)
        new_obj = cl.__new__(cl)
        super(cl, new_obj).__init__(
            particle_simulator=self._particle_simulator.shallow_copy(seed=seed,
                                                                     freeze_scenario=freeze_scenario),
            actors=actors,
            contact_simulator=self._contact_simulator.shallow_copy(actors),
            array_end=self._array_end,
            **self._original_kwargs,
        )
        return new_obj


class AreaSortingSimulator(AbstractSortingSimulator):
    """A discrete-time simulator for a (potentially controlled) sorting machine with a multi-array actor grid (2D actor
     arrangement).

    """

    def __init__(self,
                 particle_simulator,
                 actors,
                 contact_simulator_dict,
                 array_end=None,
                 physical_actor_offset_length_width=None,
                 ):
        """Initializes the simulator.

         Format of physical_actor_offset_length_width:

            [x_offset, y_offset, length, width]

        :param particle_simulator: An AreaParticleSimulator instance, the particle simulator to be used for simulating
            the particles movement and track birth and death.
        :param actors: An ActorSimulator instance, the actor simulator to be used for simulating the actors.
        :param contact_simulator_dict: A dict of kwargs with settings for AreaContactSimulator.
        :param array_end: None or a float, the end coordinate of the actuator array. Defaults to the position of the
            last actor.
        :param physical_actor_offset_length_width: None or a tuple of length 4 representing the offset of the actor
            position and length and width of the actors for visualization. If None, the disturbance areas from the
            contact model are used.
        """
        if not isinstance(particle_simulator, AreaParticleSimulator):
            raise ValueError('particle_simulator must be an AreaParticleSimulator object.')
        if physical_actor_offset_length_width is not None and (len(physical_actor_offset_length_width) != 4 or np.any(
                np.asarray(physical_actor_offset_length_width)[2:] <= 0)):
            raise ValueError('The offsets, length and width of the actors (for visualization) '
                             'physical_actor_offset_length_width must be of length 4 '
                             'and length and width must be positive.')

        contact_simulator = AreaContactSimulator(actors=actors,
                                                 **contact_simulator_dict)

        additional_attributes = {'_actor_center_for_visualization': contact_simulator.disturbance_area_center,
                                 '_actor_extent_for_visualization': contact_simulator.disturbance_area_extent}
        if physical_actor_offset_length_width is not None:
            additional_attributes['_actor_center_for_visualization'] = actors.pos + np.asarray(
                physical_actor_offset_length_width)[None, :2]
            additional_attributes['_actor_extent_for_visualization'] = np.ones_like(actors.pos) * np.asarray(
                physical_actor_offset_length_width)[None, 2:]

        super().__init__(particle_simulator=particle_simulator,
                         actors=actors,
                         contact_simulator=contact_simulator,
                         array_end=array_end,
                         **additional_attributes,
                         )


    def show_action_fn_generator(self,
                                 particle_r=None,
                                 no_show_animation=False,
                                 save_animation=False,
                                 animation_dir=None,
                                 remove_particles_from_lists_if_hit=True,
                                 **kwargs):
        """Build a function that can be called in every time step and visualizes the current situation as interactive
         animation of the simulation in a video stream.

        :param particle_r: None or a float, the radius of the particles for visualization. Defaults to 1/12 of the
            distance between the first and second actor.
        :param no_show_animation: A Boolean, whether to show the animation or not.
        :param save_animation: A Boolean, whether to save the animation as video frames or not.
        :param animation_dir: None or a string, the directory where to save the plots for visualization.
        :param remove_particles_from_lists_if_hit: A Boolean, whether to remove particles from the lists if they are hit
            (True) or keep them and set the existence to False (False). If False, visualizes the not existent particles
            in grey. Should be only False for debugging purposes.

        :returns: The create_video_frame function. A function to be called in every time step and visualizes the current
            situation. See the documentation of create_video_frame.
        """
        # sanity checks
        if save_animation and animation_dir is None:
            raise ValueError('If save_animation is true, an animation_dir must be provided.')

        particle_r = self._particle_simulator.area_width / 12 if particle_r is None else particle_r

        actor_color_map = {ActorStatus.READY: 'green',
                           ActorStatus.ACTIVATE: 'yellow',
                           ActorStatus.UP: 'orange',
                           ActorStatus.HIT: 'purple',
                           ActorStatus.DOWN: 'orange',
                           ActorStatus.RESET: 'red'}

        transformed_positions = self._actor_center_for_visualization - self._actor_extent_for_visualization / 2

        handles_particle_class = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                                         markerfacecolor='red', markersize=15),
                                  Line2D([0], [0], marker='o', color='w', label='Scatter',
                                         markerfacecolor='blue', markersize=15)]
        labels_particle_class = ['Eject', 'Keep']
        if save_animation:
            # the additional handle/label interferes with the plot (it wobbles), so we only add it when saving
            handles_particle_class.append(Line2D([0], [0], marker='o', color='w', label='Scatter',
                                                 markerfacecolor='orange', markersize=15))
            labels_particle_class.append('Disturbed')
            if not remove_particles_from_lists_if_hit:
                handles_particle_class.append(Line2D([0], [0], marker='o', color='w', label='Scatter',
                                                     markerfacecolor='grey', markersize=15))
                labels_particle_class.append('Not exist')

        handles_actor_state = [Line2D([0], [0], marker='s', color='w', label='Scatter',
                                      markerfacecolor='green', markersize=15),
                               Line2D([0], [0], marker='s', color='w', label='Scatter',
                                      markerfacecolor='yellow', markersize=15),
                               Line2D([0], [0], marker='s', color='w', label='Scatter',
                                      markerfacecolor='orange', markersize=15),
                               Line2D([0], [0], marker='s', color='w', label='Scatter',
                                      markerfacecolor='purple', markersize=15),
                               Line2D([0], [0], marker='s', color='w', label='Scatter',
                                      markerfacecolor='red', markersize=15)]
        labels_actor_state = ['Ready', 'Activate', 'Up/Down', 'Hit', 'Reset']

        def create_video_frame(time_step,
                               particle_pos,
                               particle_class,
                               particle_existence,
                               disturbed_hits,
                               actor_status):
            """Shows an interactive animation of the simulation in a video stream.

            The animation of the sorting process is interactive. The sorting progress can be interrupted by pressing `x`
            (after clicking into the animation), which activates a manual _press mode_. In press mode an arbitrary key
            (except of `y`) can be pressed to show the next frame. To exit the press mode and return to flush mode,
            press `y`.

            We show the plot like a video stream, when it is updated, the new plot overwrites the old in the same figure.
            See https://www.delftstack.com/de/howto/matplotlib/how-to-automate-plot-updates-in-matplotlib/ .

            :param time_step: An integer, the current time step.
            :param particle_pos: A np.array of shape [num_particles, 2], the positions of the particles.
            :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep
                the particle") or 1 ("eject the particle") representing the particle classes.
            :param particle_existence: A Boolean np.array of shape [num_particles], the existence of the particles. True
                indicates that a particle exists.
            :param disturbed_hits: A Boolean np.array of shape [num_particles], a mask for which particles are disturbed
                by the actors.
            :param actor_status: An object np.array of shape [num_actors] representing the actors' current status.
            """
            # add actors
            for i, pos in enumerate(transformed_positions):
                rect = plt.Rectangle((pos[0], pos[1]),
                                     width=self._actor_extent_for_visualization[i, 0],
                                     height=self._actor_extent_for_visualization[i, 1],
                                     color=actor_color_map[actor_status[i]])
                # the patches could get added faster if using patch collections or poly collections
                plt.gca().add_patch(rect)

            # add particles
            for p, c, e, d in zip(particle_pos, particle_class, particle_existence, disturbed_hits):
                if d:
                    color = 'orange'
                elif not e:
                    color = 'grey'
                else:
                    color = 'red' if c else 'blue'
                circle = plt.Circle((p[0], p[1]), particle_r, color=color)
                plt.gca().add_patch(circle)

            plt.gca().set_xlim(0, self._array_end)
            plt.gca().set_ylim(0, self._particle_simulator.area_width)
            plt.gca().set_aspect('equal')
            plt.title('Time step ' + str(time_step))
            plt.xlabel('x-coordinate')
            plt.ylabel('y-coordinate')

            # Put legends on the right to current axis
            plt.gcf().set_size_inches((6.4, 4.8))  # default matplotlib size, can be changed if needed
            particle_class_legend = plt.legend(title='Particle class', loc='upper left', bbox_to_anchor=(1.04, 1),
                                               handles=handles_particle_class,
                                               labels=labels_particle_class)
            plt.gca().add_artist(particle_class_legend)
            plt.legend(title='Actor state', loc='lower left', bbox_to_anchor=(1.04, 0),
                       handles=handles_actor_state, labels=labels_actor_state)
            if not save_animation:
                # tight_layout() displays the legend properly in the figure, but manipulates its size. Therefore, we
                # only use it when not saving the figure at the expense that the displayed (not the saved) legends can
                # be cut off when the plots are additionally saved
                plt.tight_layout()

            # save images
            if save_animation:
                plt.rcParams[
                    'savefig.bbox'] = 'tight'  # figsize (without labels) as defined, add extra space for the labels
                path = os.path.join(animation_dir, '{:05d}'.format(time_step))
                plt.savefig(path)
                plt.rcParams['savefig.bbox'] = None  # undo the change, None is default
                msg = 'Figure saved at {}'.format(path)
                logging.info(msg)
                if no_show_animation:
                    plt.close()

        if no_show_animation and not save_animation:
            return lambda *args: None
        if no_show_animation:
            return create_video_frame
        else:
            return flush_figure_generator(create_video_frame)
