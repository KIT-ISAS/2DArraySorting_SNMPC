import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from absl import logging

import json
from importlib.metadata import version

import numpy as np

from scenario_helpers import avg_num_time_steps_particle_in_range
from mtt_trackers import DummyMTTTracker
from actors import VirtualActor
from heuristics import FirstActorFirstHeuristicController
from ptcr_controller import PTCRController
from ptcr_config import pop_ptcr_nested_config

try:
    TF_VERSION = version('tensorflow')
except Exception:
    TF_VERSION = 'unknown'


class CATemporalMotionModel:
    """A temporal motion model for the GaussTaylor or Uniform model."""

    @staticmethod
    def predict(pos_last, v_last, a_last, x_pred_to):
        """Predicts a constant-acceleration (CA) travel time to a specified arrival position.

        :param pos_last: A np.array of shape [num_particles, 2], the last known positions.
        :param v_last: A np.array of shape [num_particles, 2], the last known velocities.
        :param a_last: A np.array of shape [num_particles, 2], the last known accelerations.
        :param x_pred_to: A np.array of shape [num_actors, 2], the arrival positions to which the travel time is to be
            predicted.

        :returns: A np.array of shape [num_actors, num_particles], the predicted time of arrival as delta w.r.t. the
            current time step.
        """
        x_pred_to = x_pred_to[:, 0, None]

        # we use np.float32 to avoid numerical issues when close to zero
        delta_x = np.subtract(x_pred_to, pos_last[..., 0], dtype=np.float32)
        a_last[a_last == 0] = np.finfo(float).eps  # to avoid division by zero
        sqrt_val = (np.divide(v_last[..., 0], a_last[..., 0], dtype=np.float32)) ** 2 + 2 * np.divide(delta_x,
                                                                                                      a_last[..., 0],
                                                                                                      dtype=np.float32)
        sqrt_val[sqrt_val < 0] = 0.0  # to avoid roots of negative numbers
        dt_pred = - np.divide(v_last[..., 0], a_last[..., 0], dtype=np.float32) + np.sign(a_last[..., 0]) * np.sqrt(
            sqrt_val)
        return dt_pred


class CASpatialMotionModel:
    """A spatial motion model for the GaussTaylor or Uniform model."""

    def __init__(self, upper_boundary=None, lower_boundary=None):
        """Initializes the spatial motion model.

        :param upper_boundary: None or float, the upper boundary for the y position. If None, there is no upper boundary.
        :param lower_boundary: None or float, the lower boundary for the y position. If None, there is no lower boundary.
        """
        if upper_boundary is not None and not isinstance(upper_boundary, (int, float)):
            raise ValueError("upper_boundary should be a float or int if provided.")
        if lower_boundary is not None and not isinstance(lower_boundary, (int, float)):
            raise ValueError("lower_boundary should be a float or int if provided.")
        if upper_boundary is not None and lower_boundary is not None and upper_boundary <= lower_boundary:
            raise ValueError("upper_boundary should be greater than lower_boundary.")

        self.upper_boundary = upper_boundary
        self.lower_boundary = lower_boundary

    def predict(self, pos_last, v_last, a_last, dt_pred):
        """Predicts the y position of the particles after a predicted travel time, mirroring wall collisions.

        :param pos_last: A np.array of shape [num_particles, 2], the last known positions (position component is not used for the
            prediction itself, only for mirroring).
        :param v_last: A np.array of shape [num_particles, 2], the velocities in x and y direction.
        :param a_last: A np.array of shape [num_particles, 2], the accelerations in x and y direction.
        :param dt_pred: A np.array of shape [num_actors, num_particles], the predicted time of arrival as delta w.r.t.
            the current time step.

        :returns: A np.array of shape [num_actors, num_particles], the particles' position orthogonal to the
            transport direction at the time of arrival at the actors.
        """
        y_pred = v_last[..., 1] * dt_pred + 1 / 2 * dt_pred ** 2 * a_last[..., 1]
        return self.mirror_wall_collisions(y_pred)

    def mirror_wall_collisions(self, y_pred):
        """Checks if the predicted y positions exceed the boundaries and mirrors them if so.

        :param y_pred: A np.array of shape [num_actors, num_particles], the particles' position orthogonal to the
            transport direction at the time of arrival at the actors.

        :returns: A np.array of shape [num_actors, num_particles], the particles' position orthogonal to the
            transport direction at the time of arrival at the actors, after mirroring wall collisions.
        """
        if self.upper_boundary is not None:
            beyond_upper_mask = y_pred > self.upper_boundary
            y_pred[beyond_upper_mask] = 2 * self.upper_boundary - y_pred[beyond_upper_mask]

        if self.lower_boundary is not None:
            below_lower_mask = y_pred < self.lower_boundary
            y_pred[below_lower_mask] = 2 * self.lower_boundary - y_pred[below_lower_mask]
        return y_pred


def ca_motion_model_wrapper_for_heuristics(ca_motion_model, motion_state, *args, **kwargs):
    """A wrapper for the ca_motion_model to be used in the heuristics.

    Splits the (interleaved) motion state of the form
    ``[x, vx, ax, y, vy, ay]`` into separate position, velocity and acceleration arrays that are expected by the
    constant-acceleration motion model.

    :param ca_motion_model: A callable, the constant-acceleration motion model to wrap.
    :param motion_state: A np.array of shape [..., 6], the particle motion state with layout ``[x, vx, ax, y, vy, ay]``.

    :returns: A np.The output of ca_motion_model.
    """
    p, v, a = motion_state[..., [0, 3]], motion_state[..., [1, 4]], motion_state[..., [2, 5]]
    return ca_motion_model(p, v, a, *args, **kwargs)


def select_min_per_non_unique_value(a, b):
    """Selects the indices of a where b is minimal if the corresponding value is not unique. If the value is unique,
     it is selected regardless of b.

    :param a: A np.array of shape [num_elements] containing the values to check for uniqueness.
    :param b: A np.array of shape [num_elements] containing the values to check for minimality.

    :returns: A Boolean np.array of shape [num_elements], a mask that indicates that the corresponding index is
        selected.
    """
    unique_a, inv = np.unique(a, return_inverse=True)
    # compute minimum b per unique a
    min_b = np.full(len(unique_a), np.inf)
    np.minimum.at(min_b, inv, b)
    # select indices where b equals the minimum
    mask = b == min_b[inv]
    return mask


class ControllerWrapper:
    """A wrapper for the controllers that handles the interaction with the actor model and returns the control actions
    in the format required by the sorting system.

    """

    def __init__(self, controller, actors):
        """Initializes the ControllerWrapper.

        :param controller: An AbstractController object, a controller that can be used to control the sorting system.
        :param actors: A VirtualActor object, the virtual actors to use for predicting the actor states.
        """

        if not isinstance(actors, VirtualActor):
            raise ValueError("actor_model should be an instance of VirtualActor.")

        self.controller = controller

        self.actors = actors

        # track the activations
        self._activations = np.empty((0, 3), dtype=float)  # each row is (predicted_time, actuator_idx,
        # time_step_when_predicted)

    def __call__(self, particle_states, particle_class, particle_id, time_step):
        """Runs one step of the controller given the current states.

        Format particle_states:

            [estimated_motion_state_mean, estimated_motion_state_cov, track_score]

            with estimated_motion_state_mean being a np.array of shape [num_particles, 4] and format [position_x,
            velocity_x, position_y, velocity_y], in case of a 2D optical sorter (grid sorter) containing the expected
            values of the motion state and estimated_motion_state_cov a np.array of shape [num_particles, 4, 4]
            containing the covariances of the motion state. The track_score is a np.array of shape [num_particles] used
            as the existence probabilities of the particles.

        :param particle_states: A tuple of length 3.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        :param time_step: An integer, the current time step of the system.

        :returns:
            predicted_time: A np.array of shape [num_actuators_to_activate], the predicted time of particles' arrival
                in seconds (actuator time).
            actuator_idxs: An integer np.array of shape [num_actuators_to_activate], the actuators to use.
        """
        return self.run_one_step(particle_states, particle_class, particle_id, time_step)

    def run_one_step(self, particle_states, particle_class, particle_id, time_step):
        """Runs one step of the controller given the current states.

        Format particle_states:

            [estimated_motion_state_mean, estimated_motion_state_cov, track_score]

            with estimated_motion_state_mean being a np.array of shape [num_particles, 4] and format [position_x,
            velocity_x, position_y, velocity_y], in case of a 2D optical sorter (grid sorter) containing the expected
            values of the motion state and estimated_motion_state_cov a np.array of shape [num_particles, 4, 4]
            containing the covariances of the motion state. The track_score is a np.array of shape [num_particles] used
            as the existence probabilities of the particles.

        :param particle_states: A tuple of length 3.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        :param time_step: An integer, the current time step of the system.

        :returns:
            predicted_time: A np.array of shape [num_actuators_to_activate], the predicted time of particles' arrival
                in seconds (actuator time).
            actuator_idxs: An integer np.array of shape [num_actuators_to_activate], the actuators to use.
        """
        if isinstance(self.controller, FirstActorFirstHeuristicController):
            particle_states = (particle_states[0],)

        activation_delta_t = self.controller.control_from_states(particle_states,
                                                                 particle_class,
                                                                 particle_id,
                                                                 t_act=self.actors.t_act,
                                                                 )
        activation_mask = np.isfinite(activation_delta_t)
        predicted_delta_time = activation_delta_t[activation_mask]

        # prepare the return values
        predicted_time = predicted_delta_time + time_step * self.controller.T
        actuator_idxs = np.nonzero(activation_mask)[0]

        # update the list of activations
        self._activations = np.concatenate(
            (self._activations,
             np.stack((predicted_time, actuator_idxs, time_step * np.ones_like(predicted_time)), axis=1)), axis=0)

        # get the actors to be activated in the current time step
        actors_activation_mask = np.any(
            np.logical_and(self._activations[:, 0] >= (time_step - 1) * self.controller.T,
                           self._activations[:, 1] < time_step * self.controller.T))
        a_times, a_idxs, ts_when_predicted = self._activations[actors_activation_mask].T
        # if there are multiple activations for the same actor in the current time step, we only use the one with the
        # latest time step, i.e., the most recent one
        if a_idxs.size > 0:
            actor_to_activate_mask = select_min_per_non_unique_value(a_idxs, ts_when_predicted)
            a_times, a_idxs, actors_activation_mask = (a_times[actor_to_activate_mask],
                                                       a_idxs[actor_to_activate_mask],
                                                       actors_activation_mask[actor_to_activate_mask])

        # update the actor states based on the actors that are activated in the current time step and remove the
        # corresponding activations from the list of activations
        u_act = np.zeros(len(self.actors.pos), dtype=float)
        u_act[a_idxs.astype(int)] = a_times - self.actors.t_lead
        self.actors.predict_own_actor_state(u_act)
        if actors_activation_mask.ndim > 0:
            self._activations = self._activations[actors_activation_mask]

        return predicted_time, actuator_idxs


def build_controller(actors_config_path,
                     controller_type='first_actor_first',
                     controller_config_path=None,
                     special_objects_dict=None,
                     x_velocity_guess=None,
                     verbosity_level='INFO'):
    """Builds a controller based on the provided configuration paths and parameters.

    The returned controller can be used to control the sorting system by calling its call/run_one_step(...) method.

    :param actors_config_path: A string, path to the JSON file where the configuration for the actors is stored.
    :param controller_type: A string, either first_actor_first" or "ptcr", type of the controller to build.
    :param controller_config_path: A string, path to the JSON file where the controller config is stored.
    :param special_objects_dict: A dict, where the keys are strings corresponding to the names in the controller config
        that should be added or replaced to the controller config, and the values are the corresponding objects.
    :param x_velocity_guess: None or float, the velocity guess for the x direction. Used to compute a default value for
        the number of time steps N in the PTCR controller if not provided in the controller config. Ignored in all other
        cases.
    :param verbosity_level: A string, the verbosity level for logging. Should be one of ['FATAL', 'ERROR', 'WARNING',
        'INFO', 'DEBUG'].

    :returns: A ControllerWrapper object, a controller that can be used to control the sorting system.
    """
    if x_velocity_guess is not None and not isinstance(x_velocity_guess, (int, float)):
        raise ValueError("x_velocity_guess should be a float or int if provided.")

    # Setup logging
    if verbosity_level not in ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']:
        raise ValueError("Value for verbosity_level should be one of ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']")
    logging.set_verbosity(verbosity_level)

    # Print the TensorFlow version
    logging.info(f'TensorFlow version: {TF_VERSION}')

    # import controller config to json tree
    if controller_config_path is not None:
        with open(controller_config_path) as f:
            controller_config = json.load(f)
    else:
        controller_config = {}
    if special_objects_dict is not None:
        # update controller config with special objects dict if provided
        controller_config.update(special_objects_dict)
    T = controller_config.pop('T')

    # import actors config to json tree
    with open(actors_config_path) as f:
        actors_config = json.load(f)
    actor_dict = actors_config['actor_times']

    # Define the actors
    actors = VirtualActor(pos=np.asarray(actors_config['actor_positions']),
                          length=actors_config['length'],
                          width=actors_config.get('width'),
                          T=T,
                          **actor_dict,
                          )
    for k in ['init_t_act', 'init_k', 'suppress_warnings']:
        actor_dict.pop(k, None)  # remove the keys from the actor dict if they are present, control does not expect them

    # Define the dummy MTT tracker (for controllers that do not require a particle simulator)
    mtt_tacker = DummyMTTTracker()

    # Define the default particle and contact models for the controllers
    default_particle_model_dict = {'p_detect': 0.95,
                                   'S_w': (0.0001, 0.0001),
                                   'S_v': (0.01, 0.01),
                                   }
    default_contact_model_dict = {'p_eject': [0.0, 0.9, 0.2, 0.1],
                                  'use_sum_approximation': False,
                                  }

    # Build the controller
    if controller_type == "first_actor_first":

        # Build the controller
        controller = FirstActorFirstHeuristicController(
            mtt_tacker,
            actors=actors,
            T=T,
            actor_model_dict=actor_dict,
            **controller_config,
        )

    elif controller_type == "ptcr":

        # Build the controller
        internal_T = controller_config.pop('internal_T', None)
        if x_velocity_guess is not None:
            default_N = avg_num_time_steps_particle_in_range((0.0, actors.end_array), x_velocity_guess,
                                                             internal_T if internal_T is not None else actors.t_cycle) + 1
        else:
            default_N = None  # will raise an error if N is not provided in the controller config
        accept_cost_weight = controller_config.pop("accept_cost_weight", 0.5)
        controller = PTCRController(
            mtt_tracker=mtt_tacker,
            actors=actors,
            T=T,
            N=controller_config.pop('N', default_N),
            particle_model_dict=controller_config.pop("particle_model", default_particle_model_dict),
            actor_model_dict=actor_dict,
            contact_model_dict=controller_config.pop("contact_model", default_contact_model_dict),
            particle_ordering=controller_config.pop("particle_sorter", "ParticleSorterById"),
            N_R=controller_config.pop("N_R", 2),
            internal_T=internal_T,
            use_only_terminal_costs=controller_config.pop("use_only_terminal_costs", False),
            accept_cost_weight=accept_cost_weight,
            reject_cost_weight=1 - accept_cost_weight,
            use_olf=controller_config.pop("use_olf", True),
            **pop_ptcr_nested_config(controller_config),
            **controller_config,
        )

    else:
        raise ValueError(f"Controller {controller_type} is not supported.")

    # Build the controller wrapper
    controller = ControllerWrapper(controller, actors)

    return controller
