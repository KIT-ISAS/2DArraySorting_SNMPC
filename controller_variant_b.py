from absl import logging

import numpy as np

from abc import ABC

from controller import AbstractStochasticModelPredictiveController, \
    AbstractDynamicProgrammingStochasticModelPredictiveController
from actor_model import ActorModelVariantB
from snmpc_particle_model import ParticleModel


class AbstractControllerVariantB(AbstractStochasticModelPredictiveController, ABC):
    """Abstract class for stochastic optimal controllers for control of a sorting machine with multiple actors
    per groove (1D optical sorter) or with a multi-array grid on a belt or chute (2D optical sorter) and with
    continuous actor activation times (variant B of the proposed controllers).

    These controllers (variant B) in each time steps computes a control

       u_k = [ u_{1,k}   ...   u_{N_A,k} ]^T

    with u_{j,k} in [0, inf_value] and N_A being the number of actors in the setup and u_{j,k}^act being the time to
    activate the actors as delta w.r.t. the current time step.
    """

    def __init__(
            self,
            mtt_tracker,
            actors,
            T,
            N,
            particle_model_dict,
            actor_model_dict,
            contact_model_dict,
            inf_value_factor,
            seed=None,
            **kwargs,
    ):
        """Initializes the controller.

        :param mtt_tracker: A multitarget tracker or child instance of AbstractParticleSimulator. Tracker to be used for
            tracking the particles. For simulation and evaluation purpose, you can pass a AbstractParticleSimulator
            object that simulates the particles' motion.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param N: An integer or np.inf, the control horizon.
        :param particle_model_dict: A dict of kwargs with settings for ParticleModel.
        :param actor_model_dict: A dict of kwargs with settings for ActorModelVariantB.
        :param contact_model_dict: A dict of kwargs. Accepted for interface compatibility; the
            concrete ADF contact model is built by the ADF mixin, so this dict is currently unused.
        :param inf_value_factor: A float, inf_value_factor * self.T represents infinity in the optimization problem.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        """
        if not isinstance(inf_value_factor, float) or inf_value_factor <= 0:
            raise ValueError("inf_value_factor must be a float greater than zero.")

        # build the particle, contact, and actor model
        particle_model = ParticleModel(
            T=T,
            seed=(
                seed + 123456 if seed is not None else None
            ),  # do not use the same seed as
            # for the controller
            **particle_model_dict,
        )
        actor_model = ActorModelVariantB(
            pos=actors.pos,
            length=actors.length,
            width=actors.width,
            T=T,
            **actor_model_dict,
        )
        if actors.width is not None and (
                not isinstance(particle_model.S_w, (list, tuple, np.ndarray))
                or len(particle_model.S_w) != 2
        ):
            raise ValueError(
                "If the actors are two-dimensional (2D sorter/actor grid) particle_model_dict must also "
                "define a two-dimensional process."
            )
        if actors.width is None and isinstance(
                particle_model.S_w, (list, tuple, np.ndarray)
        ):
            raise ValueError(
                "If the actors are one-dimensional (1D groove sorter) particle_model_dict must also "
                "define a one-dimensional process."
            )
        super().__init__(
            mtt_tracker=mtt_tracker,
            actors=actors,
            T=T,
            N=N,
            particle_model=particle_model,
            actor_model=actor_model,
            contact_model=None,
            seed=seed,
            **kwargs,
        )

        self._inf_value = (
                inf_value_factor * self._T
        )  # since with np.inf, we do not have gradients nor a valid point
        # for regression, defines the upper bounds for optimization

    @property
    def inf_value(self):
        """The representation for infinity in the optimization problem.

        :returns: A float, the representation for infinity in the optimization problem.
        """
        return self._inf_value

    def _synchronize_actor_model(self, actors, t_act=None):
        """Synchronizes the actor model, i.e., queries the time of the actual actors to be used for overwriting the
        internal actor time.

        Since we need to synchronize the discrete-time model from the simulator with the continuous-time model from the
        controller, we need to make one important change: Whereas time t_cycle for the discrete-time model is unequal to
        zero, it is equal to zero for the continuous-time model. This is because, for the discrete-time model to have an
        actor cycle of length t_cycle, the last time step has to be considered explicitly. The continuous time model on
        other hand, has finishes a cycle of length of t_cycle at t_cycle.

        :param actors: The actors or an ActorSimulator object, the actors to control.
        :param t_act: None or a np.array of shape [num_actors] representing the actors' internal time.

        :returns: A np.array of shape [num_actors] representing the actors' current internal time.
        """
        t_act = super()._synchronize_actor_model(actors, t_act).copy()  # copy to avoid modifying the original array
        # in the next line
        return self._t_cycle_to_zero(t_act)

    def _t_cycle_to_zero(self, t_act):
        """Sets t_act to zero if t_act=t_cycle.

        Since we need to synchronize the discrete-time model from the simulator with the continuous-time model from the
        controller, we need to make one important change: Whereas time t_cycle for the discrete-time model is unequal to
        zero, it is equal to zero for the continuous-time model. This is because, for the discrete-time model to have an
        actor cycle of length t_cycle, the last time step has to be considered explicitly. The continuous time model on
        other hand, has finishes a cycle of length of t_cycle at t_cycle.

        :param t_act: None or a np.array of shape [num_actors] representing the actors' internal time.

        :returns: A np.array of shape [num_actors] representing the actors' current internal time.
        """
        t_act[t_act == self._actor_model.t_cycle] = 0.0
        return t_act

    def _activation_delta_t_to_control_sequence(self, activation_delta_t, N, inf_value=None, **kwargs):
        """Transforms the activation from format [num_actors, max_actions] with values being activation times to format
         [num_actors, N] where the values are ordered by time steps.

        If there are multiple activations per time step for one actor, only the first one is used.

        :param activation_delta_t: A np.array of shape [num_actors, max_actions], the time to activate the actors as
            delta w.r.t. the current time step.
        :param N: An integer, the number of time steps to consider.
        :param inf_value: None or float (including np.inf), the value to use for infinity. Note that np.inf, depending
            on the algorithms, can make some problems with bounds and gradients.

        :returns A np.array of shape [num_actors, N], the time to activate the actors as delta w.r.t. the current
            time step.
        """
        u_seq = np.empty((len(activation_delta_t), N), dtype=float)
        u_seq[:] = self._inf_value if inf_value is None else inf_value

        for i in range(N):
            # TODO: Use np.apply_along_axis?
            mask = np.logical_and(
                self._T * i <= activation_delta_t,
                activation_delta_t < self._T * (i + 1),
            )
            # activation_delta_t may have more than one activation per time step, so use only the first one
            if np.any(mask):
                u_seq[np.any(mask, axis=1), i] = activation_delta_t[
                    np.nonzero(np.any(mask, axis=1))[0],
                    np.argmax(mask, axis=1)[np.nonzero(np.any(mask, axis=1))[0]],
                ]
        return u_seq

    def _postprocess_activations(self, u, t_act):
        """Postprocess the activations, i.e., suppress activations that cannot be realized.

        :param u: A np.array of shape [num_actors, num_time_steps_in_sequence], the sequence of control inputs, i.e.,
            the time to activate the actors as delta w.r.t. the current time step. The sequence starts at the initial
            time step k and ends with the input for time step k + num_time_steps_in_sequence - 1.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns: A np.array of shape [num_actors, num_time_steps_in_sequence], the sequence of control inputs, i.e.,
            the time to activate the actors as delta w.r.t. the current time step. The sequence starts at the initial
            time step k and ends with the input for time step k + num_time_steps_in_sequence - 1.
        """
        u[
            np.logical_and(
                t_act[:, None] > 0, self._actor_model.t_cycle - t_act[:, None] > u
            )
        ] = self._inf_value
        return u


class AbstractDynamicProgrammingControllerVariantB(
    AbstractControllerVariantB,
    AbstractDynamicProgrammingStochasticModelPredictiveController,
    ABC):
    """Abstract class for stochastic optimal controllers for control of a sorting machine with multiple actors
    per groove (1D optical sorter) or with a multi-array grid on a belt or chute (2D optical sorter) and with continuous
    actor activation times (variant B of the proposed controllers) that use dynamic programming, that is, rollouts on a
    given action sequence, for calculating the cost-to-gos.

    These controllers (variant B) in each time steps computes a control

       u_k = [ u_{1,k}   ...   u_{N_A,k} ]^T

    with u_{j,k} in [0, inf_value] and N_A being the number of actors in the setup and u_{j,k}^act being the time to
    activate the actors as delta w.r.t. the current time step.
    """

    def __init__(
            self,
            mtt_tracker,
            actors,
            T,
            N,
            particle_model_dict,
            actor_model_dict,
            contact_model_dict,
            N_limited_look_ahead=None,
            accept_cost_weight=0.5,
            reject_cost_weight=0.5,
            use_only_terminal_costs=True,
            use_mle_for_existence_meas=True,
            use_olf=False,
            sampling_method_for_rollout=("mle", {}),
            delta_k_est=1,
            seed=None,
    ):
        """Initializes the CLF controller.

        :param mtt_tracker: A multitarget tracker or an AreaParticleSimulator object. Tracker to be used for tracking
            the particles. For simulation and evaluation purpose, you can pass an AreaParticleSimulator object that
            simulates the particles' motion.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param N: An integer, the control horizon.
        :param N_limited_look_ahead: None or an integer, the limited look-ahead horizon, i.e., the control horizon for
            which the costs are calculated exactly and the optimization of the inputs is performed. For the remaining
            N - N_limited_look_ahead time steps, a rollout heuristic is used to approximate the cost to go. If None,
            the full control horizon is used, i.e., rollout is disabled.
        :param particle_model_dict: A dict of kwargs with settings for ParticleModel.
        :param actor_model_dict: A dict of kwargs with settings for ActorModelVariantB.
        :param contact_model_dict: A dict of kwargs. Accepted for interface compatibility; the
            concrete ADF contact model is built by the ADF mixin, so this dict is currently unused.
        :param accept_cost_weight: The weighting factor for the costs associated with ejected accept particles.
        :param reject_cost_weight: The weighting factor for the costs associated with not ejected reject particles.
        :param use_only_terminal_costs: A Boolean, whether to apply the costs for the particles only at the last stage
            and use no step costs (True) or apply step costs in every time step during the control horizon but do not
            apply terminal costs (False).
        :param use_mle_for_existence_meas: A Boolean, whether to use the maximum likelihood estimates for the predicted
            existence measurements (True) or a measurement-tuple where the first measurement is that no particle exists
            and the second measurement is that all particles exist.
        :param use_olf: A Boolean, whether to use an open-loop feedback control (OLF) approach (True) or not (False).
        :param sampling_method_for_rollout: A tuple of length 2, the first element is a string representing the sampling
            method for the rollout policy ('mle' or 'random') and the second element is a dict of kwargs for the
            respective method. Ignored if N_limited_look_ahead is None.
        :param delta_k_est: An integer, the amount of controller time steps after which a new measurement is received.
            In this case, only affects the motion state. For open-loop feedback control (OLF), set this greater than N.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        """
        super().__init__(
            mtt_tracker=mtt_tracker,
            actors=actors,
            T=T,
            N=N,
            particle_model_dict=particle_model_dict,
            actor_model_dict=actor_model_dict,
            contact_model_dict=contact_model_dict,
            N_limited_look_ahead=N_limited_look_ahead,
            accept_cost_weight=accept_cost_weight,
            reject_cost_weight=reject_cost_weight,
            use_only_terminal_costs=use_only_terminal_costs,
            sampling_method_for_rollout=sampling_method_for_rollout,
            inf_value_factor=1.2,
            seed=seed,
        )

        # overwrite the definition of self._N_limited_look_ahead for disabling rollout
        self._N_limited_look_ahead = (
            self._N_limited_look_ahead
            if self._N_limited_look_ahead is not None
            else self._N
        )

        self._k = 0  # time step counter (only used for the plot function)
