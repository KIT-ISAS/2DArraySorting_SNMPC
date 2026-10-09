from abc import ABC, abstractmethod

import numpy as np

from simulator import AbstractParticleSimulator, ActorSimulator, AreaParticleSimulator
from mtt_trackers import DummyMTTTracker
from snmpc_particle_model import ParticleModel
from actor_model import AbstractActorModel
from contact_model import AbstractContactModel


class AbstractController(ABC):
    """Abstract class for all controllers for control of a sorting machine with a 2D optical sorter.

    The goal of the proposed sorter is to activate the actuators so that an (sub-)optimal sorting result (in terms of a
    many as possible ejected reject particles and as few as possible ejected accept particles) is achieved.
    """

    def __init__(self,
                 mtt_tracker,
                 actors,
                 T,
                 N,
                 max_actions=10,
                 ):
        """Initializes the controller.

        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. If a callable, multitarget tracker to be used for tracking the particles. For the required
            signature of the callable see control_from_measurements(...). For simulation and evaluation purpose, you can
            alternatively pass an AbstractParticleSimulator object that simulates the particles' motion. If you pass a
            DummyMTTTracker, no internal tracker is applied and only calls to the controller via
            control_from_states(...) are permitted.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param N: An integer or np.inf, the control horizon.
        :param max_actions: An integer, the maximum number of actions the controller can transmit to each actor during
            one control cycle (one time step). Correlates to the ratio of T and the time for one actor cycle. For
            example, if actors are fast enough to be activated in 10 times during one controller time step, max_actions
            should be at least 10.
        """
        if not callable(mtt_tracker) and not isinstance(mtt_tracker, (AbstractParticleSimulator, DummyMTTTracker)):
            raise ValueError('mtt_tracker currently only supports callables or '
                             'AbstractParticleSimulator and DummyMMTTracker instances.')
        if not isinstance(actors, ActorSimulator):
            raise ValueError('actors currently only supports ActorSimulator instances.')
        if isinstance(mtt_tracker, AbstractParticleSimulator):
            if not isinstance(mtt_tracker, AreaParticleSimulator):
                raise ValueError('If the actors are two-dimensional (2D sorter/actor grid), mtt_tracker must also '
                                 'track a two-dimensional process.')
        if not np.isscalar(T) or T <= 0:
            raise ValueError('The time interval between two consecutive time steps T must be a positive scalar.')
        if not np.isinf(N) and not isinstance(N, int) or N < 1:
            raise ValueError('The control horizon N must be an integer greater than or equal to 1 or np.inf.')
        if not isinstance(max_actions, int) or max_actions < 1:
            raise ValueError('max_actions must be an integer greater than or equal to 1.')

        self._mtt_tracker = mtt_tracker

        self._actors = actors

        self._T = float(T)
        self._N = N
        self._max_actions = max_actions

    @property
    @abstractmethod
    def particle_model(self):
        """The particle model used by the controller.

        :returns: None or a ParticleModel object, the particle model in use.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def actor_model(self):
        """The actor model used by the controller.

        :returns: None or an AbstractActorModel object, the actor model in use.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    @abstractmethod
    def contact_model(self):
        """The contact model used by the controller.

        :returns: An AbstractContactModel object, the contact model in use.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @property
    def mtt_tracker(self):
        """The multitarget tracker used by the controller.

        :returns: A callable, a child instance of AbstractParticleSimulator, or an instance of DummyMTTTracker. The
            multitarget tracker used by the controller.
        """
        return self._mtt_tracker

    @property
    def T(self):
        """The sampling time difference.

        :returns: A float representing the time interval between two consecutive time steps.
        """
        return self._T

    @property
    def N(self):
        """The control horizon.

        :returns: An integer, the control horizon.
        """
        return self._N

    def __call__(self, measurements):
        """Returns the actors' input as output of the controller's decision. Therefore, it first tracks the particles
        and then decides for an actor scheduling strategy including activation times for the actors.

        Format measurements:

            [x, y, label]
                in case of a 2D optical sorter (grid sorter),

            with label either 0 ("keep the particle") or 1 ("eject the particle").

        :param measurements: A np.array of shape [num_measurements, measurement_vector_length], the measurements
            (including x-position and class label) for the current time step.

        :returns activation_delta_t: A np.array of shape [num_actors, self._max_actions], the time to activate the
            actors as delta w.r.t. the current time step.
        """
        return self.control_from_measurements(measurements, mtt_tracker=self._mtt_tracker, actors=self._actors)

    @abstractmethod
    def control_from_states(self, particle_states, particle_class, particle_id, t_act):
        """Returns the actors' input as output of the controller's decision.

        Format particle_states:

            [estimated_motion_state_mean, estimated_motion_state_cov, track_score],

            with estimated_motion_state_mean being a np.array of shape [num_particles, 4] and format [position_x,
            velocity_x, position_y, velocity_y], in case of a 2D optical sorter (grid sorter) containing the expected
            values of the motion state and estimated_motion_state_cov a np.array of shape [num_particles, 4, 4]
            containing the covariances of the motion state. The track_score is a np.array of shape [num_particles] used
            as the existence probabilities of the particles.

        :param particle_states: A tuple of length 3.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        :param t_act: A np.array of shape [num_actors] representing the actors' internal time.

        :returns activation_delta_t: A np.array of shape [num_actors, self._max_actions], the time to activate the
            actors as delta w.r.t. the current time step.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def control_from_measurements(self, measurements, mtt_tracker, actors):
        """Returns the actors' input as output of the controller's decision. Therefore, it first tracks the particles
        and then decides for an actor scheduling strategy including activation times for the actors.

        Format measurements:

            [x, y, label]
                in case of a 2D optical sorter (grid sorter),

            with label either 0 ("keep the particle") or 1 ("eject the particle").

        :param measurements: A np.array of shape [num_measurements, measurement_vector_length], the measurements
            (including x-position and class label) for the current time step.
        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. Multitarget tracker to be used for tracking the particles. Signature of the callable should
            be measurements -> (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id,
            track_scores), with
                estimated_motion_state_mean: A np.array of shape [num_particles, state_length], the estimated motion
                    state mean,
                estimated_motion_state_cov: A np.array of shape [num_particles, state_length, state_length], the
                    estimated motion state covariance,
                particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
                    particle") or 1 ("eject the particle") representing the particle classes,
                particle_id: An integer np.array of shape [num_particles], the ids of the particles,
                track_score: A np.array of shape [num_particles] in [0, 1], the track scores. Used as estimated
                    existence probability for each particle.
            For simulation and evaluation purpose, you can pass alternatively pass an AbstractParticleSimulator object
            that simulates the particles' motion. If you pass a DummyMTTTracker, you must call the controller
                via control_from_states(...) instead of this method.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.

        :returns activation_delta_t: A np.array of shape [num_actors, self._max_actions], the time to activate the
            actors as delta w.r.t. the current time step.
        """
        if mtt_tracker != self._mtt_tracker:
            if not callable(mtt_tracker) and not isinstance(mtt_tracker, AbstractParticleSimulator):
                raise ValueError('mtt_tracker only supports callables and AbstractParticleSimulator instances.')
        if actors != self._actors:
            if not isinstance(actors, ActorSimulator):
                raise ValueError('actors currently only supports ActorSimulator instances.')
            if not isinstance(mtt_tracker, AreaParticleSimulator):
                raise ValueError('If the actors are two-dimensional (2D sorter/actor grid), mtt_tracker must also '
                                 'track a two-dimensional process.')

        # first, get the actor time from the real actors (we assume we can perfectly measure them)
        init_t_act = self._synchronize_actor_model(actors)

        # second, perform a tracking step
        # get the initial values from the multitarget tracker
        if isinstance(mtt_tracker, AbstractParticleSimulator):
            (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id,
             existence) = mtt_tracker.track_particles(measurements)
            # then, make a decision and compute the time of arrivals
            return self.control_from_states(
                particle_states=(estimated_motion_state_mean[existence], estimated_motion_state_cov[existence],
                                 np.ones(np.count_nonzero(existence))),
                particle_class=particle_class[existence],
                particle_id=particle_id[existence],
                t_act=init_t_act,
            )
        else:
            (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id,
             track_score) = mtt_tracker(
                measurements)
            # then, make a decision and compute the time of arrivals
            return self.control_from_states(
                particle_states=(estimated_motion_state_mean, estimated_motion_state_cov, track_score),
                particle_class=particle_class,
                particle_id=particle_id,
                t_act=init_t_act,
            )

    @staticmethod
    def _synchronize_actor_model(actors, t_act=None):
        """Synchronizes the actor model, i.e., queries the time of the actual actors to be used for overwriting the
        internal actor time.

        :param actors: The actors or an ActorSimulator object, the actors to control.
        :param t_act: None or a np.array of shape [num_actors] representing the actors' internal time.

        :returns: A np.array of shape [num_actors] representing the actors' current internal time.
        """
        if t_act is not None and not np.all(t_act == actors.t_act):
            raise RuntimeError(
                'The actor models internal time t_act and the actors time are not synchronized, check your settings.')
        return actors.t_act


class AbstractStochasticModelPredictiveController(AbstractController, ABC):
    """Abstract class for all stochastic model predictive controllers for control of a sorting machine with a 2D
    optical sorter.

    The goal of the proposed sorter is to activate the actuators so that an (sub-)optimal sorting result (in terms of a
    many as possible ejected reject particles and as few as possible ejected accept particles) is achieved.
    """
    def __init__(self,
                 mtt_tracker,
                 actors,
                 T,
                 N,
                 particle_model,
                 actor_model,
                 contact_model,
                 seed=None,
                 ):
        """Initializes the controller.

        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. Multitarget tracker to be used for tracking the particles. For simulation and evaluation
            purpose, you can alternatively pass an AbstractParticleSimulator object that simulates the particles'
            motion. If you pass a DummyMTTTracker, only calls via control_from_states(...) are permitted.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param N: An integer or np.inf, the control horizon.
        :param particle_model: A ParticleModel instance, the particle model in use.
        :param actor_model: A child instance of AbstractActorModel, the actor model in use.
        :param contact_model: None or an AbstractContactModel instance. ADF-based
            controllers may install their contact model after base initialization.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        """
        # sanity checks
        if not isinstance(particle_model, ParticleModel):
            raise ValueError('particle_model must be an instance of ParticleModel.')
        if not isinstance(actor_model, AbstractActorModel):
            raise ValueError('actor_model must be a child instance of AbstractActorModel.')
        if contact_model is not None and not isinstance(contact_model, AbstractContactModel):
            raise ValueError('contact_model must be None or a child instance of AbstractContactModel.')
        if not isinstance(particle_model.S_w, (list, tuple, np.ndarray)) or len(particle_model.S_w) != 2:
            raise ValueError('If the actors are two-dimensional (2D sorter/actor grid), particle_model must'
                             'also implement a a two-dimensional process.')
        if actor_model.width is None:
            raise ValueError('If the actors are two-dimensional (2D sorter/actor grid), actor_model must also '
                             'implement a two-dimensional sorter')

        super().__init__(mtt_tracker=mtt_tracker,
                         actors=actors,
                         T=T,
                         N=N,
                         max_actions=1,  # always one (by key assumption)
                         )

        self._particle_model = particle_model
        self._actor_model = actor_model
        self._contact_model = contact_model

        # set the seed and create the random number generator
        self._rng = np.random.default_rng(seed)

    @property
    def particle_model(self):
        return self._particle_model

    @property
    def actor_model(self):
        return self._actor_model

    @property
    def contact_model(self):
        return self._contact_model

    @property
    def rng(self):
        """The random number generator used by the controller.

        :returns: A np.random.Generator object, the random number generator used by the controller.
        """
        return self._rng

    @abstractmethod
    def _activation_delta_t_to_control_sequence(self, activation_delta_t, N, **kwargs):
        """Transforms the activation from format [num_actors, max_actions] with values being activation times to format
         [num_actors, N] where the values are ordered by time steps.


        :param activation_delta_t: A np.array of shape [num_actors, max_actions], the time to activate the actors as
            delta w.r.t. the current time step.
        :param N: An integer, the number of time steps to consider.

        :returns A (Boolean or float) np.array of shape [num_actors, N], the time (steps) to activate the actors in an
            appropriate format for the controller. See the implementation of the subclass for details.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')


class AbstractDynamicProgrammingStochasticModelPredictiveController(AbstractStochasticModelPredictiveController, ABC):
    """Abstract class for all stochastic model predictive controllers for control of a sorting machine with a 2D
    optical sorter based on dynamic programming.

    The goal of the proposed sorter is to activate the actuators so that an (sub-)optimal sorting result (in terms of a
    many as possible ejected reject particles and as few as possible ejected accept particles) is achieved.
    """

    def __init__(self,
                 mtt_tracker,
                 actors,
                 T,
                 N,
                 particle_model,
                 actor_model,
                 contact_model,
                 N_limited_look_ahead=None,
                 accept_cost_weight=0.5,
                 reject_cost_weight=0.5,
                 use_only_terminal_costs=True,
                 sampling_method_for_rollout=('mle', {}),
                 seed=None,
                 ):
        """Initializes the controller.

        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. Multitarget tracker to be used for tracking the particles. For simulation and evaluation
            purpose, you can alternatively pass an AbstractParticleSimulator object that simulates the particles'
            motion. If you pass a DummyMTTTracker, only calls via control_from_states(...) are permitted.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param N: An integer or np.inf, the control horizon.
        :param particle_model: A ParticleModel instance, the particle model in use.
        :param actor_model: A child instance of AbstractActorModel, the actor model in use.
        :param contact_model: A child instance of AbstractContactModel, the particle model in use.
        :param N_limited_look_ahead: None or an integer, the limited look-ahead horizon, i.e., the control horizon for
            which the costs are calculated exactly and the optimization of the inputs is performed. For the remaining
            N - N_limited_look_ahead time steps, a rollout heuristic is used to approximate the cost to go. If None,
            the full control horizon is used, i.e., rollout is disabled.
        :param accept_cost_weight: The weighting factor for the costs associated with ejected accept particles.
        :param reject_cost_weight: The weighting factor for the costs associated with not ejected reject particles.
        :param use_only_terminal_costs: A Boolean, whether to apply the costs for the particles only at the last stage
            and use no step costs (True) or apply step costs in every time step during the control horizon but do not
            apply terminal costs (False).
        :param sampling_method_for_rollout: A tuple of length 2, the first element is a string representing the sampling
            method for the rollout policy ('mle' or 'random') and the second element is a dict of kwargs for the
            respective method. Ignored if N_limited_look_ahead is None.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        """
        # sanity checks
        if accept_cost_weight + reject_cost_weight != 1:
            raise ValueError('The weighting factors accept_cost_weight should sum up to 1.')
        if N_limited_look_ahead is not None:
            if not np.isinf(N_limited_look_ahead) and not isinstance(N_limited_look_ahead,
                                                                     int) or N_limited_look_ahead < 1:
                raise ValueError(
                    'The limited look-ahead horizon N_limited_look_ahead must be an integer greater than or equal to 1.')
            if N_limited_look_ahead > N:
                raise ValueError('N_limited_look_ahead must be smaller or equal to the control horizon N.')

        super().__init__(mtt_tracker=mtt_tracker,
                         actors=actors,
                         T=T,
                         N=N,
                         particle_model=particle_model,
                         actor_model=actor_model,
                         contact_model=contact_model,
                         seed=seed,
                         )

        self._accept_cost_weight = accept_cost_weight
        self._reject_cost_weight = reject_cost_weight

        self._use_only_terminal_costs = use_only_terminal_costs

        self._N_limited_look_ahead = N_limited_look_ahead
        if N_limited_look_ahead == N:
            self._N_limited_look_ahead = None


        # build the rollout sampling method
        # TODO: Add support for general rollout controllers, no matter if they are stochastic or deterministic
        # TODO: control_from_states could also take existence probabilities as input to avoid sampling for stochastic
        #  policies
        if sampling_method_for_rollout[0] == 'mle':
            self._create_set_of_existence_samples_fn = self._create_set_of_existence_samples_mle
        elif sampling_method_for_rollout[0] == 'random':
            self._create_set_of_existence_samples_fn = lambda \
                existence_probs: self._create_set_of_existence_samples_random(existence_probs=existence_probs,
                                                                              **sampling_method_for_rollout[1])
        else:
            raise ValueError('The sampling method for the rollout policy must be either "mle" or "random".')

    def _create_set_of_existence_samples_mle(existence_probs):
        """Creates a set of existence samples from the existence probabilities using maximum likelihood, i.e., only one
        sample is drawn per particle.

        This function can be used to transform the existence probabilities into a set of existence samples that can be
        processed by a deterministic controller, e.g., a rollout heuristic.

        :param existence_probs: A np.array of shape [num_particles, 2], the distribution of ex_{k}, i.e., the
            probabilities [ P(exist not), P (exist) ], or, to be more precise, the probabilities
            [ P(ex_{k} = 0 | I_{k}) , P(ex_{k} = 1 | I_{k})].

        :returns particle_exists_masks: A Boolean np.array of shape [1, num_particles], a mask for the existence of
            each particle representing samples from existence_probs. The mask is True if the particle exists and
            False if the particle does not.
        """
        particle_exists_masks = np.expand_dims(existence_probs[:, 1] >= 0.5, axis=0)
        return particle_exists_masks

    def _create_set_of_existence_samples_random(self, existence_probs, num_samples):
        """Creates a set of existence samples from the existence probabilities using random sampling.

        This function can be used to transform the existence probabilities into a set of existence samples that can be
        processed by a deterministic controller, e.g., a rollout heuristic.

        :param existence_probs: A np.array of shape [num_particles, 2], the distribution of ex_{k}, i.e., the
            probabilities [ P(exist not), P (exist) ], or, to be more precise, the probabilities
            [ P(ex_{k} = 0 | I_{k}) , P(ex_{k} = 1 | I_{k})].

        :returns particle_exists_masks: A Boolean np.array of shape [num_samples, num_particles], a mask for the
            existence of each particle representing samples from existence_probs. The mask is True if the particle
            exists and False if the particle does not.
        """
        return self._rng.binomial(n=1, p=existence_probs[:, 1],
                                  size=(num_samples, existence_probs.shape[0])).astype(bool)
