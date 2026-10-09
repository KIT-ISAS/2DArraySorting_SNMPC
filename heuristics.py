from absl import logging

from abc import ABC, abstractmethod

import numpy as np

from controller import AbstractController
from actor_model import ActorModelVariantB2


class AbstractHeuristicController(AbstractController, ABC):
    """Abstract class for heuristics for control of a sorting machine with a multi-array grid on a belt or chute
    (2D optical sorter).

    The goal of the proposed sorter is to activate the actuators so that a (sub-)optimal sorting result (in terms of a
    many as possible ejected reject particles and as few as possible ejected accept particles) is achieved.

    These heuristics in each time steps compute a control

        u_k = [[ u_{1,1,k}     ...   u_{1,max_actions,k}   ]


               [ u_{N_A,1,k}   ...   u_{N_A,max_actions,k} ]]

    with components u_{j,l,k}, where u_{j,l,k} correspond to an expected arrival time of a particle at actor j minus an
    offset considering the time for activation of an actor, N_A being the number of actors in the setup, and max_actions
    the maximum number of actions for one actor per time step.
    """

    def __init__(self,
                 mtt_tracker,
                 actors,
                 T,
                 actor_model_dict,
                 temporal_point_predictor=None,
                 spatial_point_predictor=None,
                 max_actions=100):
        """Initializes the heuristic controller.

        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. Multitarget tracker to be used for tracking the particles. For the required signature of
            the callable see AbstractController.control_from_measurements(...). For simulation and evaluation purpose,
            you can alternatively pass an AbstractParticleSimulator object that simulates the particles' motion. If you
            pass a DummyMTTTracker, only calls via control_from_states(...) are permitted.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param actor_model_dict: A dict of kwargs with settings for ActorModelVariantB2.
        :param temporal_point_predictor: None or a function with signature (motion_state, actor_pos) -> delta_toa. See
            self._default_temporal_point_predictor for details on the required input and output format. If None, the
            default model (assuming constant-velocity behavior and states) is used.
        :param spatial_point_predictor: None or a function with signature (motion_state, actor_pos) -> y_at_toa. See
            self._default_spatial_point_predictor for details on the required input and output format. If None, the
            default model (assuming constant-velocity behavior and states) is used.
        :param max_actions: An integer, the maximum number of actions the controller can transmit to each actor during
            one control cycle (one time step). Correlates to the ratio of T and the time for one actor cycle. For
            example, if actors are fast enough to be activated in 10 times during one controller time step, max_actions
            should be at least 10.
        """
        super().__init__(mtt_tracker=mtt_tracker,
                         actors=actors,
                         T=T,
                         N=np.inf,  # it will proceed until all particles are processed
                         max_actions=max_actions,
                         )

        self._actor_model = ActorModelVariantB2(pos=actors.pos,
                                                length=actors.length,
                                                width=actors.width,
                                                T=T,
                                                **actor_model_dict)

        self._temporal_point_predictor = self._default_temporal_point_predictor if temporal_point_predictor is None \
            else temporal_point_predictor
        self._spatial_point_predictor = self._default_spatial_point_predictor if spatial_point_predictor is None \
            else spatial_point_predictor

    @property
    def particle_model(self):
        # Note: heuristics do not use an explicit particle model (rollout operates on point predictions).
        return None

    @property
    def actor_model(self):
        return self._actor_model

    @property
    def contact_model(self):
        # Note: heuristics do not use an explicit contact model.
        return None

    @abstractmethod
    def _assign_particle_to_actor(self,
                                  decision_mask,
                                  delta_toa,
                                  particle_class,
                                  particle_idx,
                                  t_actors_next_time_ready,
                                  ):
        """Decides which actor to activate for the eject particles under consideration.

        :param decision_mask: A Boolean np.array of shape [num_actors, num_particles], the mask for delta_toa before the
            strategy for the current particle (specified by particle_idx) is applied.
        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_idx: Integer, the index of the current particle to look at.
        :param t_actors_next_time_ready: A np.array of shape [num_actors], the time step when the actors will be in
            ready-status before the strategy is applied as delta w.r.t. to the current actor time step.

        :returns:
            decision_mask: A Boolean np.array of shape [num_actors, num_particles], the mask for delta_toa after the
                    strategy for the current particle is applied.
            t_actors_next_time_ready: A np.array of shape [num_actors], the time step when the actors will be in
                ready-status after the chosen strategy is applied as delta w.r.t. to the current actor time step.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @abstractmethod
    def predict_time_of_arrival(self, motion_state):
        """Predicts the time of the particles' arrivals at the actors. If a particle does not move across an actor,
        it's time of arrival at this actor is -np.inf.

         Format delta_toa:

            delta_toa_ij = number of time steps until arrival of particle j at actor i

        :param motion_state: A np.array of shape [num_particles, motion_state_length], the current value for the motion
            state.

        :returns delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
                    w.r.t. the current time step.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @staticmethod
    @abstractmethod
    def _default_temporal_point_predictor(motion_state, actor_pos):
        """Default model for predicting the time of the particles' arrivals at the actors.

        The default model assumes a constant velocity motion model for the particles and state vector that is formatted
        accordingly.

           Format delta_toa:

              delta_toa_ij = number of time steps until arrival of particle j at actor i

           Format motion_state:

              [position_x, velocity_x, position_y, velocity_y]
                  in case of a 2D optical sorter (grid sorter).

          :param motion_state: A np.array of shape [num_particles, 4], the current value for the motion state.
          :param actor_pos: A np.array of shape [num_actors, 2], the positions of actors.

          :returns delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
                      w.r.t. the current time step.
          """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    @staticmethod
    def _default_spatial_point_predictor(motion_state, delta_toa):
        """Default model for predicting the position orthogonal to the transport direction of the particles at the
        arrival times at the actors.

        The default model assumes a constant velocity motion model for the particles and state vector that is formatted
        accordingly.

           Format delta_toa:

              delta_toa_ij = number of time steps until arrival of particle j at actor i

           Format motion_state:

              [position_x, velocity_x, position_y, velocity_y]
                  in case of a 2D optical sorter (grid sorter).

          :param motion_state: A np.array of shape [num_particles, 4], the current value for the motion state.
          :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.

          :returns y_at_toa: A np.array of shape [num_actors, num_particles], the particles' position orthogonal to the
                        transport direction at the time of arrival at the actors.
          """
        # To be overwritten by subclass
        raise NotImplementedError(
            '_default_spatial_point_predictor must be implemented for class {}.'.format(self.__class__.__name__))

    def control_from_measurements(self, measurements, mtt_tracker, actors):
        """Returns the actors' input as output of the controller's decision. Therefore, it first tracks the particles
        and then decides for an actor scheduling strategy including activation times for the actors based on predicted
        time of particles' arrivals.

        Note that the controller's output, activation_delta_t, may contain activations not realizable by the actors.
        This happens if an eject particle cannot be ejected. This behavior is intended since it the real actors may
        differ from the assumed actor model and thus should be given the chance to eject the particle.

        Format measurements:

            [x, y, label]
                in case of a 2D optical sorter (grid sorter),

            with label either 0 ("keep the particle") or 1 ("eject the particle").

        :param measurements: A np.array of shape [num_measurements, measurement_vector_length], the measurements
            (including x-position and class label) for the current time step.
        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. Multitarget tracker to be used for tracking the particles. See
            AbstractController.control_from_measurements(...) for the required callable signature and details.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.

        :returns activation_delta_t: A np.array of shape [num_actors, self._max_actions], the time to activate the
            actors as delta w.r.t. the current time step.
        """
        return super().control_from_measurements(measurements, mtt_tracker, actors)

    def control_from_states(self, particle_states, particle_class, particle_id, t_act):
        """Returns the actors' input as output of the controller's decision. Therefore, it decides for an actor
        scheduling strategy including activation times for the actors based on predicted time of particles' arrivals.

        Note that the controller's output, activation_delta_t, may contain activations not realizable by the actors.
        This happens if an eject particle cannot be ejected. This behavior is intended since it the real actors may
        differ from the assumed actor model and thus should be given the chance to eject the particle.

        Format particle_states:

            [estimated_motion_state_mean, _, _]

            with motion_state being a np.array of shape [num_particles, 4] and format [position_x, velocity_x,
            position_y, velocity_y], in case of a 2D optical sorter (grid sorter).

        :param particle_states: A tuple of length >= 1.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns activation_delta_t: A np.array of shape [num_actors, self._max_actions], the time to activate the
            actors as delta w.r.t. the current time step.
        """
        # check that max_actions >= num_particles (implementation reasons only)
        if self._max_actions < particle_class.shape[0]:
            raise ValueError(
                'max_actions needs to be greater or equal to the current number of particles in the field of view.')

        # expand the states
        motion_state_mean = particle_states[0]
        init_t_act = t_act

        # initialize the array
        activation_delta_t = np.empty((len(self._actor_model.pos), self._max_actions))
        activation_delta_t[:] = np.inf

        # if there are no particles, return infinite activation times
        if motion_state_mean.size == 0:
            return activation_delta_t

        # else, predict to the actors and compute the time of arrivals
        # get the time of arrival
        delta_toa = self.predict_time_of_arrival(motion_state=motion_state_mean)  # array of shape
        # [num_actors, num_particles]

        # make a decision
        decision_mask = self.make_decision(delta_toa, particle_class, init_t_act)

        # go from hitting times to activation times
        activation_delta_t[:, :delta_toa.shape[1]] = np.where(decision_mask, delta_toa,
                                                              np.inf) - self._actor_model.t_lead

        return activation_delta_t

    def make_decision(self, delta_toa, particle_class, init_t_act, prescribed_control_sequence=None):
        """Decides which actor to activate for the eject particles to be shot out and when to activate the actors.
         Therefore, the function returns the time steps when the actors to be activated should ideally hit the particles.

        Decision-making is dependent on (from restrictive to less restrictive):

            * Class of particles
            * Predicted time of particles' arrivals (TOAs)
            * Actor status at TOAs
            * Correlations/risks between TOAs of different particles introduced by potentially blocking actors or
               already ejected particles by previous actors

        Assumptions:

          * For this controller, we now assume that a particle can be associated to exactly zero or one actor, so that
            the mapping from particles to actors is unique. However, one actor can of course be used to eject more than
            one particle. This also means that if the particle is not hit by the scheduled actor, it is certainly missed
            (as long as it crosses the actor field before a new measurement arrives) since there is no redundancy in the
            activations of the actors for a single particle.

          * We furthermore specify that in all cases particles of the class to keep do not cause activations.

          * We activate the actors so that they will hit the particles at the first time step of the actors' t_hit,
            i.e., we implicitly assume that there is no optimal particle hitting time as long as it is hit
            within the actors' hitting time and that this time interval as well as the particles time of arrival
            are perfectly known (with uncertainty zero).

        For the mapping from predicted time of arrival to actor decisions, we build a decision mask which describes
        these associations. The decision mask thus is a mask for the predicted time of arrival that only keeps the ones
        that should cause an activation of the corresponding actor. As explained before, for the associations, we
        assume that each particle is only hit by one actor. Each column of the decision mask must therefore contain at
        most one True entry. If a particle is not of the eject class or the algorithm decides not the shoot out a
        particle in order to increase the probability of successfully hitting or not hitting preceding or following
        particles, the corresponding columns only contain False entries. Thus, the algorithm terminates, if in each
        column exits at most one True entry.

         Example (with 2 actors and 4 particles):

                 toa                         decision_mask                activation_delta_t
             [2, 4, 6, 8]        &           [1, 0, 0, 1]      ==>       [2, inf, inf, 8]
             [3, 5, 7, 9]                    [0, 1, 0, 0]                [inf, 5, inf, inf]

        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param init_t_act: A np.array of shape [num_actors] representing the initial actors' internal time.
        :param prescribed_control_sequence: None or a np.array of shape [num_actors, N], already scheduled actor
            activation times as delta w.r.t. the initial time step. The activations times are already split by time
            steps.

        :returns decision_mask: A Boolean np.array of shape [num_actors, num_particles], the mask for delta_toa after
            the strategy has been applied.
        """
        # build a decision mask
        decision_mask = np.ones(delta_toa.shape, dtype=bool)  # array of shape [num_actors, num_particles]

        # start with the two most restrictive criteria, the class and the actor status at the time of arrival
        # class must be of ejection class
        decision_mask[:, np.logical_not(particle_class)] = False
        # actor must be able to hit at delta_toa
        actors_next_time_ready = np.where(init_t_act == 0, np.zeros(len(init_t_act)),
                                          self._actor_model.t_cycle - init_t_act)
        decision_mask[
            delta_toa < actors_next_time_ready[:, None] + self._actor_model.t_lead] = False

        # then we need to check which actors were triggered by the last control loop and might now hit or
        # disturb particles (because we need to know if a particle will be hit by an actor triggered in the loop
        # before, if so, we don't need to trigger another actor for that particle again)
        t_next_hit = self._actor_model.t_activate + self._actor_model.t_up - init_t_act
        for i in range(len(self._actor_model.pos)):
            if init_t_act[i] > 0 and init_t_act[
                i] <= (self._actor_model.t_activate + self._actor_model.t_up +
                       self._actor_model.t_hit + self._actor_model.t_down):
                # if not in actor status READY or RESET
                disturbed = np.logical_and(delta_toa[i, :] >= t_next_hit[i] - self._actor_model.t_up,
                                           delta_toa[i, :] <= t_next_hit[
                                               i] + self._actor_model.t_hit + self._actor_model.t_down)
                decision_mask[:, disturbed] = False  # if one particle is disturbed it is disturbed for all actors

        # then we check for prescribed activations, i.e., if a particle-actor assignment is blocked by already scheduled
        # activations (this is only the case if the heuristics builds upon decisions of other controller)
        if prescribed_control_sequence is not None:
            not_blocked = self._detect_if_particle_actor_assignment_is_not_blocked_by_prescribed_activations(
                delta_toa, prescribed_control_sequence)  # shape [num_actors, num_particles]
            decision_mask = np.logical_and(decision_mask, not_blocked)

        # then we need to check for implicit associations, i.e., particles for which only one actor is left and adjust
        # actors_next_time_ready
        implicit_decision_particles_mask = np.sum(decision_mask, axis=0) == 1
        actors_next_time_ready = self._assign_implicit_decisions(
            actors_next_time_ready,
            decision_mask,
            delta_toa,
            implicit_decision_particles_mask,
        )

        # check if we are already done, i.e., all particles are associated, iterate over particles otherwise
        # we iterate from the first particle (closest to the actors) to last particle in the field of view, thus
        # deciding for each particle sequentially and not for all at once.
        while np.any(np.sum(decision_mask, axis=0) > 1):
            # determine all particles that are not associated and are of the eject class
            particles_idxs_to_associate = \
                np.nonzero(np.logical_and(np.sum(decision_mask, axis=0) > 1, particle_class))[0]

            # look at the first eject particle that is not associated
            particle_idx = particles_idxs_to_associate[0]

            # investigate, decide for and apply a strategy for the actors that can be used to eject this particle
            decision_mask, actors_next_time_ready = self._assign_particle_to_actor(
                decision_mask,
                delta_toa,
                particle_class,
                particle_idx,
                t_actors_next_time_ready=actors_next_time_ready,
            )

            # check if the application of the strategy for the current particle has an effect on decisions for other
            # particles via blocked actors
            implicit_decisions_particles_mask = self._detect_implicit_associations(
                actors_next_time_ready,
                decision_mask,
                delta_toa,
                particle_class)
            actors_next_time_ready = self._assign_implicit_decisions(
                actors_next_time_ready,
                decision_mask,
                delta_toa,
                implicit_decisions_particles_mask)

        return decision_mask

    def _assign_implicit_decisions(self,
                                   actors_next_time_ready,
                                   decision_mask,
                                   delta_toa,
                                   implicit_decisions_particles_mask):
        """Assigns implicit decisions to the actors, i.e., manipulates actors_next_time_ready, based on the decision
        mask and the particles that are implicitly associated to the actors.

        :param actors_next_time_ready: A np.array of shape [num_actors] containing the time when the actor will be
            in READY status for the first time again as delta w.r.t. the current time step.
        :param decision_mask: A Boolean np.array of shape [num_actors, num_particles], the mask for delta_toa before the
            strategy for the current particle (specified by particle_idx) is applied.
        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param implicit_decisions_particles_mask: A Boolean np.array of shape [num_particles], a mask indicating which
            particles are implicitly associated.

        :returns:
            actors_next_time_ready: A np.array of shape [num_actors] containing the updated time when the actor will be
                in READY status for the first time again as delta w.r.t. the current time step.
        """
        implicit_decisions_actor_idxs, implicit_decisions_particle_idxs = np.nonzero(
            np.logical_and(decision_mask, implicit_decisions_particles_mask[None, :]))
        return self._actor_model.predict_actor_state(actors_next_time_ready,
                                                     delta_toa,
                                                     actor_indices=implicit_decisions_actor_idxs,
                                                     particle_indices=implicit_decisions_particle_idxs)

    def _detect_implicit_associations(self,
                                      actors_next_time_ready,
                                      decision_mask,
                                      delta_toa,
                                      particle_class):
        """Detects implicit associations, i.e., particles that are implicitly associated to the actors.

        :param actors_next_time_ready: A np.array of shape [num_actors] containing the time when the actor will be
            in READY status for the first time again as delta w.r.t. the current time step.
        :param decision_mask: A Boolean np.array of shape [num_actors, num_particles], the mask for delta_toa before the
            strategy for the current particle (specified by particle_idx) is applied.
        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.

        :returns:
            implicit_decisions_particles_mask: A Boolean np.array of shape [num_particles], a mask indicating which
            particles are implicitly associated.
        """
        particles_idxs_to_associate_before_implicit_association = np.logical_and(np.sum(decision_mask, axis=0) > 1,
                                                                                 particle_class)
        particles_blocked = delta_toa < actors_next_time_ready[:, None] + self._actor_model.t_lead
        decision_mask[np.logical_and((np.sum(decision_mask, axis=0) > 1)[None, :], particles_blocked)] = False  # only
        # manipulate the decision mask for particles that are not already associated
        particles_idxs_to_associate_after_implicit_association = np.logical_and(np.sum(decision_mask, axis=0) > 1,
                                                                                particle_class)
        return np.logical_and(
            particles_idxs_to_associate_before_implicit_association,
            np.logical_not(particles_idxs_to_associate_after_implicit_association))

    def _detect_if_particle_actor_assignment_is_not_blocked_by_prescribed_activations(self,
                                                                                      delta_toa,
                                                                                      prescribed_control_sequence):
        """Detects if an actor is blocked by activations of a already scheduled control sequences (these may also
        contain activations for future time steps).

        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param prescribed_control_sequence: A np.array of shape [num_actors, N], already scheduled actor activation
            times as delta w.r.t. the initial time step. The activations times are already split by time steps.

        :returns: A Boolean np.array of shape [num_actors, num_particles], a mask for whether the actor can be activated
            for the particle under consideration without violating the prescribed activation times.
        """
        activation_starts_after_prescribed_activation = delta_toa[:, :, None] - self.actor_model.t_lead > \
                                                        prescribed_control_sequence[:, None, :]
        # shape [num_actors, num_particles, N]

        # actor is still running (from a previous prescribed activation)
        actor_still_running = np.any(np.logical_and(
            activation_starts_after_prescribed_activation,
            delta_toa[:, :, None] - self.actor_model.t_lead <
            prescribed_control_sequence[:, None, :] + self._actor_model.t_cycle), axis=-1)

        # actor is not finished when required for a prescribed activation later on
        actor_needed_later = np.any(np.logical_and(
            np.logical_not(activation_starts_after_prescribed_activation),
            delta_toa[:, :, None] - self.actor_model.t_lead + self.actor_model.t_cycle >
            prescribed_control_sequence[:, None, :]), axis=-1)

        return np.logical_not(np.logical_or(actor_still_running, actor_needed_later))

    def _shoot_out_at_first_available_actor(self, decision_mask, delta_toa, particle_idx, t_actors_next_time_ready):
        """Applies a strategy where the considered eject particle is shot out at the first available (non-blocking)
        actor.

        :param decision_mask: A Boolean np.array of shape [num_actors, num_particles], the mask for delta_toa before the
            strategy for the current particle (specified by particle_idx) is applied.
        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param particle_idx: Integer, the index of the current particle to look at.
        :param t_actors_next_time_ready: A np.array of shape [num_actors], the time step when the actors will be in
            ready-status before the strategy is applied as delta w.r.t. to the current actor time step.

        :returns:
            decision_mask: A Boolean np.array of shape [num_actors, num_particles], the mask for delta_toa after the
                strategy for the current particle is applied.
            t_actors_next_time_ready: A np.array of shape [num_actors], the time step when the actors will be in
                ready-status after the chosen strategy is applied as delta w.r.t. to the current actor time step.
        """
        for a_idx in range(len(self._actor_model.pos)):
            if decision_mask[a_idx, particle_idx] and delta_toa[a_idx, particle_idx] >= t_actors_next_time_ready[
                a_idx] + self._actor_model.t_lead:
                decision_mask[:, particle_idx] = False
                decision_mask[a_idx, particle_idx] = True
                t_actors_next_time_ready = self._actor_model.predict_actor_state(t_actors_next_time_ready,
                                                                                 delta_toa,
                                                                                 actor_indices=a_idx,
                                                                                 particle_indices=particle_idx)
                break

        if np.sum(decision_mask[:, particle_idx]) > 1:
            logging.warning("One particle could not be ejected.")
            decision_mask[:, particle_idx] = False

        return decision_mask, t_actors_next_time_ready

    def __getstate__(self):
        """Returns the object's state that should be pickled.

        Note that python's multiprocessing module relies on the pickle module for serialization. The pickle module
        can only serialize objects that are accessible from the top level of a module. This excludes certain objects,
        e.g., lambda functions or closures, in which case a PicklingError is raised. To avoid this error, we exclude
        attributes that are not picklable from the state (we can only do this here since the excluded attributes are not
        required in the function that is executed in multiprocessing).

        From the pickle documentation (https://docs.python.org/3/library/pickle.html):

            Pickling class instances follows the (default) behavior:

                def save(obj):
                    return (obj.__class__, obj.__dict__)

                def restore(cls, attributes):
                    obj = cls.__new__(cls)
                    obj.__dict__.update(attributes)
                    return obj

            Classes can alter the default behaviour by providing one or several special methods:

                * __getstate__() : should return the state that should be pickled.
                * __setstate__() : should take the state that was returned by __getstate__() and set the instance's
                    state.
                * __getnewargs_ex__() : should return an (args, kwargs) tuple that are passed to __new__() during
                    unpickling.
                * __getnewargs__(): should return a tuple of args that are passed to __new__() during unpickling.
        """
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_mtt_tracker']
        return state


class AbstractHeuristicController2D(AbstractHeuristicController, ABC):
    """Abstract class for heuristics for control of a sorting machine with a multi-array grid on a belt or chute
    (2D optical sorter).

    The goal of the proposed sorter is to activate the actuators so that a (sub-)optimal sorting result (in terms of a
    many as possible ejected reject particles and as few as possible ejected accept particles) is achieved.

    These heuristics in each time steps compute a control

        u_k = [[ u_{1,1,k}     ...   u_{1,max_actions,k}   ]


               [ u_{N_A,1,k}   ...   u_{N_A,max_actions,k} ]]

    with components u_{j,l,k}, where u_{j,l,k} correspond to an expected arrival time of a particle at actor j minus an
    offset considering the time for activation of an actor, N_A being the number of actors in the setup, and max_actions
    the maximum number of actions for one actor per time step.
    """

    def predict_time_of_arrival(self, motion_state):
        """Predicts the time of the particles' arrivals at the actors. If a particle does not move across an actor,
        it's time of arrival at this actor is -np.inf.

         Format delta_toa:

            delta_toa_ij = number of time steps until arrival of particle j at actor i

         Format motion_state:

            [position_x, velocity_x, position_y, velocity_y]

        :param motion_state: A np.array of shape [num_particles, 4], the current value for the motion state.

        :returns delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
                    w.r.t. the current time step. If a particle does not move across an actor, it's time of arrival at
                    this actor is -np.inf.
        """
        delta_toa = self._temporal_point_predictor(motion_state, self._actor_model.pos)
        arrival_y_positions = self._spatial_point_predictor(motion_state, delta_toa)

        y_position_check_mask = np.logical_and(
            self._actor_model.pos[:, 1, None] - self._actor_model.width[:, None] / 2 < arrival_y_positions,
            self._actor_model.pos[:, 1, None] + self._actor_model.width[:, None] / 2 > arrival_y_positions)

        return np.where(y_position_check_mask, delta_toa, -np.inf)

    @staticmethod
    def _default_temporal_point_predictor(motion_state, actor_pos):
        # assume CV behavior
        delta_pos = actor_pos[:, 0, None] - motion_state[:, 0]
        return np.divide(delta_pos, motion_state[:, 1], dtype=np.float32)  # we use np.float32 to avoid numerical
        # issues in the > and < comparison here and in self._make_decision

    @staticmethod
    def _default_spatial_point_predictor(motion_state, delta_toa):
        # assume CV behavior
        return (motion_state[:, 2] + delta_toa * motion_state[:, 3]).astype(np.float32)


class FirstActorFirstHeuristicController(AbstractHeuristicController2D):
    """Heuristic controller for control of a sorting machine with a multi-array grid on a belt or chute (2D optical
    sorter). The heuristic simply chooses the first (nearest) actor that is able to eject and eject particle without
    consideration of any accept particles.

    The goal of the proposed sorter is to activate the actuators so that an (sub-)optimal sorting result (in terms of a
    many as possible ejected reject particles and as few as possible ejected accept particles) is achieved.

    These heuristics in each time steps compute a control

        u_k = [[ u_{1,1,k}     ...   u_{1,max_actions,k}   ]


               [ u_{N_A,1,k}   ...   u_{N_A,max_actions,k} ]]

    with components u_{j,l,k}, where u_{j,l,k} correspond to an expected arrival time of a particle at actor j minus an
    offset considering the time for activation of an actor, N_A being the number of actors in the setup, and max_actions
    the maximum number of actions for one actor per time step.
    """

    def __init__(self,
                 mtt_tracker,
                 actors,
                 T,
                 actor_model_dict,
                 temporal_point_predictor=None,
                 spatial_point_predictor=None,
                 max_actions=100):
        """Initializes the heuristic controller.

        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. Multitarget tracker to be used for tracking the particles. For the required signature of
            the callable see AbstractController.control_from_measurements(...). For simulation and evaluation purpose,
            you can alternatively pass an AbstractParticleSimulator object that simulates the particles' motion. If you
            pass a DummyMTTTracker, only calls via control_from_states(...) are permitted.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param actor_model_dict: A dict of kwargs with settings for ActorModelVariantB2.
        :param temporal_point_predictor: None or a function with signature (motion_state, actor_pos) -> delta_toa. See
            self._default_temporal_point_predictor for details on the required input and output format. If None, the
            default model (assuming constant-velocity behavior and states) is used.
        :param spatial_point_predictor: None or a function with signature (motion_state, actor_pos) -> y_at_toa. See
            self._default_spatial_point_predictor for details on the required input and output format. If None, the
            default model (assuming constant-velocity behavior and states) is used.
        :param max_actions: An integer, the maximum number of actions the controller can transmit to each actor during
            one control cycle (one time step). Correlates to the ratio of T and the time for one actor cycle. For
            example, if actors are fast enough to be activated in 10 times during one controller time step, max_actions
            should be at least 10.
        """
        super().__init__(mtt_tracker=mtt_tracker,
                         actors=actors,
                         T=T,
                         actor_model_dict=actor_model_dict,
                         temporal_point_predictor=temporal_point_predictor,
                         spatial_point_predictor=spatial_point_predictor,
                         max_actions=max_actions,
                         )

    def _assign_particle_to_actor(self,
                                  decision_mask,
                                  delta_toa,
                                  particle_class,
                                  particle_idx,
                                  t_actors_next_time_ready,
                                  ):
        return self._shoot_out_at_first_available_actor(
            decision_mask,
            delta_toa,
            particle_idx,
            t_actors_next_time_ready,
        )
