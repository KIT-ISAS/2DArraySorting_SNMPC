from absl import logging
from typing import List, Optional

import numpy as np
from collections import deque

from dataclasses import dataclass

from adf.adf_forward_backward import ADFOLFCostMixin
from heuristics import FirstActorFirstHeuristicController
from ptcr_particle_ordering import AbstractParticleOrderingScheme
from ptcr_tree_search import (
    search_activation_sequences_fast,
    tree_numba_enabled,
    warmup_ptcr_tree_numba,
)
from adf_mc_validation.adf_mc_controller_hooks import AdfMcComparisonHooks
from ptcr_config import (
    apply_adf_runtime_settings,
    apply_tree_search_runtime_settings,
    normalize_adf_config,
    normalize_tree_search_config,
)


@dataclass
class NodeData:
    """Data class representing information associated with a node in the scheduling process.

    A node in this case can be thought of an assignment of a particles to an actor. That is, the nodes of the same depth
    represent the same particle being assigned to different actors (or not being assigned at all).

    Attributes:
        depth: An integer, the depth of the node in the tree structure, i.e., which particle is being considered for
            assignment.
        assigned_actor_index: A list of integers of length depth, the indexes of the actors to which the particles to
            which the particles have been assigned to on the "path" from the root to this node.
    """
    depth: int
    assigned_actor_index: List[int]


class PTCRController(ADFOLFCostMixin):
    """Partial time-continuous controller with rollout for multi-actor optical sorting (variant B).

    The controller decides among alternatives created by a tree search over the particles to be ejected and the actors
    that eject them. The tree is built by iterating over the possible assignments of the particles to be ejected to the
    actors that can eject them and checking for conflicts in the resulting activation times. The action sequence
    resulting in the best cost-to-go is then selected, i.e., the control sequence with the lowest expected cumulative
    cost over the control horizon. To avoid combinatorial explosion, the tree is only built up to a certain depth N_R
    and for the remaining particles the First-Actor-First (FAF) heuristic is used for the rollout part to assign them to
    the actors.

    These controllers (variant B) in each time step compute a control

        u_k = [ u_{1,k}   ...   u_{N_A,k} ]^T

    with u_{j,k} in [0, inf_value] and N_A being the number of actors in the setup and u_{j,k} being the time to
    activate the actors as delta w.r.t. the current time step.

    Cost path:

        The controller uses the ADF open-loop feedback cost (Paper) via
        :class:`~adf.adf_forward_backward.ADFOLFCostMixin` (Gauß–Taylor maps + truncation
        boxes). Costs are always OLF; the legacy approximate existence-chain / CLF path is
        not part of this controller anymore.
    """

    def __init__(self,
                 mtt_tracker,
                 actors,
                 T,
                 N,
                 particle_model_dict,
                 actor_model_dict,
                 contact_model_dict,
                 accept_cost_weight=0.5,
                 reject_cost_weight=0.5,
                 use_only_terminal_costs=True,
                 use_olf=True,
                 particle_ordering='id',
                 N_R=1,
                 internal_T=None,
                 max_branches=np.inf,
                 retain_faf_candidate=False,
                 seed=None,
                 debug=False,
                 faf_kwargs=None,
                 adf=None,
                 tree_search=None,
                 ):
        """Initializes the PTCR controller.

        :param mtt_tracker: A callable, a child instance of AbstractParticleSimulator, or an instance of
            DummyMTTTracker. Multitarget tracker to be used for tracking the particles. For the required signature of
            the callable see AbstractController.control_from_measurements(...). For simulation and evaluation purpose,
            you can alternatively pass an AbstractParticleSimulator object that simulates the particles' motion. If you
            pass a DummyMTTTracker, only calls via control_from_states(...) are permitted.
        :param actors: The actors or an ActorSimulator object, the actors to control. This is only used to synchronize
            the actors and the internal actor model at the beginning of each time step. For prediction, the internal
            actor model is used.
        :param T: A float representing the time interval between two consecutive time steps.
        :param N: An integer or np.inf, the control horizon. Note that for this controller, the time interval between
            two consecutive time steps within the control horizon is given by internal_T, which can be different from T.
            Therefore, the control horizon has length (N-1)*internal_T not (N-1)*T, as usually.
        :param particle_model_dict: A dict of kwargs with settings for ParticleModel.
        :param actor_model_dict: A dict of kwargs with settings for ActorModelVariantB.
        :param contact_model_dict: A dict of kwargs. Accepted for interface compatibility; the
            concrete ADF contact model is built by the ADF mixin, so this dict is currently unused.
        :param accept_cost_weight: The weighting factor for the costs associated with ejected accept particles.
        :param reject_cost_weight: The weighting factor for the costs associated with not ejected reject particles.
        :param use_only_terminal_costs: A Boolean, whether to apply the costs for the particles only at the last stage
            and use no step costs (True) or apply step costs in every time step during the control horizon but do not
            apply terminal costs (False).
        :param use_olf: A Boolean, whether to use an open-loop feedback control (OLF) approach (True) or not (False)
            on the existence / base-class path (``delta_k_est``). On the ADF cost path, costs remain OLF regardless.
        :param particle_ordering: A string, the name of the particle ordering scheme to be used to determine the order in
            which the particles are considered for building the tree. See the name of particle ordering schemes in
            particle ordering for available strings.
        :param N_R: An integer, the depth of (number of considered particles in) the full search tree.
        :param internal_T: None or a float representing the time interval between two consecutive time steps within the
            OLF control horizon. If None, it is set to actors.t_cycle, that is, the maximum allowed time interval
            between two consecutive time steps. Note that a higher internal_T leads to a lower number of time steps and
            control sequences to consider and thus faster computation. Its only downside is that if use_olf = False and
            internal_T != T, the considered measurements are asynchronous to measurements as if they were arriving in
            the real system.
        :param max_branches: An integer or np.inf, the maximum considered number of valid branches (actor assignments)
            for each particle in the search tree. Collecting more actor assignments for this particle is stopped once
            max_branches valid assignments are found. This is a simple way yet effective way to limit the computational
            cost of the controller while maintaining appropriate accuracy.
        :param retain_faf_candidate: A Boolean, if True, append the standalone FAF control sequence to the candidate
            set after tree search (if not already present). Restores the rollout guarantee that the selected sequence
            is not worse than FAF under the ADF cost. Default False.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        :param debug: A Boolean, whether to enable debug mode (True) or not (False). In debug mode, additional variables
            are saved for comparing costs and control sequences, especially for comparison to FAF.
        :param faf_kwargs: A dict of kwargs with settings for the first actor first heuristic controller used for the
            rollout part. See the first actor first heuristic controller for available settings.
        :param adf: None or a nested dict (``use_numba``,
            ``bvn_rect_cdf_backend``, ``mc_comparison``); see :mod:`ptcr_config`.
        :param tree_search: None or a nested dict (``use_fast``, ``use_numba``).
        """
        adf_cfg = normalize_adf_config(adf)
        tree_cfg = normalize_tree_search_config(tree_search)
        apply_adf_runtime_settings(adf_cfg)
        apply_tree_search_runtime_settings(tree_cfg)
        if particle_ordering not in AbstractParticleOrderingScheme.particle_ordering_scheme_registry.keys():
            raise ValueError('particle_ordering must be one of ' + ', '.join(
                AbstractParticleOrderingScheme.particle_ordering_scheme_registry.keys()) + ', but is {}.'.format(
                particle_ordering))
        if not isinstance(N_R, int) or N_R < 1:
            raise ValueError(
                'The the depth of (number of considered particles in) the full search tree N_R must be an integer '
                'greater than or equal to 1.')
        if not np.isscalar(T) or T <= 0:
            raise ValueError(
                'The time interval between two consecutive time steps T must be a positive scalar.')
        if internal_T is not None and (not np.isscalar(internal_T) or internal_T <= 0):
            raise ValueError(
                'The internal time interval between two consecutive time steps internal_T must be a positive scalar.')
        if not np.isposinf(max_branches) and (
                not isinstance(max_branches, int) or max_branches < 1):
            raise ValueError(
                'The maximum number of branches to consider max_branches must be a positive integer or np.inf.')

        internal_T = internal_T if internal_T is not None else actors.t_cycle
        self._external_T = T
        self._N_R = N_R

        super().__init__(
            mtt_tracker=mtt_tracker,
            actors=actors,
            T=internal_T,  # use the internal time interval for the control horizon
            N=N,
            particle_model_dict=particle_model_dict,
            actor_model_dict=actor_model_dict,
            contact_model_dict=contact_model_dict,
            N_limited_look_ahead=None,  # not used in this controller
            accept_cost_weight=accept_cost_weight,
            reject_cost_weight=reject_cost_weight,
            use_only_terminal_costs=use_only_terminal_costs,
            use_mle_for_existence_meas=True,  # base class currently only supports MLE virtual measurements
            use_olf=use_olf,
            delta_k_est=1 if not use_olf else np.inf,
            seed=seed,
        )

        self._particle_ordering = AbstractParticleOrderingScheme.particle_ordering_scheme_registry[particle_ordering](
            end_array=self.actor_model.end_array)
        self._max_branches = max_branches
        self._retain_faf_candidate = bool(retain_faf_candidate)

        self._rollout_policy = FirstActorFirstHeuristicController(
            mtt_tracker=mtt_tracker,
            actors=actors,
            T=internal_T,
            actor_model_dict=actor_model_dict,
            **faf_kwargs if faf_kwargs is not None else {}
        )

        self._num_actors = self.actor_model.pos.shape[0]  # just a helper

        if not use_olf:
            raise NotImplementedError(
                "Only open-loop feedback control (use_olf=True) is supported: the ADF "
                "cost path is always OLF; the non-ADF existence path is not available."
            )
        self._use_fast_tree = tree_cfg["use_fast"]

        if self._use_fast_tree and tree_numba_enabled():
            warmup_ptcr_tree_numba(
                num_actors=self._num_actors,
                t_lead=float(self.actor_model.t_lead),
                T_step=float(self._T),
                t_cycle=float(self.actor_model.t_cycle),
            )

        self.debug = debug
        if debug:
            self.number_control_sequences = []  # List to store the number of unique control sequences considered
            self.considered_control_sequences = []  # List to store all created/considered control sequences
            self.optimal_activation_delta_t = []  # List to store the optimal sequences for each time step that are
            # actually executed
            self.all_costs = []  # List to store the costs for each time step that are actually executed
            self.faf_costs = []  # List to store the FAF costs for each time step that are actually executed
            self.faf_control_sequences = []  # List to store the FAF control sequences for each time step that are
            # actually executed

        self._adf_mc_hooks = AdfMcComparisonHooks(self, adf_cfg["mc_comparison"])

    @property
    def T(self):
        """The external controller time interval between consecutive calls.

        :returns: A float, the time interval ``T`` used for synchronizing with the
            simulation / MTT (may differ from the internal OLF stage length).
        """
        return self._external_T

    def control_from_states(self,
                            particle_states,
                            particle_class,
                            particle_id,
                            t_act,
                            ):
        """Returns the actors' input as output of the controller's decision.

        Format particle_states:

            [estimated_motion_state_mean, estimated_motion_state_cov, track_score],

            with estimated_motion_state_mean being a np.array of shape [num_particles, 2] and format
            [position, velocity] containing the expected values of the motion state and estimated_motion_state_cov a
            np.array of shape [num_particles, 2, 2] containing the covariances of the motion state. The track_score is
            a np.array of shape [num_particles] used as the existence probabilities of the particles.

        :param particle_states: A tuple of length 3.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns activation_delta_t: A np.array of shape [num_actors, 1], the time to activate the actors as delta
            w.r.t. the current time step.
        """
        estimated_motion_state_mean, estimated_motion_state_cov, track_score = particle_states
        init_t_act = t_act

        activation_delta_t = np.empty((len(self._actor_model.pos), 1))
        activation_delta_t[:] = np.inf

        if estimated_motion_state_mean.size == 0:
            return activation_delta_t

        sorted_particle_class, sorted_delta_toa, sorted_motion_state_mean = self._prepare_sorted_toa(
            particle_states, particle_class, particle_id, estimated_motion_state_mean
        )

        control_sequences_delta_t, faf_control_sequence = self._build_candidate_sequences(
            sorted_particle_class,
            sorted_delta_toa,
            init_t_act,
            estimated_motion_state_mean,
            particle_class,
            particle_id,
        )

        costs, min_index, min_costs, init_existence_probs = self._evaluate_sequence_costs(
            control_sequences_delta_t,
            init_t_act,
            estimated_motion_state_mean,
            estimated_motion_state_cov,
            track_score,
            particle_class,
            sorted_motion_state_mean.shape[0],
        )

        activation_delta_t[:] = control_sequences_delta_t[min_index][:, 0].reshape(self._num_actors, 1)

        if self.debug:
            self._log_information(
                control_sequences_delta_t,
                activation_delta_t,
                min_costs,
                estimated_motion_state_mean,
                estimated_motion_state_cov,
                init_existence_probs,
                particle_class,
                particle_id,
                init_t_act,
                faf_control_sequence=faf_control_sequence,
            )

        costs_for_hooks = costs
        self._adf_mc_hooks.maybe_cost_eval(
            control_sequences_delta_t=control_sequences_delta_t,
            costs_adf=costs_for_hooks,
            estimated_motion_state_mean=estimated_motion_state_mean,
            estimated_motion_state_cov=estimated_motion_state_cov,
            init_existence_probs=init_existence_probs,
            particle_class=particle_class,
            t_act=init_t_act,
        )
        self._adf_mc_hooks.maybe_validate(
            u_star=control_sequences_delta_t[min_index:min_index + 1],
            estimated_motion_state_mean=estimated_motion_state_mean,
            estimated_motion_state_cov=estimated_motion_state_cov,
            particle_class=particle_class,
            particle_id=particle_id,
            t_act=init_t_act,
        )

        return activation_delta_t

    def _prepare_sorted_toa(
        self, particle_states, particle_class, particle_id, estimated_motion_state_mean
    ):
        """Predicts TOAs and returns particle data sorted by the ordering scheme.

        :param particle_states: The particle state tuple passed to :meth:`control_from_states`.
        :param particle_class: An integer np.array of shape [num_particles].
        :param particle_id: An integer np.array of shape [num_particles].
        :param estimated_motion_state_mean: A np.array of shape [num_particles, state_dim].

        :returns: A tuple
            ``(sorted_particle_class, sorted_delta_toa, sorted_motion_state_mean)`` where
            ``sorted_delta_toa`` has shape ``[num_actors, num_particles]``.
        """
        delta_toa = self._rollout_policy.predict_time_of_arrival(estimated_motion_state_mean)
        sort_order = self._particle_ordering(
            particle_states=particle_states,
            particle_id=particle_id,
            particle_class=particle_class,
        )
        return (
            particle_class[sort_order],
            delta_toa[:, sort_order],
            estimated_motion_state_mean[sort_order],
        )

    def _build_candidate_sequences(
        self,
        sorted_particle_class,
        sorted_delta_toa,
        init_t_act,
        estimated_motion_state_mean,
        particle_class,
        particle_id,
    ):
        """Runs tree search, postprocessing, and optional FAF retention.

        :param sorted_particle_class: Sorted particle classes.
        :param sorted_delta_toa: Sorted TOAs of shape ``[num_actors, num_particles]``.
        :param init_t_act: A np.array of shape ``[num_actors]``, actors' internal time.
        :param estimated_motion_state_mean: Unsorted motion means (for FAF).
        :param particle_class: Unsorted particle classes (for FAF).
        :param particle_id: Unsorted particle ids (for FAF).

        :returns: A tuple ``(control_sequences_delta_t, faf_control_sequence)`` where
            ``control_sequences_delta_t`` has shape ``[num_sequences, num_actors, N]`` and
            ``faf_control_sequence`` is None or the standalone FAF sequence.
        """
        possible_control_sequences = self._search_for_possible_activation_sequences(
            particle_class=sorted_particle_class,
            delta_toa=sorted_delta_toa,
            t_act=init_t_act,
        )
        control_sequences_delta_t = self._postprocess_control_sequences(possible_control_sequences)
        faf_control_sequence = None
        if self._retain_faf_candidate:
            control_sequences_delta_t, faf_control_sequence = self._ensure_faf_in_candidates(
                control_sequences_delta_t,
                estimated_motion_state_mean,
                particle_class,
                particle_id,
                init_t_act,
            )
        return control_sequences_delta_t, faf_control_sequence

    def _evaluate_sequence_costs(
        self,
        control_sequences_delta_t,
        init_t_act,
        estimated_motion_state_mean,
        estimated_motion_state_cov,
        track_score,
        particle_class,
        num_sorted_particles,
    ):
        """Evaluates candidate sequences with ADF costs.

        :param control_sequences_delta_t: A np.array of shape ``[num_sequences, num_actors, N]``.
        :param init_t_act: A np.array of shape ``[num_actors]``.
        :param estimated_motion_state_mean: A np.array of shape ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param track_score: A np.array of shape ``[num_particles]``, existence / track scores.
        :param particle_class: An integer np.array of shape ``[num_particles]``.
        :param num_sorted_particles: An integer, used when costs are skipped (single candidate).

        :returns: A tuple ``(costs, min_index, min_costs, init_existence_probs)`` where
            ``costs`` may be None if evaluation was skipped.
        """
        need_costs = (
            len(control_sequences_delta_t) > 1
            or self.debug
            or self._adf_mc_hooks.cost_eval
        )
        if not need_costs:
            return (
                None,
                0,
                None,
                np.ones(num_sorted_particles),
            )

        init_existence_probs = track_score
        costs = self._adf_cum_cost(
            control_sequences_delta_t,
            init_t_act,
            estimated_motion_state_mean,
            estimated_motion_state_cov,
            init_existence_probs,
            particle_class,
        )
        min_index = int(np.argmin(costs))
        return costs, min_index, costs[min_index], init_existence_probs

    def finalize_adf_mc_cost_eval(self, save_path: Optional[str] = None, show: Optional[bool] = None):
        """Writes the final ADF/MC cost-eval summary at the end of a simulation run.

        Delegates to :meth:`AdfMcComparisonHooks.finalize_cost_eval`.

        :param save_path: None or a string, path for the summary PNG.
        :param show: None or a Boolean override for ``plt.show()``.

        :returns: See :meth:`AdfMcComparisonHooks.finalize_cost_eval`.
        """
        return self._adf_mc_hooks.finalize_cost_eval(save_path=save_path, show=show)

    def _search_for_possible_activation_sequences(self, delta_toa, particle_class, t_act):
        """Fills the search tree for the given particle states and returns the valid activation sequences.

        Each node represents either a particle-actuator pair or the decision not to assign an actuator to that particle.
        For each possible node in the tree, namely each combination of a particle p_i and an actuator a_j, the
        following conditions must be satisfied to be considered a valid particle-actuator assignment.

        Ejection Condition 1. Be an undisturbed particle!

            A particle’s trajectory must be undisturbed by upstream actuators. That is, for every actuator a_l that the
            particle p_i passes before reaching a_j, it must hold that Status(a_l , tau_l^i ) ∉ {HIT, UP, DOWN} ,
            where Status(a_l , tau_j^i )describes the status of an actuator a_l at the time tau_l^i when particle p_i
            passes it.

        Ejection Condition 2. Be Ready!

            Actuator a_j must be in ready state when it should be activated. This is expressed by the condition
            Status(a_j , tau_j^i  − t_lead ) = READY , where tau_j^i − t_lead describes the time actuator a_j must be
            activated to hit particle p_i at time tau_j^i.

        Ejection Condition 3. Be Free!

            a) The actuator must be able to complete its full activation cycle before its next scheduled activation. The
            condition is expressed as act tau_j^i + t_cycle − t_lead < NextActivation(a_j , current_path), with
            tau_j^i + t_cycle − t_lead being the time at which actuator a_j finishes its cycle when it would be chosen
            to eject p_i. The function NextActivation(a_j , current_path) returns for the current path the earliest
            activation that is  already scheduled for actuator a_j that lays after the currently evaluated activation.
            The next activations are prescribed by earlier actuator assignments that were made to this actuator
            previously in the current path.

            b) Similarly, a_j should not be activated before any previous cycle is finished, formulated as
            PreviousCycleEnd(a_j , current_path) < tau_j^i  − t_lead , with PreviousCycleEnd(a_j , current_path) as the
            latest time actuator a_j finishes one of its cycles that was started before the currently evaluated
            activation w.r.t. the current path. Again, the earlier assignments are assignments that were made previously
            in current_path.

        Ejection Condition 4. Be allowed to activate!

            Each actuator may be activated at most once within its cycle time. This constraint ensures that a path
            through the tree constructed over particles can subsequently be transformed into a valid control sequence
            defined over the time intervals of length t_cycle and naturally prescribed by the actuator dynamics.

        Note that ejection conditions 1 and 2 are checks based on the actor states that result from previous control
        decisions (i.e., from applied controls in the time steps before). In particular, condition 2 is entirely based
        on the current actor states and does not depend on the current path, while condition 1 is based on both the
        current actor states but also on the current particle-actor assignment we are looking at (since we need to
        determine which actors are "upstream"). Ejection conditions 3 and 4 are checks based only on current_path that
        describes how the actor state evolve in the control horizon (i.e., from virtual scenarios created by the tree
        search).

        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns control_sequences: A np.array of shape [num_action_sequences, num_actors, N], the time to activate the
            actors as delta w.r.t. the initial time step for each control_sequence. The activations times are already
            split by the time steps.

        Fast path (default): compact BFS in ``ptcr_tree_search`` (scalar/Numba checks, reconstruct at leaves).
        Set config ``tree_search.use_fast=false`` for the original Python BFS;
        ``tree_search.use_numba=false`` disables only the Numba expand helper.
        """
        # find the eject particles to build the tree over
        ejectable_mask = particle_class == 1
        delta_toa = delta_toa[:, ejectable_mask]   # update delta_toa

        # Ejection Condition 1. Be an undisturbed particle! (partially)
        undisturbed_particle_actor_pairs_mask = self._partially_check_ejection_condition_be_an_undisturbed_particle(
            delta_toa, t_act)  # shape [num_actors, num_considered_particles]
        # note that the reduction does not hold true, i.e., np.all(undisturbed_particle_actor_pairs_mask, axis=0) is too
        # restrictive, since it can be that a particle is ejected (by a scheduled activation of the tree search) before
        # it passes and actor that would disturb it. Therefore, we check if the particle is undisturbed only for
        # "upstream" actors, once we decide to assign an actor to this particle in the tree search and therefore know
        # which actors are "upstream". This test is carried out below.

        # Ejection Condition 2. Be Ready!
        valid_particle_actor_pairs_mask = self._check_ejection_condition_be_ready(delta_toa, t_act)
        # shape [num_actors, num_considered_particles]

        # if there is no actor that can eject the particle, we can directly exclude it from the tree search
        at_least_one_actor_valid_mask = np.any(valid_particle_actor_pairs_mask , axis=0)
        # update delta_toa, undisturbed mask, and valid mask
        delta_toa = delta_toa[:, at_least_one_actor_valid_mask]
        undisturbed_particle_actor_pairs_mask = undisturbed_particle_actor_pairs_mask[:, at_least_one_actor_valid_mask]
        valid_particle_actor_pairs_mask = valid_particle_actor_pairs_mask[:, at_least_one_actor_valid_mask]

        if self._use_fast_tree:
            return search_activation_sequences_fast(
                delta_toa=delta_toa,
                particle_class=particle_class,
                t_act=t_act,
                undisturbed_mask=undisturbed_particle_actor_pairs_mask,
                valid_mask=valid_particle_actor_pairs_mask,
                t_lead=float(self.actor_model.t_lead),
                T_step=float(self._T),
                t_cycle=float(self.actor_model.t_cycle),
                N_R=int(self._N_R),
                N_horizon=int(self._N),
                max_branches=self._max_branches,
                num_actors=int(self._num_actors),
                perform_rollout=self._perform_rollout_for_unassigned_particles,
            )

        return self._search_for_possible_activation_sequences_legacy(
            delta_toa=delta_toa,
            t_act=t_act,
            undisturbed_particle_actor_pairs_mask=undisturbed_particle_actor_pairs_mask,
            valid_particle_actor_pairs_mask=valid_particle_actor_pairs_mask,
        )

    def _search_for_possible_activation_sequences_legacy(
        self,
        delta_toa,
        t_act,
        undisturbed_particle_actor_pairs_mask,
        valid_particle_actor_pairs_mask,
    ):
        """Original Python BFS tree search (fallback when ``tree_search.use_fast`` is false).

        Called from :meth:`_search_for_possible_activation_sequences` after the
        undisturbed / ready masks have already been applied and empty particles
        removed.

        :param delta_toa: A np.array of shape ``[num_actors, num_considered_particles]``,
            times of arrival as deltas w.r.t. the current time step (already filtered
            to ejectable / ready particles).
        :param t_act: A np.array of shape ``[num_actors]``, actors' current internal time.
        :param undisturbed_particle_actor_pairs_mask: A Boolean np.array of shape
            ``[num_actors, num_considered_particles]``, partial ejection-condition-1 mask.
        :param valid_particle_actor_pairs_mask: A Boolean np.array of shape
            ``[num_actors, num_considered_particles]``, ejection-condition-2 (ready) mask.

        :returns: A list of np.arrays of shape ``[num_actors, N]``, candidate control
            sequences as activation deltas w.r.t. the initial time step.
        """
        # initialize the breadth-first search through the tree (tree checks conditions 3 and 4)
        # initialize the queue
        queue: deque[NodeData] = deque()
        queue.append(NodeData(0, []))

        # breadth-first search through the tree
        control_sequences = []
        while queue:
            current_node = queue.popleft()
            current_depth = current_node.depth
            # if we are finished (all particles have been looked at)
            if current_depth == delta_toa.shape[1]:
                # reconstruct control sequence from actor_activation_times
                control_sequences.append(self._control_sequence_from_node(current_node, delta_toa))
                continue

            # rollout and continue if we reached the tree depth N_R
            if current_depth > self._N_R - 1:
                # reconstruct control sequence from actor_activation_times
                control_sequence = self._control_sequence_from_node(current_node, delta_toa)
                # perform the rollout for the unassigned particles and get the resulting control sequence
                control_sequence = self._perform_rollout_for_unassigned_particles(
                    control_sequence=control_sequence,
                    assigned_particle_idxs=np.array([], dtype=int),  # we mask them directly here, see below
                    delta_toa=delta_toa[:, current_depth:],
                    particle_class=np.ones(delta_toa.shape[1] - current_depth),
                    t_act=t_act)
                control_sequences.append(control_sequence)
                continue

            # Ejection Condition 2. Be Ready!
            actor_indexes_to_check = np.nonzero(valid_particle_actor_pairs_mask[:, current_depth])[0]  # directly
            # exclude the actors that are not valid for this particle from the tree search

            # iterate over all potential actors + one additional option for no actor selected
            added_branches = 0
            for actor_idx in (*actor_indexes_to_check, self._num_actors):

                # if we already have collected enough control sequences, we stop collecting more, note that if using
                # this option, there is an additional dependency on the actor ordering
                if added_branches >= self._max_branches:
                    break

                # no actor selected
                if actor_idx == self._num_actors:
                    child = NodeData(
                        depth=current_depth + 1,
                        assigned_actor_index=current_node.assigned_actor_index + [-1],
                    )
                    queue.append(child)
                    continue

                current_toa = delta_toa[actor_idx, current_depth]  # toa for current particle-actor pair
                current_activation_time = current_toa - self.actor_model.t_lead
                current_activation_time_step = int(current_activation_time / self._T)
                # note that self._T is the internal time interval for the control horizon, while
                # self.T = self._external_T is the time interval between the time steps in which the controller is
                # called

                # Ejection Condition 1. Be an undisturbed particle! (partially)
                upstream_toas_mask = delta_toa[:, current_depth] <= current_toa  # shape [num_actors]
                if not np.all(undisturbed_particle_actor_pairs_mask[upstream_toas_mask, current_depth]):
                    continue

                # calculate the already assigned activation times for this actor based on the current path in the tree
                already_assigned_activation_times_for_actor = delta_toa[np.full(
                    np.count_nonzero(np.asarray(current_node.assigned_actor_index, dtype=int) == actor_idx),
                    actor_idx), :current_depth] - self.actor_model.t_lead
                # the actor slice very likely contains inf values (when other actors are used for the assigned
                # particles), we remove them since they do not represent actual assigned activation times
                already_assigned_activation_times_for_actor = already_assigned_activation_times_for_actor[
                    np.logical_not(np.isinf(already_assigned_activation_times_for_actor))]

                # Ejection Condition 4. Be allowed to activate!
                activation_time_steps = np.floor(already_assigned_activation_times_for_actor / self._T).astype(int)
                if np.any(activation_time_steps == current_activation_time_step):
                    # if there is already an activation scheduled for this time step and this actor, we cannot activate
                    # it a second time within the same time step
                    continue

                # Ejection Condition 3. Be Free!
                if not self._check_ejection_condition_be_free(current_activation_time,
                                                              already_assigned_activation_times_for_actor):
                    continue

                added_branches += 1
                child = NodeData(
                    depth=current_depth + 1,
                    assigned_actor_index=current_node.assigned_actor_index + [actor_idx],
                )
                queue.append(child)

        if len(control_sequences) == 0:
            return [np.full((self._num_actors, 1), np.inf)]

        return np.asarray(control_sequences)

    def _partially_check_ejection_condition_be_an_undisturbed_particle(self, delta_toa, t_act):
        """Checks ejection condition 1, whether the particle is an undisturbed particle based on the already scheduled
        activations (given by t_act).

        Ejection Condition 1. Be an undisturbed particle!

            A particle’s trajectory must be undisturbed by upstream actuators. That is, for every actuator a_l that the
            particle p_i passes before reaching a_j, it must hold that Status(a_l , tau_l^i ) ∉ {HIT, UP, DOWN} ,
            where Status(a_l , tau_j^i )describes the status of an actuator a_l at the time tau_l^i when particle p_i
            passes it.

        This check is only done partially, that is, we only check for potential disturbances based on the current actor
        states and the particle arrival times, but we do not yet check if the particle would actually pass the actor
        that disturbs it, i.e., we do not check if the actors are "upstream" of the particle. This will be done in the
        next step in the tree search when we know which actors are "upstream".

        :param delta_toa: A np.array of shape [num_actors, num_considered_particles], the particles' time of arrival as
            delta w.r.t. the current time step.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns: A Boolean np.array of shape [num_actors, num_considered_particles], where each component is True if
            the particle is undisturbed (and not hit) by the actors.
        """
        # an actor does not disturb any particle if it is not activated (t_act == 0) or if it is already finished with
        # its DOWN status
        time_until_reset = self.actor_model.t_cycle - self.actor_model.t_reset
        resting_actors = np.logical_or(t_act == 0, t_act >= time_until_reset)

        # check if a particle will be disturbed by an actor triggered in the loop before if activated so that it will
        # be hit at delta_toa. Here, we do not yet check if the particle would pass the actor that disturbs it, i.e.,
        # we do not check if the actors are "upstream" of the particle. This will be done in the tree search.
        undisturbed_particle_actor_pairs = np.logical_or(
            delta_toa < self.actor_model.t_activate - t_act[:, None],  # particle arrives before actor is in UP status
            delta_toa > (time_until_reset - t_act)[:, None]  # particle arrives after actor is finished with DOWN status
        )
        return np.logical_or(resting_actors[:, None], undisturbed_particle_actor_pairs)

    def _check_ejection_condition_be_ready(self, delta_toa, t_act):
        """Checks ejection condition 2, whether the particle is ready based on the already scheduled activations (given
        by t_act).

        Ejection Condition 2. Be Ready!

            Actuator a_j must be in ready state when it should be activated. This is expressed by the condition
            Status(a_j , tau_j^i  − t_lead ) = READY , where tau_j^i − t_lead describes the time actuator a_j must be
            activated to hit particle p_i at time tau_j^i.

        :param delta_toa: A np.array of shape [num_actors, num_considered_particles], the particles' time of arrival as
            delta w.r.t. the current time step.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns: A Boolean np.array of shape [num_actors, num_considered_particles], where each component is True if
            the particle-actor pair is a valid assignment.
        """
        # an actor is ready if it can hit at the particle arrival time
        actors_next_time_ready = np.where(t_act == 0, np.zeros(len(t_act)),
                                          self._actor_model.t_cycle - t_act)
        return delta_toa >= actors_next_time_ready[:, None] + self.actor_model.t_lead

    def _check_ejection_condition_be_free(self, activation_time, already_assigned_activation_times_for_actor):
        """Checks ejection condition 3, whether the particle is free based on the already scheduled activations in the
        path (given by current_node).

            Ejection Condition 3. Be Free!

            a) The actuator must be able to complete its full activation cycle before its next scheduled activation. The
            condition is expressed as act tau_j^i + t_cycle − t_lead < NextActivation(a_j , current_path), with
            tau_j^i + t_cycle − t_lead being the time at which actuator a_j finishes its cycle when it would be chosen
            to eject p_i. The function NextActivation(a_j , current_path) returns for the current path the earliest
            activation that is  already scheduled for actuator a_j that lays after the currently evaluated activation.
            The next activations are prescribed by earlier actuator assignments that were made to this actuator
            previously in the current path.

            b) Similarly, a_j should not be activated before any previous cycle is finished, formulated as
            PreviousCycleEnd(a_j , current_path) < tau_j^i  − t_lead , with PreviousCycleEnd(a_j , current_path) as the
            latest time actuator a_j finishes one of its cycles that was started before the currently evaluated
            activation w.r.t. the current path. Again, the earlier assignments are assignments that were made previously
            in current_path.

        :param activation_time: A float, the time when to activate the actor as delta w.r.t. the current time step.
        :param already_assigned_activation_times_for_actor: A np.array of shape
            [num_activations_for_this_actor_in_current_path], the activation times that are already assigned to this
            actor in the current path as delta w.r.t. the current time step.

        :returns: A Boolean, True if the particle is free based on the current path and the time that remains until the
            actors have finished their current cycle.
        """
        # get the latest time when a previous cycle of this actor finishes based on the already assigned activations
        latest_previous_cycle_finished_time = np.max(
            already_assigned_activation_times_for_actor[already_assigned_activation_times_for_actor <= activation_time],
            initial=-np.inf) + self.actor_model.t_cycle

        # check if it is ready again
        no_conflicting_previous_activation = latest_previous_cycle_finished_time <= activation_time

        # get the earliest time when the next activation of this actor starts based on the already assigned activations
        next_following_activation_time = np.min(
            already_assigned_activation_times_for_actor[already_assigned_activation_times_for_actor > activation_time],
            initial=np.inf)

        # check if actor finished before the next activation
        no_conflicting_subsequent_activation = (
                    activation_time + self.actor_model.t_cycle <= next_following_activation_time)

        return no_conflicting_previous_activation and no_conflicting_subsequent_activation

    def _control_sequence_from_node(self, current_node, delta_toa):
        """
        Builds a control-sequence array from a tree node assignment path.

        :param current_node: A :class:`NodeData` instance with ``depth`` and
            ``assigned_actor_index``.
        :param delta_toa: A np.array of shape ``[num_actors, num_considered_particles]``,
            arrival-time deltas for the particles in the tree.
        :param kwargs: Unused extra keyword arguments.

        :returns: A np.array of shape ``[num_actors, N]``, the activation deltas for the
            assignment encoded by ``current_node``.
        """
        actor_indexes = np.asarray(current_node.assigned_actor_index, dtype=int)
        activation_mask = actor_indexes != -1  # mask for the assigned actors (excludes the "no actor assigned" option)
        actors_to_activate = actor_indexes[activation_mask]

        activation_times = delta_toa[actors_to_activate, np.arange(current_node.depth)[
            activation_mask]] - self.actor_model.t_lead  # does not include any inf

        assert np.all(
            np.isfinite(activation_times)), 'The activation_times for the control sequence contain inf/nan values'

        time_steps = np.floor(activation_times / self._T).astype(int)

        if np.any(time_steps >= self._N):
            mask = time_steps < self._N
            logging.warning(
                'Activation time step {} exceeds control horizon of length {}. Truncating to fit. '
                'Consider increasing N to avoid truncation.'.format(time_steps[np.logical_not(mask)], self._N))
            time_steps = time_steps[mask]
            activation_times = activation_times[mask]
            actors_to_activate = actors_to_activate[mask]

        control_sequence = np.full((delta_toa.shape[0], self._N), np.inf)
        control_sequence[actors_to_activate, time_steps] = activation_times

        return control_sequence

    def _perform_rollout_for_unassigned_particles(self,
                                                  control_sequence,
                                                  assigned_particle_idxs,
                                                  delta_toa,
                                                  particle_class,
                                                  t_act,
                                                  ):
        """Performs rollout using FAF for the unassigned particles and merges the rollout control sequence with the
        control sequence from the tree.

        :param control_sequence: A np.array of shape [num_actors, N], the time to activate the actors as delta w.r.t.
            the initial time step. The activations times are already split by the time steps.
        :param assigned_particle_idxs: An integer np.array of shape [num_assigned_particles], the indices of the
            particles that have already been assigned by the full tree search and thus should not be assigned by the
            rollout.
        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns control_sequence: A np.array of shape [num_actors, N], the time to activate the actors as delta w.r.t.
            the initial time step as the result of the original control sequence merged with the rollout control
            sequence. The activations times are already split by the time steps.
        """
        # mask particles and actors that are already assigned
        masked_particle_class = np.copy(particle_class)
        masked_particle_class[assigned_particle_idxs] = 0

        # determine the decision mask for the rollout policy
        decision_mask = self._rollout_policy.make_decision(
            delta_toa,
            particle_class=masked_particle_class,
            init_t_act=t_act,
            prescribed_control_sequence=control_sequence
        )

        # go from hitting times to activation times
        activation_delta_t = np.where(decision_mask, delta_toa, np.inf) - self._actor_model.t_lead
        # shape [num_actors, num_particles]

        # determine the last time step in the control horizon in which an activation from the rollout happens and
        # check if it exceeds the control horizon
        time_step_with_last_activation = int(
            np.max(activation_delta_t, initial=0.0, where=np.isfinite(activation_delta_t)) / self._T)
        if time_step_with_last_activation >= self._N - 1:
            logging.warning(
                'Activation time step {} exceeds control horizon of length {}. Truncating to fit. '
                'Consider increasing N to avoid truncation.'.format(time_step_with_last_activation, self._N))

        control_sequence_rollout = self._activation_delta_t_to_control_sequence(
            activation_delta_t, N=self._N, inf_value=np.inf)

        return self._merge_control_sequences(control_sequence_rollout, control_sequence)

    @staticmethod
    def _merge_control_sequences(control_sequence_rollout, control_sequence):
        """Merges the control sequences from the rollout and the tree.

        :param control_sequence: A np.array of shape [num_actors, N], the time to activate the actors as delta w.r.t.
            the initial time step as the result of the tree search. The activations times are already split by the time
            steps.
        :param control_sequence_rollout: A np.array of shape [num_actors, N], the time to activate the actors as delta
            w.r.t. the initial time step as the result of the rollout. The activations times are already split by the
            time steps.

        :returns control_sequence: A np.array of shape [num_actors, N], the time to activate the actors as
            delta w.r.t. the initial time step as the result of the original control sequence merged with the rollout
            control sequence. The activations times are already split by the time steps.
        """
        rollout_mask = np.isfinite(control_sequence_rollout)

        # check that rollout has no values where the original control sequence has some
        if np.any(rollout_mask[np.isfinite(control_sequence)]):
            raise RuntimeError(
                'The rollout control sequence has finite values where the original control sequence also '
                'has finite values. This indicates a severe error in the implementation of the rollout.')

        # write the rollout values into the merged control sequence where the rollout has finite values (and thus the
        # original control sequence has inf)
        control_sequence[rollout_mask] = control_sequence_rollout[rollout_mask]

        return control_sequence

    def _postprocess_control_sequences(self, control_sequences):
        """Postprocesses the control sequences to have unique activation values, and go from delta activation times
        w.r.t. the initial time step to delta activation times w.r.t. the internal time steps in the control horizon.

        :param control_sequences: A np.array of shape [num_action_sequences, num_actors, N], the time to activate the
            actors as delta w.r.t. the initial time step for each control_sequence. The activations times are already
            split by the time steps.

        :returns control_sequences_delta_t: A np.array of shape [num_action_sequences, num_actors, N], the time to
            activate the actors as delta w.r.t. the internal time steps in the control horizon.
        """
        # go from delta activation times w.r.t. the initial time step to delta activation times w.r.t. the internal time
        # steps in the control horizon
        control_sequences_delta_t = np.remainder(control_sequences,
                                                 self._T,
                                                 # note that self._T is the internal time interval for the control
                                                 # horizon, while self.T = self._external_T is the time interval between
                                                 # the time steps in which the controller is called
                                                 where=np.isfinite(control_sequences),
                                                 out=control_sequences)

        # ensure uniqueness of the activation deltas
        return np.unique(control_sequences_delta_t, axis=0)

    def _adf_cum_cost(self,
                      control_sequences_delta_t,
                      init_t_act,
                      estimated_motion_state_mean,
                      estimated_motion_state_cov,
                      init_existence_probs,
                      particle_class,
                      ):
        """
        Evaluates ADF-OLF cumulative costs for candidate control sequences.

        :param control_sequences_delta_t: A np.array of shape
            ``[num_action_sequences, num_actors, num_internal_time_steps]``, the
            activation times as deltas w.r.t. the internal stages.
        :param init_t_act: A np.array of shape ``[num_actors]``, the initial actors'
            internal times.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``, the current motion-state means.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``, the current motion-state covariances.
        :param particle_class: An integer np.array of shape ``[num_particles]``.
        :param init_existence_probs: A np.array of shape ``[num_particles]``, the initial
            existence probabilities.

        :returns: A np.array of shape ``[num_action_sequences]``, the expected cumulative
            ADF-OLF costs for each candidate sequence.
        """
        # create the probabilities [ P(exists not), P(exists) ], respectively, shape [num_particles, 2].
        # Shared across sequences — `_forward_chain` / Numba horizon broadcast without np.repeat.
        est_existence_probs = np.c_[1 - init_existence_probs, init_existence_probs]
        init_t_act = np.asarray(init_t_act, dtype=float)

        # update the particle model with the new particles
        self._particle_model.particle_class = particle_class

        return self._forward_chain(control_sequences_delta_t,
                                   estimated_motion_state_mean,
                                   estimated_motion_state_cov,
                                   est_existence_probs,
                                   t_act_k_at_T=init_t_act,
                                   )

    def _log_information(self,
                         control_sequences_delta_t,
                         u_opt,
                         min_costs,
                         estimated_motion_state_mean,
                         estimated_motion_state_cov,
                         init_existence_probs,
                         particle_class,
                         particle_id,
                         t_act,
                         faf_control_sequence=None,
                         ):
        """Logs information about the control sequences considered and the costs for the optimal sequence and the FAF
         sequence for comparison.

        :param control_sequences_delta_t: A np.array of shape [num_action_sequences, num_actors,
            num_internal_time_steps], the time to activate the actors as delta w.r.t. the internal time steps in the
            control horizon.
        :param u_opt: A np.array of shape [num_actors, 1], the time to activate the actors as delta w.r.t. the current
            time step, as the optimal control sequence that is actually executed.
        :param min_costs: A float, the costs for the optimal control sequence that is actually executed.
        :param estimated_motion_state_mean: A np.array of shape [num_particles, motion_state_length], the estimated
            motion state mean.
        :param estimated_motion_state_cov: A np.array of shape [num_particles, motion_state_length, motion_state_length],
            the estimated motion state covariance.
        :param init_existence_probs: A np.array of shape [num_particles], the initial existence probabilities.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.
        :param faf_control_sequence: None or a np.array of shape [1, num_actors, N], a precomputed FAF sequence to
            reuse (e.g., from ``_ensure_faf_in_candidates``). If None, FAF is computed here.
        """
        # store information
        self.number_control_sequences.append(len(control_sequences_delta_t))
        self.considered_control_sequences = control_sequences_delta_t
        self.optimal_activation_delta_t.append(u_opt.copy())
        self.all_costs.append(min_costs)

        # calculate the FAF control sequence and its cost for comparison (reuse if already computed)
        if faf_control_sequence is None:
            faf_control_sequence = self._faf_control_from_states(
                estimated_motion_state_mean,
                particle_class,
                particle_id,
                t_act)

        faf_in_candidates = faf_control_sequence.shape[-1] == control_sequences_delta_t.shape[-1] and np.any(
            [np.allclose(seq, faf_control_sequence[0], atol=1e-6) for seq in control_sequences_delta_t])
        if not faf_in_candidates:
            logging.info(
                f"Timestep {len(self.all_costs)}: "
                f"FAF control sequence is not in the list of considered control sequences.")

        # calculate the cost for the FAF control sequence with the same ADF cost as candidate ranking
        cost = self._adf_cum_cost(
            control_sequences_delta_t=faf_control_sequence,
            init_t_act=t_act,
            estimated_motion_state_mean=estimated_motion_state_mean,
            estimated_motion_state_cov=estimated_motion_state_cov,
            init_existence_probs=init_existence_probs,
            particle_class=particle_class,
        )[0]
        self.faf_costs.append(cost)
        self.faf_control_sequences.append(faf_control_sequence)

        # log the comparison between the optimal sequence and the FAF sequence
        if cost < min_costs:
            logging.info(
                f"Timestep {len(self.all_costs)}: "
                f"FAF control sequence is better than the optimal sequence with cost {cost} < {min_costs}.")

    def _ensure_faf_in_candidates(self,
                                 control_sequences_delta_t,
                                 estimated_motion_state_mean,
                                 particle_class,
                                 particle_id,
                                 t_act,
                                 ):
        """Appends the standalone FAF sequence to the candidate set if it is not already present.

        :param control_sequences_delta_t: A np.array of shape [num_action_sequences, num_actors, N], candidate
            sequences after postprocessing.
        :param estimated_motion_state_mean: A np.array of shape [num_particles, motion_state_length].
        :param particle_class: An integer np.array of shape [num_particles].
        :param particle_id: An integer np.array of shape [num_particles].
        :param t_act: A np.array of shape [num_actors], current actor internal times.

        :returns:
            control_sequences_delta_t: A np.array of shape [num_action_sequences', num_actors, N], candidates with FAF
                included.
            faf_control_sequence: A np.array of shape [1, num_actors, N], the standalone FAF sequence.
        """
        faf_control_sequence = self._faf_control_from_states(
            estimated_motion_state_mean,
            particle_class,
            particle_id,
            t_act,
        )

        control_sequences_delta_t = np.asarray(control_sequences_delta_t)
        if control_sequences_delta_t.ndim == 2:
            control_sequences_delta_t = control_sequences_delta_t[None, ...]

        already_present = (
            control_sequences_delta_t.shape[0] > 0
            and control_sequences_delta_t.shape[-1] == faf_control_sequence.shape[-1]
            and np.any(
                [np.allclose(seq, faf_control_sequence[0], atol=1e-6) for seq in control_sequences_delta_t])
        )
        if not already_present:
            control_sequences_delta_t = np.concatenate(
                [control_sequences_delta_t, faf_control_sequence], axis=0)

        return control_sequences_delta_t, faf_control_sequence

    def _faf_control_from_states(self, estimated_motion_state_mean, particle_class, particle_id, t_act):
        """Computes the FAF control sequence from the particle states.

        :param estimated_motion_state_mean: A np.array of shape [num_particles, motion_state_length], the estimated
            motion state mean.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param particle_id: An integer np.array of shape [num_particles], the ids of the particles.
        :param t_act: A np.array of shape [num_actors] representing the actors' current internal time.

        :returns control_sequences_delta_t: A np.array of shape [1, num_actors, num_internal_time_steps], the time to
            activate the actors as delta w.r.t. the internal time steps in the control horizon.
        """
        activation_delta_t = self._rollout_policy.control_from_states(
            (estimated_motion_state_mean,), particle_class, particle_id, t_act)

        control_sequence = self._activation_delta_t_to_control_sequence(activation_delta_t, N=self._N, inf_value=np.inf)

        # go from delta activation times w.r.t. the initial time step to delta activation times w.r.t. the internal time
        # steps in the control horizon
        return self._postprocess_control_sequences(control_sequence[None, ...])
