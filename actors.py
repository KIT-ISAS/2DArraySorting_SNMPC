from absl import logging

import numpy as np

from actor_model import ActorModelVariantB


class VirtualActor(ActorModelVariantB):
    """A simulator for the actors in continuous time. Builds upon ActorModelVariantB."""

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
        self._k += 1
        self._t_act = self.predict_actor_state(self._t_act, u_act)

    def _check_for_valid_u_act(self, t_act_k_minus_one_at_T, u_act):
        valid_u_act_mask = super()._check_for_valid_u_act(t_act_k_minus_one_at_T, u_act)

        if not self.suppress_warnings and not np.all(valid_u_act_mask):
            logging.warning(
                'Actor at index {} can only be activated if it is in ready position.'.format(
                    np.nonzero(np.logical_not(valid_u_act_mask))[0]))

        return valid_u_act_mask

    def reset(self):
        """Resets the actors to the initial state."""
        self._t_act = self._init_t_act.copy()
        self._k = self._init_k
