"""Minimal ADF controller fixture (no TensorFlow / simulator / controller stack).

Used by MC validation playground and smoke tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np



from snmpc_particle_model import ParticleModel
from actor_model import ActorModelVariantB
from adf.adf_truncation_boxes import ADFTruncationBoxesMixin
from adf.adf_olf_core import forward_chain_adf_numpy, parse_forward_inputs
from adf_mc_validation.adf_mc_forward import forward_chain_mc


class MinimalADFController:
    """Standalone ADF+MC cost object without PTCR / MTT / tree search.

    Uses a small 2D CV setup so Gauß–Taylor ``R`` for ``ξ=[τ,y]`` is non-singular
    (1D maps force a zero lateral variance and break the NumPy ADF path).
    """

    def __init__(
        self,
        *,
        T: float = 5.0,
        S_w: Tuple[float, float] | float = (1e-4, 1e-4),
        actor_pos: Optional[np.ndarray] = None,
        actor_timing: Optional[dict] = None,
        p_eject=None,
        accept_cost_weight: float = 0.5,
        reject_cost_weight: float = 0.5,
        use_only_terminal_costs: bool = True,
        seed: Optional[int] = 42,
    ):
        """Initializes a minimal 2D-CV ADF+MC controller for offline smoke tests.

        :param T: A float, stage / control time interval.
        :param S_w: A float or length-2 sequence, process-noise PSD for the CV model.
        :param actor_pos: None or a np.array of shape ``[num_actors, 2]``, actor
            positions; defaults to a small two-actuator layout.
        :param actor_timing: None or a dict of actuator timing kwargs
            (``t_activate``, ``t_up``, ``t_hit``, ``t_down``, ``t_reset``).
        :param p_eject: None or ejection probabilities for the contact model.
        :param accept_cost_weight: A float, OLF weight ``c_acc``.
        :param reject_cost_weight: A float, OLF weight ``c_rej``.
        :param use_only_terminal_costs: A Boolean, terminal vs step OLF costs.
        :param seed: None or an integer, RNG seed for particle / contact models.
        """
        actor_timing = actor_timing or {
            "t_activate": 1.0,
            "t_up": 1.0,
            "t_hit": 2.0,
            "t_down": 1.0,
            "t_reset": 4.0,
        }
        if actor_pos is None:
            actor_pos = np.array([[10.0, 0.5], [13.0, 0.5]], dtype=float)
        if p_eject is None:
            p_eject = [0.0, 0.95, 0.3, 0.2]
        if np.isscalar(S_w):
            S_w = (float(S_w), float(S_w))

        self._T = float(T)
        self._N = 4
        self._seed = seed
        self._accept_cost_weight = float(accept_cost_weight)
        self._reject_cost_weight = float(reject_cost_weight)
        self._use_only_terminal_costs = bool(use_only_terminal_costs)

        self._particle_model = ParticleModel(
            T=T,
            p_detect=0.95,
            S_w=S_w,
            S_v=(0.01, 0.01),
            motion_model_type="CV",
            seed=seed,
        )
        self._actor_model = ActorModelVariantB(
            pos=actor_pos,
            length=0.5,
            width=2.0,
            T=T,
            **actor_timing,
        )
        self._contact_model = ADFTruncationBoxesMixin(
            actor_model=self._actor_model,
            p_eject=p_eject,
            use_sum_approximation=False,
            seed=None if seed is None else seed + 654321,
        )

    @property
    def T(self) -> float:
        """The stage / control time interval.

        :returns: A float.
        """
        return self._T

    def _forward_chain(
        self,
        u: np.ndarray,
        estimated_motion_state_mean: np.ndarray,
        estimated_motion_state_cov: np.ndarray,
        est_existence_probs: np.ndarray,
        t_act_k_at_T: np.ndarray,
        initial_time_step: int = 0,
    ) -> np.ndarray:
        """
        ADF forward chain returning OLF costs (NumPy path).

        Signature matches :meth:`ADFOLFCostMixin._forward_chain`.

        :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
            ``[batch_size, num_particles, 2]``.
        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``.
        :param initial_time_step: An integer, unused (signature parity with the mixin).

        :returns: A np.array of shape ``[batch_size]``, the OLF costs per sequence.
        """
        del initial_time_step
        (
            u,
            mu0,
            P0,
            pi_shared,
            t_act_shared,
            *_rest,
        ) = parse_forward_inputs(
            u,
            estimated_motion_state_mean,
            estimated_motion_state_cov,
            est_existence_probs,
            t_act_k_at_T,
        )
        cost, _ = forward_chain_adf_numpy(
            particle_model=self._particle_model,
            contact_model=self._contact_model,
            actor_model=self._actor_model,
            u=u,
            mu0=mu0,
            P0=P0,
            pi_shared=pi_shared,
            t_act_shared=t_act_shared,
            accept_cost_weight=self._accept_cost_weight,
            reject_cost_weight=self._reject_cost_weight,
            use_terminal=self._use_only_terminal_costs,
            collect_states=False,
        )
        return cost

    def _forward_chain_states(
        self,
        u: np.ndarray,
        estimated_motion_state_mean: np.ndarray,
        estimated_motion_state_cov: np.ndarray,
        est_existence_probs: np.ndarray,
        t_act_k_at_T: np.ndarray,
        initial_time_step: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        ADF forward chain returning costs and per-stage dual-mode states.

        Signature matches :meth:`ADFOLFCostMixin._forward_chain_states`.

        :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
            ``[batch_size, num_particles, 2]``.
        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``.
        :param initial_time_step: An integer, unused (signature parity with the mixin).

        :returns: A tuple ``(cost, states)`` where ``cost`` is a np.array of shape
            ``[batch_size]`` and ``states`` is a dict of per-stage dual-mode ADF arrays.
        """
        del initial_time_step
        (
            u,
            mu0,
            P0,
            pi_shared,
            t_act_shared,
            *_rest,
        ) = parse_forward_inputs(
            u,
            estimated_motion_state_mean,
            estimated_motion_state_cov,
            est_existence_probs,
            t_act_k_at_T,
        )
        return forward_chain_adf_numpy(
            particle_model=self._particle_model,
            contact_model=self._contact_model,
            actor_model=self._actor_model,
            u=u,
            mu0=mu0,
            P0=P0,
            pi_shared=pi_shared,
            t_act_shared=t_act_shared,
            accept_cost_weight=self._accept_cost_weight,
            reject_cost_weight=self._reject_cost_weight,
            use_terminal=self._use_only_terminal_costs,
            collect_states=True,
        )

    def _forward_chain_mc(
        self,
        u: np.ndarray,
        estimated_motion_state_mean: np.ndarray,
        estimated_motion_state_cov: np.ndarray,
        est_existence_probs: np.ndarray,
        t_act_k_at_T: np.ndarray,
        num_samples: int = 2000,
        num_micro_steps: int = 100,
        return_diagnostics: bool = False,
        rng: Optional[np.random.Generator] = None,
        initial_time_step: int = 0,
        contact_mode: str = "geometric",
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Monte-Carlo forward chain; signature matches :meth:`ADFOLFCostMixin._forward_chain_mc`.

        :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
            ``[batch_size, num_particles, 2]``.
        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``.
        :param num_samples: An integer, the number of Monte-Carlo samples.
        :param num_micro_steps: An integer, fine-``Δt`` steps per stage (geometric mode).
        :param return_diagnostics: A Boolean, if True also return per-stage diagnostics.
        :param rng: None or a ``numpy.random.Generator``.
        :param initial_time_step: An integer, unused (signature parity).
        :param contact_mode: A string, ``"geometric"`` or ``"gt_box"``.

        :returns: A np.array of shape ``[batch_size]`` with OLF costs, or a tuple
            ``(cost, diagnostics)`` when ``return_diagnostics`` is True.
        """
        return forward_chain_mc(
            self,
            u=u,
            estimated_motion_state_mean=estimated_motion_state_mean,
            estimated_motion_state_cov=estimated_motion_state_cov,
            est_existence_probs=est_existence_probs,
            t_act_k_at_T=t_act_k_at_T,
            num_samples=num_samples,
            num_micro_steps=num_micro_steps,
            return_diagnostics=return_diagnostics,
            rng=rng,
            initial_time_step=initial_time_step,
            contact_mode=contact_mode,
        )


def build_2d_controller(
    T: float = 5.0,
    S_w: float = 1e-4,
    seed: int = 42,
    use_only_terminal_costs: bool = True,
) -> MinimalADFController:
    """Builds a minimal 2D-CV ADF controller for offline ADF/MC comparison.

    The historical name is kept for call-site compatibility; motion is 2D so the
    ADF measurement covariance ``R`` for ``ξ=[τ, y]`` is full rank.

    :param T: A float, stage / control time interval.
    :param S_w: A float, isotropic process-noise PSD applied to both axes.
    :param seed: An integer, RNG seed.
    :param use_only_terminal_costs: A Boolean, terminal vs step OLF costs.

    :returns: A :class:`MinimalADFController` instance.
    """
    return MinimalADFController(
        T=T,
        S_w=(float(S_w), float(S_w)),
        seed=seed,
        use_only_terminal_costs=use_only_terminal_costs,
    )


def default_scenario(cov_scale: float = 1e-3):
    """Builds a two-particle scenario approaching the first actuator within one stage.

    State is 2D CV ``[x, v_x, y, v_y]``. Control ``u`` is the activation time
    (without ``t_lead``); with ``t_lead=3`` and ``u=0.5`` the HIT centre is at 3.5.

    :param cov_scale: A float, scalar multiplier for the diagonal initial
        covariances.

    :returns: A tuple ``(mu0, P0, pi0, particle_class, t_act, u)`` suitable as
        inputs to the minimal controller forward chains.
    """
    mu0 = np.array(
        [
            [6.0, 1.0, 0.5, 0.0],
            [7.0, 1.0, 0.5, 0.0],
        ],
        dtype=float,
    )
    P0 = np.stack(
        [cov_scale * np.diag([1.0, 0.1, 1.0, 0.1])] * 2
    )
    pi0 = np.array([[0.05, 0.95], [0.05, 0.95]], dtype=float)
    particle_class = np.array([1, 0], dtype=int)
    t_act = np.zeros(2, dtype=float)
    u = np.full((1, 2, 1), np.nan, dtype=float)
    # activation time → HIT centre = u + t_lead = 0.5 + 3 = 3.5
    u[0, 0, 0] = 0.5
    return mu0, P0, pi0, particle_class, t_act, u
