"""Combined ADF forward prediction and OLF cost evaluation.

Two cost variants (paper Sec. ``trajectory_evaluation``):

* **Terminal** (``use_only_terminal_costs=True``): ``g_{k,n}=0`` for ``n<N``,

      J = Σ_i c_acc 1(acc) π̃_N^{(0),i}  +  Σ_i c_rej 1(rej) π̃_N^{(1),i}.

* **Step** (``use_only_terminal_costs=False``): ``g_{k,N}=0``, stage costs

      E[g_n] = Σ_i (c_acc 1(acc) − c_rej 1(rej)) π̃_n^{(1),i} P(ej_n^i=1),

  with ``π̃^{(1)} P(ej)`` equal to the ADF ejected mass ``M_0^{ej}`` per stage.

Hot path (``n∈{2,3,4,6}``, Numba on): one fused kernel over the whole horizon.
Actor/phase geometry is precomputed once in a sequential Numba pass.
Shared initial ``t_act`` / ``π`` avoid per-sequence ``np.repeat`` copies.

On construction, ``ADFOLFCostMixin`` calls ``warmup_adf_numba`` whenever the ADF
Numba path is active (``numba_enabled()``).

Monte-Carlo validation (fine-``Δt``, no Gauß–Taylor) lives in ``adf_mc_forward``
and is exposed via ``_forward_chain_mc``.  Per-stage ADF dual-mode states for
diagnostics are available from ``_forward_chain_states`` (NumPy path only).
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from absl import logging

from adf.adf_olf_core import (
    forward_chain_adf_numpy,
    parse_forward_inputs,
    stage_olf_cost,
    terminal_olf_cost,
)
from adf.adf_truncation_boxes import ADFTruncationBoxesMixin
from controller_variant_b import AbstractDynamicProgrammingControllerVariantB


def _as_f64_c(a: np.ndarray) -> np.ndarray:
    """Returns a float64 C-contiguous array, copying only when necessary.

    :param a: A np.array of any dtype / order.

    :returns: A C-contiguous ``float64`` view or copy of ``a``.
    """
    a = np.asarray(a)
    if a.dtype == np.float64 and a.flags.c_contiguous:
        return a
    return np.ascontiguousarray(a, dtype=np.float64)


class ADFOLFCostMixin(AbstractDynamicProgrammingControllerVariantB, ABC):
    """Mixin: ADF open-loop prediction + OLF sorting cost in one pass.

    Required on ``self``:

    * ``_particle_model`` – ``predict_motion``, ``particle_class``, ``S_w``, ``_A``, ``_C_w``
    * ``_actor_model`` – ``predict_actor_state``, ``t_lead``
    * ``_contact_model`` – phase / GT helpers
    * ``_accept_cost_weight``, ``_reject_cost_weight`` – ``c_acc``, ``c_rej``
    * ``_use_only_terminal_costs`` – terminal vs step cost variant
    """

    def __init__(self, *args, contact_model_dict=None, seed=None, **kwargs):
        """Initializes the ADF OLF cost mixin and warms Numba kernels if available.

        :param args: Positional arguments forwarded to the parent controller.
        :param contact_model_dict: None or a dict of kwargs for
            :class:`ADFTruncationBoxesMixin` / the contact model.
        :param seed: None or an integer, RNG seed (also offsets the contact-model seed).
        :param kwargs: Keyword arguments forwarded to the parent controller.
        """
        contact_model_dict_copy = contact_model_dict.copy() if contact_model_dict is not None else {}

        super().__init__(*args, contact_model_dict=contact_model_dict, **kwargs)

        # Set the concrete ADF contact model (Gauß–Taylor truncation boxes).
        self._contact_model = ADFTruncationBoxesMixin(
            actor_model=self._actor_model,
            seed=seed + 654321 if seed is not None else None,
            **contact_model_dict_copy,
        )
        self._seed = seed

        from adf.adf_numba_kernels import numba_enabled, warmup_adf_numba

        if numba_enabled():
            logging.info(
                "Compiling ADF Numba kernels (state_dim=%d)…",
                int(self._particle_model.motion_state_length),
            )
            warmup_adf_numba(state_dim=int(self._particle_model.motion_state_length))

    def _parse_forward_inputs(
        self,
        u: np.ndarray,
        estimated_motion_state_mean: np.ndarray,
        estimated_motion_state_cov: np.ndarray,
        est_existence_probs: np.ndarray,
        t_act_k_at_T: np.ndarray,
    ):
        """Validates and normalizes ADF/MC forward-chain inputs.

        :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
            ``[batch_size, num_particles, 2]``.
        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``.

        :returns: The tuple returned by :func:`parse_forward_inputs`
            (normalized arrays and shape scalars).
        """
        return parse_forward_inputs(
            u,
            estimated_motion_state_mean,
            estimated_motion_state_cov,
            est_existence_probs,
            t_act_k_at_T,
        )

    def _forward_chain(
        self,
        u: np.ndarray,
        estimated_motion_state_mean: np.ndarray,
        estimated_motion_state_cov: np.ndarray,
        est_existence_probs: np.ndarray,
        t_act_k_at_T: np.ndarray,
        initial_time_step: int = 0,
    ) -> np.ndarray:
        """ADF-predicts existence over the horizon and returns OLF costs.

        Uses the Numba horizon kernel when available and supported, otherwise the
        NumPy staged path :func:`forward_chain_adf_numpy`.

        :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
            ``[batch_size, num_particles, 2]``.
        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``.
        :param initial_time_step: An integer, unused (kept for API compatibility).

        :returns: A np.array of shape ``[batch_size]``, the OLF costs.
        """
        del initial_time_step

        (
            u,
            mu0,
            P0,
            pi_shared,
            t_act_shared,
            _batch_size,
            _num_actors,
            num_time_steps,
            _num_particles,
            state_dim,
        ) = self._parse_forward_inputs(
            u,
            estimated_motion_state_mean,
            estimated_motion_state_cov,
            est_existence_probs,
            t_act_k_at_T,
        )

        particle_class = np.asarray(self._particle_model.particle_class, dtype=int)
        u_lead = np.add(u, self._actor_model.t_lead, dtype=float)
        use_terminal = bool(self._use_only_terminal_costs)

        (
            _t_act_seq,
            lower_seq,
            upper_seq,
            phase_active_seq,
            p_phase,
            actuator_active_seq,
            actor_x,
        ) = self._contact_model.precompute_horizon_phase_geometry(
            t_act_k_at_T=t_act_shared,
            u_act_with_lead=u_lead,
            particle_class=particle_class,
        )

        try:
            from adf.adf_numba_kernels import adf_olf_horizon_numba, numba_enabled
        except ImportError:
            numba_enabled = lambda: False  # noqa: E731
            adf_olf_horizon_numba = None

        if (
            numba_enabled()
            and adf_olf_horizon_numba is not None
            and state_dim in (2, 3, 4, 6)
            and num_time_steps >= 1
        ):
            S_w = np.asarray(self._particle_model.S_w, dtype=float)
            if state_dim in (2, 3):
                swx = float(S_w) if S_w.ndim == 0 else float(S_w.reshape(-1)[0])
                swy = swx
            else:
                swx = float(S_w.reshape(-1)[0])
                swy = float(S_w.reshape(-1)[1]) if S_w.size > 1 else swx

            boundaries = getattr(self._particle_model, "_boundaries", None)
            apply_walls = boundaries is not None and state_dim in (4, 6)
            y_ind = int(state_dim // 2) if apply_walls else 0
            y_lo = float(boundaries[0]) if apply_walls else 0.0
            y_hi = float(boundaries[1]) if apply_walls else 1.0

            is_acc = particle_class == 0
            is_rej = particle_class == 1
            return adf_olf_horizon_numba(
                _as_f64_c(pi_shared),
                _as_f64_c(mu0),
                _as_f64_c(P0),
                _as_f64_c(lower_seq),
                _as_f64_c(upper_seq),
                np.ascontiguousarray(phase_active_seq, dtype=np.bool_),
                _as_f64_c(p_phase),
                _as_f64_c(actor_x),
                _as_f64_c(self._particle_model._A),
                _as_f64_c(self._particle_model._C_w),
                swx,
                swy,
                bool(apply_walls),
                y_lo,
                y_hi,
                y_ind,
                np.ascontiguousarray(is_acc),
                np.ascontiguousarray(is_rej),
                float(self._accept_cost_weight),
                float(self._reject_cost_weight),
                use_terminal,
                1e-15,
            )

        cost, _states = forward_chain_adf_numpy(
            particle_model=self._particle_model,
            contact_model=self._contact_model,
            actor_model=self._actor_model,
            u=u,
            mu0=mu0,
            P0=P0,
            pi_shared=pi_shared,
            t_act_shared=t_act_shared,
            accept_cost_weight=float(self._accept_cost_weight),
            reject_cost_weight=float(self._reject_cost_weight),
            use_terminal=use_terminal,
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
        """ADF NumPy forward chain returning per-stage dual-mode ``(π, μ, P)``.

        Always uses the NumPy path (no Numba cost-only shortcut) so diagnostics /
        plots can inspect dual-mode moments after each stage.

        :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
            ``[batch_size, num_particles, 2]``.
        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``.
        :param initial_time_step: An integer, unused (kept for API compatibility).

        :returns: A tuple ``(cost, states)`` where ``cost`` has shape
            ``[batch_size]`` and ``states`` is a dict with ``pi_stages``,
            ``mu_stages``, ``P_stages`` (each a list of length ``N+1``).
        """
        del initial_time_step
        (
            u,
            mu0,
            P0,
            pi_shared,
            t_act_shared,
            _batch_size,
            _num_actors,
            _num_time_steps,
            _num_particles,
            _state_dim,
        ) = self._parse_forward_inputs(
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
            accept_cost_weight=float(self._accept_cost_weight),
            reject_cost_weight=float(self._reject_cost_weight),
            use_terminal=bool(self._use_only_terminal_costs),
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
        """Monte-Carlo forward prediction parallel to :meth:`_forward_chain`.

        :param u: A np.array of shape ``[batch_size, num_actors, num_time_steps]``.
        :param estimated_motion_state_mean: A np.array of shape
            ``[num_particles, state_dim]``.
        :param estimated_motion_state_cov: A np.array of shape
            ``[num_particles, state_dim, state_dim]``.
        :param est_existence_probs: A np.array of shape ``[num_particles, 2]`` or
            ``[batch_size, num_particles, 2]``.
        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``.
        :param num_samples: An integer, number of MC samples.
        :param num_micro_steps: An integer, fine-``Δt`` steps per stage (geometric).
        :param return_diagnostics: A Boolean, if ``True`` also return per-stage
            samples.
        :param rng: None or a ``numpy.random.Generator``.
        :param initial_time_step: An integer, unused (kept for API compatibility).
        :param contact_mode: A string, ``"geometric"`` (fine-``Δt``, no Gauß–Taylor)
            or ``"gt_box"`` (ADF soft-Bernoulli boxes on GT maps).

        :returns: A np.array of shape ``[batch_size]``, or
            ``(cost, diagnostics)`` when ``return_diagnostics`` is ``True``.
        """
        from adf_mc_validation.adf_mc_forward import forward_chain_mc

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

    def _stage_olf_cost(
        self,
        m_ej: np.ndarray,
        is_acc: np.ndarray,
        is_rej: np.ndarray,
    ) -> np.ndarray:
        """Expected stage OLF cost from ejection masses (signed step-cost form).

        :param m_ej: A np.array with a trailing particle axis, expected ejection
            mass per particle.
        :param is_acc: A Boolean np.array of shape ``[num_particles]``, accept mask.
        :param is_rej: A Boolean np.array of shape ``[num_particles]``, reject mask.

        :returns: A np.array of stage costs (batch axes of ``m_ej`` preserved).
        """
        return stage_olf_cost(
            m_ej,
            is_acc,
            is_rej,
            float(self._accept_cost_weight),
            float(self._reject_cost_weight),
        )

    def _terminal_olf_cost(
        self,
        pi_N: np.ndarray,
        particle_class: np.ndarray,
    ) -> np.ndarray:
        """Terminal OLF cost from dual-mode existence at the horizon end.

        :param pi_N: A np.array of shape ``[..., num_particles, 2]``, existence
            probabilities at the last stage.
        :param particle_class: An integer np.array of shape ``[num_particles]``.

        :returns: A np.array of terminal costs (leading batch axes of ``pi_N``
            preserved).
        """
        return terminal_olf_cost(
            pi_N,
            particle_class,
            float(self._accept_cost_weight),
            float(self._reject_cost_weight),
        )
