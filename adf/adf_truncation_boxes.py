"""Batched ADF truncation boxes: Gauß–Taylor maps + actuator–phase rectangles ``S``.

``build_adf_truncation_boxes`` returns actuator-level maps and phase-level
rectangles (no ``Nb = Na·3`` expansion, no particle broadcast of geometry).

For the ADF forward chain, phase / actor geometry can be precomputed over the
whole horizon (independent of particle motion); only the Gauß–Taylor maps
remain inside the time loop.
"""

from __future__ import annotations

from abc import ABC
from typing import Tuple

import numpy as np

from adf.adf_gauss_taylor import gauss_taylor_linear_map
from contact_model import AbstractContactModel

# Phase order matches contact_model temporal stacking: HIT, UP, DOWN
_PHASE_NAMES = ("hit", "up", "down")


class ADFTruncationBoxesMixin(AbstractContactModel, ABC):
    """Mixin implementing batched ADF box construction on a contact model.

    Required on ``self``:

    * ``_actor_model`` – ``pos``, ``width``, ``output_t_{hit,up,down}_interval``,
      ``predict_actor_state``, ``t_lead``
    * ``_hit_success_probs`` – shape ``(4, 2)`` or ``(4,)`` as in ``AbstractContactModel``
      (rows: else, HIT, UP, DOWN)
    """

    def predict_ejection(self, motion_state_mean, motion_state_cov, particle_class, *args, **kwargs):
        """
        Not implemented on the ADF truncation-boxes contact model.

        The ADF forward chain computes ejection probabilities via Gauß–Taylor maps and
        truncation boxes directly, so the classic per-particle ``predict_ejection`` is not
        available here.

        :param motion_state_mean: Unused; accepted for API compatibility.
        :param motion_state_cov: Unused; accepted for API compatibility.
        :param particle_class: Unused; accepted for API compatibility.
        :param args: Unused positional extras.
        :param kwargs: Unused keyword extras (e.g. ``S_w``, ``t_act_k_minus_one_at_T``).

        :raises NotImplementedError: Always.
        """
        raise NotImplementedError(
            "ADFTruncationBoxesMixin.predict_ejection is not implemented; "
            "the ADF forward chain computes ejection probabilities directly."
        )

    def actuator_phase_time_intervals(
        self,
        t_act_k_minus_one_at_T: np.ndarray,
        u_act: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Physical begin/end times of HIT / UP / DOWN within the current stage.

        :param t_act_k_minus_one_at_T: A np.array of shape ``[batch_size, num_actors]``,
            the actors' internal times at the end of the previous stage.
        :param u_act: A np.array of shape ``[batch_size, num_actors]``, the hitting-time
            controls (including ``t_lead``).

        :returns: A tuple ``(t_hit, t_up, t_down)``, each a np.array of shape
            ``[batch_size, num_actors, 2]``, where ``[..., 0]`` is the phase begin and
            ``[..., 1]`` is the phase end (``np.inf`` if the phase does not occur).
        """
        t_act = np.asarray(t_act_k_minus_one_at_T, dtype=float)
        u_act = np.asarray(u_act, dtype=float)
        batch_size, num_actors = t_act.shape
        flat_t = t_act.reshape(-1)
        flat_u = u_act.reshape(-1)

        # Prefer fused HIT/UP/DOWN (Numba) when available on Variant B.
        if hasattr(self._actor_model, "output_phase_time_intervals"):
            t_hit, t_up, t_down = self._actor_model.output_phase_time_intervals(
                flat_t, flat_u
            )
        else:
            t_hit = self._actor_model.output_t_hit_interval(flat_t, flat_u)
            t_up = self._actor_model.output_t_up_interval(flat_t, flat_u)
            t_down = self._actor_model.output_t_down_interval(flat_t, flat_u)

        return (
            t_hit.reshape(batch_size, num_actors, 2),
            t_up.reshape(batch_size, num_actors, 2),
            t_down.reshape(batch_size, num_actors, 2),
        )

    def _actor_lateral_windows(self, num_actors: int):
        """
        Returns streamwise actor positions and lateral acceptance windows.

        :param num_actors: An integer, the number of actors ``Na``.

        :returns: A tuple ``(actor_x, y_lo, y_hi)``, each a np.array of shape ``[Na,]``.
            For 1D setups, ``y_lo`` / ``y_hi`` are ``±inf``.
        """
        actor_pos = np.asarray(self._actor_model.pos, dtype=float)
        if actor_pos.ndim == 1:
            actor_x = actor_pos
            y_lo = np.full(num_actors, -np.inf)
            y_hi = np.full(num_actors, np.inf)
        else:
            actor_x = actor_pos[:, 0]
            actor_y = actor_pos[:, 1]
            width = np.asarray(self._actor_model.width, dtype=float)
            if np.ndim(width) == 0:
                width = np.full(num_actors, float(width))
            y_lo = actor_y - 0.5 * width
            y_hi = actor_y + 0.5 * width
        return actor_x, y_lo, y_hi

    def build_adf_phase_geometry(
        self,
        particle_class: np.ndarray,
        t_act_k_minus_one_at_T: np.ndarray,
        u_act: np.ndarray,
        num_particles: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds phase rectangles and sparse ejection factors (no particle motion / no GT).

        :param particle_class: An integer np.array of shape ``[num_particles]``, with
            ``0`` for accept and ``1`` for reject.
        :param t_act_k_minus_one_at_T: A np.array of shape ``[batch_size, num_actors]``,
            the actors' internal times at the end of the previous stage.
        :param u_act: A np.array of shape ``[batch_size, num_actors]``, the hitting-time
            controls (including ``t_lead``).
        :param num_particles: None or an integer, the number of particles. If None, taken
            from ``particle_class.shape[0]``.

        :returns: A tuple
            ``(lower, upper, actuator_active, actor_x, phase_active, p_phase)`` where
            ``lower`` / ``upper`` are np.arrays of shape ``[batch_size, Na, 3, 2]``
            (HIT/UP/DOWN rectangles), ``actuator_active`` is a Boolean np.array of shape
            ``[batch_size, Na]``, ``actor_x`` is a np.array of shape ``[Na,]``,
            ``phase_active`` is a Boolean np.array of shape ``[batch_size, Na, 3]``, and
            ``p_phase`` is a np.array of shape ``[num_particles, 3]`` with per-particle
            HIT/UP/DOWN ejection probabilities.
        """
        particle_class = np.asarray(particle_class, dtype=int)
        t_act = np.asarray(t_act_k_minus_one_at_T, dtype=float)
        u_act = np.asarray(u_act, dtype=float)
        batch_size, num_actors = t_act.shape
        if num_particles is None:
            num_particles = int(particle_class.shape[0])
        n_phases = 3

        t_hit, t_up, t_down = self.actuator_phase_time_intervals(t_act, u_act)
        t_intervals = np.stack([t_hit, t_up, t_down], axis=2)
        phase_active = (
            np.isfinite(t_intervals[..., 0])
            & np.isfinite(t_intervals[..., 1])
            & (t_intervals[..., 1] > t_intervals[..., 0])
        )
        # No scheduled activation → treat all phases inactive (was p_eject=0).
        no_act = (t_act == 0.0) & ~np.isfinite(u_act)  # (B, Na)
        phase_active = phase_active & ~no_act[:, :, None]
        actuator_active = np.any(phase_active, axis=2)

        actor_x, y_lo, y_hi = self._actor_lateral_windows(num_actors)
        y_lo_b = np.broadcast_to(
            y_lo.reshape(1, num_actors, 1),
            (batch_size, num_actors, n_phases),
        )
        y_hi_b = np.broadcast_to(
            y_hi.reshape(1, num_actors, 1),
            (batch_size, num_actors, n_phases),
        )
        lower = np.stack([t_intervals[..., 0], y_lo_b], axis=-1)
        upper = np.stack([t_intervals[..., 1], y_hi_b], axis=-1)

        probs = np.asarray(self._hit_success_probs, dtype=float)
        if probs.ndim == 1:
            probs = np.stack([probs, probs], axis=-1)
        phase_probs = probs[1:4]
        p_phase = phase_probs[:, particle_class].T.astype(float, copy=False)  # (N, 3)
        if p_phase.shape[0] != num_particles:
            raise ValueError(
                f"particle_class length {p_phase.shape[0]} != num_particles {num_particles}"
            )

        return lower, upper, actuator_active, actor_x, phase_active, p_phase

    def build_adf_gt_maps(
        self,
        motion_state_mean: np.ndarray,
        S_w,
        actuator_active: np.ndarray,
        actor_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds Gauß–Taylor maps ``(H, d, R)`` for active ``(batch, actuator)`` pairs.

        :param motion_state_mean: A np.array of shape
            ``[batch_size, num_particles, state_dim]``, the living-mode motion means.
        :param S_w: A float or a length-2 sequence of floats, the process-noise PSD.
        :param actuator_active: A Boolean np.array of shape ``[batch_size, Na]``, True
            where at least one phase is active.
        :param actor_x: A np.array of shape ``[Na,]``, the streamwise actor positions.

        :returns: A tuple ``(H, d, R)`` of np.arrays of shapes
            ``[B, N, Na, 2, state_dim]``, ``[B, N, Na, 2]``, and ``[B, N, Na, 2, 2]``.
            Inactive actuators keep zeros.
        """
        mu = np.asarray(motion_state_mean, dtype=float)
        actuator_active = np.asarray(actuator_active, dtype=bool)
        actor_x = np.asarray(actor_x, dtype=float)
        batch_size, num_particles, state_dim = mu.shape
        num_actors = actuator_active.shape[1]

        H_act = np.zeros((batch_size, num_particles, num_actors, 2, state_dim))
        d_act = np.zeros((batch_size, num_particles, num_actors, 2))
        R_act = np.zeros((batch_size, num_particles, num_actors, 2, 2))
        if np.any(actuator_active):
            b_idx, na_idx = np.nonzero(actuator_active)
            mu_sub = mu[b_idx]
            x_sub = np.broadcast_to(
                actor_x[na_idx].reshape(-1, 1),
                (b_idx.shape[0], num_particles),
            )
            H_sub, d_sub, R_sub = gauss_taylor_linear_map(
                mu_sub,
                x_sub,
                S_w=S_w,
                t_L=0.0,
            )
            H_act[b_idx, :, na_idx] = H_sub
            d_act[b_idx, :, na_idx] = d_sub
            R_act[b_idx, :, na_idx] = R_sub
        return H_act, d_act, R_act

    def precompute_horizon_phase_geometry(
        self,
        t_act_k_at_T: np.ndarray,
        u_act_with_lead: np.ndarray,
        particle_class: np.ndarray,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Precomputes actor times and phase geometry for the whole OLF horizon.

        Actor dynamics and phase rectangles depend only on ``(t_act, u)``, not on
        particle motion — so this can run once before the ADF time loop.

        :param t_act_k_at_T: A np.array of shape ``[num_actors]`` or
            ``[batch_size, num_actors]``, the actor times at the start of stage 0. A 1-D
            vector is shared across the batch (typical for PTCR).
        :param u_act_with_lead: A np.array of shape
            ``[batch_size, num_actors, num_time_steps]``, the controls including ``t_lead``.
        :param particle_class: An integer np.array of shape ``[num_particles]``.

        :returns: A tuple
            ``(t_act_seq, lower, upper, phase_active, p_phase, actuator_active, actor_x)``
            where ``t_act_seq`` has shape ``[num_time_steps + 1, B, Na]``
            (``t_act_seq[n]`` is used at stage ``n``), ``lower`` / ``upper`` have shape
            ``[num_time_steps, B, Na, 3, 2]``, ``phase_active`` has shape
            ``[num_time_steps, B, Na, 3]``, ``p_phase`` has shape ``[N, 3]``,
            ``actuator_active`` has shape ``[num_time_steps, B, Na]``, and ``actor_x``
            has shape ``[Na,]``.
        """
        u = np.asarray(u_act_with_lead, dtype=float)
        t_act = np.asarray(t_act_k_at_T, dtype=float)
        particle_class = np.asarray(particle_class, dtype=int)
        batch_size, num_actors, num_time_steps = u.shape
        num_particles = int(particle_class.shape[0])

        if t_act.ndim == 1:
            t_act_shared = t_act
        else:
            if t_act.shape != (batch_size, num_actors):
                raise ValueError(
                    f"t_act_k_at_T shape {t_act.shape} incompatible with u {u.shape}"
                )
            t_act_shared = t_act[0]

        actor_x, y_lo, y_hi = self._actor_lateral_windows(num_actors)

        probs = np.asarray(self._hit_success_probs, dtype=float)
        if probs.ndim == 1:
            probs = np.stack([probs, probs], axis=-1)
        phase_probs = probs[1:4]
        p_phase = phase_probs[:, particle_class].T.astype(float, copy=False)  # (N, 3)
        if p_phase.shape[0] != num_particles:
            raise ValueError(
                f"particle_class length {p_phase.shape[0]} != num_particles {num_particles}"
            )

        try:
            from adf.adf_numba_kernels import (
                numba_enabled,
                precompute_horizon_phase_geometry_numba,
            )
        except ImportError:
            numba_enabled = lambda: False  # noqa: E731
            precompute_horizon_phase_geometry_numba = None

        am = self._actor_model
        if (
            numba_enabled()
            and precompute_horizon_phase_geometry_numba is not None
            and hasattr(am, "_t_activate")
            and hasattr(am, "_T")
        ):
            t_act_seq, lower, upper, phase_active, actuator_active = (
                precompute_horizon_phase_geometry_numba(
                    np.ascontiguousarray(t_act_shared, dtype=np.float64),
                    np.ascontiguousarray(u, dtype=np.float64),
                    np.ascontiguousarray(y_lo, dtype=np.float64),
                    np.ascontiguousarray(y_hi, dtype=np.float64),
                    float(am._T),
                    float(am.t_cycle),
                    float(am._t_activate),
                    float(am._t_up),
                    float(am._t_hit),
                    float(am._t_down),
                )
            )
            return (
                t_act_seq,
                lower,
                upper,
                phase_active,
                p_phase,
                actuator_active,
                actor_x,
            )

        # --- NumPy fallback (stage loop) ----------------------------------
        if t_act.ndim == 1:
            t_act = np.broadcast_to(t_act, (batch_size, num_actors)).copy()

        t_act_seq = np.empty((num_time_steps + 1, batch_size, num_actors), dtype=float)
        t_act_seq[0] = t_act

        lower = np.empty((num_time_steps, batch_size, num_actors, 3, 2), dtype=float)
        upper = np.empty((num_time_steps, batch_size, num_actors, 3, 2), dtype=float)
        phase_active = np.empty(
            (num_time_steps, batch_size, num_actors, 3), dtype=bool
        )
        actuator_active = np.empty(
            (num_time_steps, batch_size, num_actors), dtype=bool
        )

        for n in range(num_time_steps):
            lo, up, act, actor_x, ph_act, p_phase = self.build_adf_phase_geometry(
                particle_class=particle_class,
                t_act_k_minus_one_at_T=t_act_seq[n],
                u_act=u[:, :, n],
                num_particles=num_particles,
            )
            lower[n] = lo
            upper[n] = up
            phase_active[n] = ph_act
            actuator_active[n] = act
            t_act_seq[n + 1] = self._actor_model.predict_actor_state(
                t_act_k_minus_one_at_T=t_act_seq[n].reshape(-1),
                u_act=u[:, :, n].reshape(-1),
            ).reshape(batch_size, num_actors)

        return t_act_seq, lower, upper, phase_active, p_phase, actuator_active, actor_x

    def build_adf_truncation_boxes(
        self,
        motion_state_mean: np.ndarray,
        motion_state_cov: np.ndarray,
        particle_class: np.ndarray,
        S_w,
        t_act_k_minus_one_at_T: np.ndarray,
        u_act: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds soft-threshold boxes without full ``(B, N, Nb)`` materialization.

        :param motion_state_mean: A np.array of shape
            ``[batch_size, num_particles, state_dim]``, the living-mode means ``μ^{(1)}``.
        :param motion_state_cov: A np.array of shape
            ``[batch_size, num_particles, state_dim, state_dim]``. Unused for
            ``(H, d, R)`` (enters later via ``H P Hᵀ``); kept for API symmetry.
        :param particle_class: An integer np.array of shape ``[num_particles]``, with
            ``0`` for accept and ``1`` for reject.
        :param S_w: A float or a length-2 sequence of floats, the process-noise PSD.
        :param t_act_k_minus_one_at_T: A np.array of shape ``[batch_size, num_actors]``,
            the actors' internal times at the end of the previous stage.
        :param u_act: A np.array of shape ``[batch_size, num_actors]``, the hitting-time
            controls (including ``t_lead``).

        :returns: A tuple ``(H, d, lower, upper, p_eject, R)`` where ``H``, ``d``, ``R``
            are the Gauß–Taylor maps, ``lower`` / ``upper`` are phase rectangles of shape
            ``[B, Na, 3, 2]``, and ``p_eject`` is a dense np.array of shape
            ``[B, N, Na, 3]`` (legacy layout; the hot path prefers sparse factors).
        """
        del motion_state_cov  # see docstring
        mu = np.asarray(motion_state_mean, dtype=float)
        batch_size, num_particles, _ = mu.shape
        lower, upper, actuator_active, actor_x, phase_active, p_phase = (
            self.build_adf_phase_geometry(
                particle_class=particle_class,
                t_act_k_minus_one_at_T=t_act_k_minus_one_at_T,
                u_act=u_act,
                num_particles=num_particles,
            )
        )
        # Dense p_eject for legacy matching / tests (hot path uses sparse factors).
        num_actors = actuator_active.shape[1]
        p_eject = np.broadcast_to(
            p_phase.reshape(1, num_particles, 1, 3),
            (batch_size, num_particles, num_actors, 3),
        ).astype(float, copy=True)
        p_eject = np.where(phase_active[:, None, :, :], p_eject, 0.0)
        H, d, R = self.build_adf_gt_maps(
            motion_state_mean=mu,
            S_w=S_w,
            actuator_active=actuator_active,
            actor_x=actor_x,
        )
        return H, d, lower, upper, p_eject, R
