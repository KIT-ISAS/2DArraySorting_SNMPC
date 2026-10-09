import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np


class DummyMTTTracker:
    """A dummy MTT tracker that can be used if the controller is only accessed via controller.control_from_states(...)
    and does not require tracking from measurements.

    """

    def track_particles(self, *args, **kwargs):
        """Not implemented; use ``control_from_states`` instead of measurement-driven control.

        :param args: Ignored positional arguments.
        :param kwargs: Ignored keyword arguments.

        :raises NotImplementedError: Always, because this tracker has no measurement
            update. Controllers should call ``control_from_states(...)`` instead of
            ``call(...)`` / ``control_from_measurements(...)``.
        """
        raise NotImplementedError('Object of type DummyMTTTracker does not implement method track_particles. '
                                  'Controllers using this tracker should call controller.control_from_states(...) '
                                  'instead of the controller.call(...)/controller.from_measurements(...) methods.')


def physical_noise_to_tracksort_s_w_s_v(S_w,
                                        S_v,
                                        x_normalization_constant,
                                        y_normalization_constant,
                                        time_normalization_constant,
                                        measurement_interval):
    """Converts physical process and measurement noise to TrackSort/KF normalized scalars s_w and s_v.

    Assumes a CV model for both simulator and TrackSort KF for convertion.

    TrackSort normalizes positions by the FOV extents and time by time_normalization_constant (in measurement
    steps). The KF uses dt = 1 / time_normalization_constant per measurement step. With
    T_scale = time_normalization_constant * measurement_interval (physical time corresponding to normalized
    time 1), the conversions are

        s_w_i = S_w_i * T_scale**3 / L_i**2 ,
        s_v_i = S_v_i / L_i**2 ,

    and the returned scalars are the averages over the spatial axes (the KF model options are isotropic).

    :param S_w: A float or a sequence of two floats, the process-noise power spectral density in physical units
        (length**2 / time**3).
    :param S_v: A float or a sequence of two floats, the measurement-noise variance in physical units (length**2).
    :param x_normalization_constant: A float, the x FOV extent used for position normalization.
    :param y_normalization_constant: A float, the y FOV extent used for position normalization.
    :param time_normalization_constant: A float, the time normalization constant (measurement steps for a typical
        particle to cross the FOV).
    :param measurement_interval: A float, the physical time between consecutive MTT measurement updates.

    :returns: A tuple (s_w, s_v) of floats for TrackSort KF model_options.
    """
    S_w = np.atleast_1d(np.asarray(S_w, dtype=float))
    S_v = np.atleast_1d(np.asarray(S_v, dtype=float))
    if S_w.size == 1:
        S_w = np.repeat(S_w, 2)
    if S_v.size == 1:
        S_v = np.repeat(S_v, 2)
    if S_w.size != 2 or S_v.size != 2:
        raise ValueError('S_w and S_v must each be a scalar or a sequence of length 2.')

    lengths = np.array([x_normalization_constant, y_normalization_constant], dtype=float)
    if np.any(lengths <= 0):
        raise ValueError('Normalization constants must be positive.')
    if time_normalization_constant <= 0 or measurement_interval <= 0:
        raise ValueError('time_normalization_constant and measurement_interval must be positive.')

    t_scale = time_normalization_constant * measurement_interval
    s_w = float(np.mean(S_w * t_scale ** 3 / lengths ** 2))
    s_v = float(np.mean(S_v / lengths ** 2))
    return s_w, s_v


def _override_kf_noise_in_model_config(model_config, s_w=None, s_v=None):
    """Overrides ``s_w`` and/or ``s_v`` in all KF predictor ``model_options`` of a TrackSort config.

    :param model_config: A dict, TrackSort model configuration (mutated in place and
        returned).
    :param s_w: None or a float, process-noise PSD override.
    :param s_v: None or a float, measurement-noise variance override.

    :returns: The (possibly modified) ``model_config`` dict.
    """
    model_config = copy.deepcopy(model_config)
    predictors = model_config.get('predictors', {})
    for pred_cfg in predictors.values():
        if pred_cfg.get('type') != 'KF':
            continue
        model_options = pred_cfg.setdefault('model_options', {})
        if s_w is not None:
            model_options['s_w'] = s_w
        if s_v is not None:
            model_options['s_v'] = s_v
    return model_config


class TrackSortMTTTracker:
    """Callable multitarget tracker wrapping Adaptive_Filtering's functional TrackSort MTT model.

    Signature of __call__:

        measurements -> (estimated_motion_state_mean, estimated_motion_state_cov, particle_class, particle_id,
            track_score)

    The underlying FunctionalLiveMultiTargetTrackingModel returns a ``states`` tuple
    (means, covs, most_freq_label, track_ids, track_scores, is_artificial, info); this wrapper
    converts that into the controller-facing return above and stores ``info`` on
    :attr:`track_info` (keys ``n_associated_meas``, ``last_meas_idx``), aligned with the
    returned tracks after filtering.

    Measurements are expected in physical coordinates as [x, y, label] (label_idx=2 in the IO config).
    """

    def __init__(self,
                 array_end,
                 area_width,
                 avg_num_time_steps_per_particle,
                 measurement_interval,
                 feed_tracks_with_position_greater=None,
                 feed_track_scores_to_controller=False,
                 model_config_path=None,
                 io_config_path=None,
                 s_w=None,
                 s_v=None,
                 expected_max_num_tracks=12000,
                 expected_max_total_num_timesteps=6400,
                 expected_max_diff_track_ids=128,
                 expected_max_length_tracks=60,
                 verbosity_level='INFO'):
        """Initializes the TrackSort MTT tracker.

        IO geometry (FOV, boundaries, time normalization) is taken from io_config_path and overridden in memory with
        array_end / area_width / avg_num_time_steps_per_particle. Template config files on disk are not modified.
        Optional s_w / s_v override all KF predictor model_options in the model config (also in memory only).

        :param array_end: A float, the end of the observable x-range (camera FOV width / upper x bound).
        :param area_width: A float, the observable y-range (camera FOV height / upper boundary).
        :param avg_num_time_steps_per_particle: A float, time normalization constant (typical number of measurement
            steps a particle spends in the FOV).
        :param measurement_interval: A float, the physical time between consecutive MTT measurement updates.
        :param feed_tracks_with_position_greater: None or a float, if given, only tracks with x-position greater than
            this value are fed to the controller.
        :param feed_track_scores_to_controller: A Bool, if True, the track scores are fed to the controller and may be
            used as initial existence probabilities for the particles. If False, only the particles that are not
            artificial are fed to the controller (with initial existence probability 1.0).
        :param model_config_path: A string, path to the TrackSort model config JSON.
        :param io_config_path: A string, path to the TrackSort IO config JSON template.
        :param s_w: None or a float, if given, overrides KF process-noise PSD in the model config.
        :param s_v: None or a float, if given, overrides KF measurement-noise variance in the model config.
        :param expected_max_num_tracks: An integer, buffer size passed to build_adaptive_track_sort_model.
        :param expected_max_total_num_timesteps: An integer, buffer size passed to build_adaptive_track_sort_model.
        :param expected_max_diff_track_ids: An integer, buffer size passed to build_adaptive_track_sort_model.
        :param expected_max_length_tracks: An integer, buffer size passed to build_adaptive_track_sort_model.
        :param verbosity_level: A string, logging verbosity for Adaptive_Filtering.
        """
        # Lazy import: Adaptive_Filtering pulls in TensorFlow.
        from Adaptive_Filtering import build_adaptive_track_sort_model

        config_dir = Path(__file__).resolve().parent / 'tracksort_configs'
        model_config_path = model_config_path or config_dir / 'tracksort_kf_mtt_live_config.json'
        io_config_path = io_config_path or config_dir / 'tracksort_io_config.json'

        with open(io_config_path) as f:
            io_config = json.load(f)

        io_config['data']['camera_fov_width'] = array_end
        io_config['data']['camera_fov_height'] = area_width
        io_config['data']['time_normalization_constant'] = avg_num_time_steps_per_particle
        io_config['data']['upper_boundary'] = area_width
        io_config['data']['lower_boundary'] = 0.0

        with open(model_config_path) as f:
            model_config = json.load(f)

        if s_w is not None or s_v is not None:
            model_config = _override_kf_noise_in_model_config(model_config, s_w=s_w, s_v=s_v)

        self._temp_dir = tempfile.TemporaryDirectory(prefix='tracksort_mtt_')
        runtime_io_path = os.path.join(self._temp_dir.name, 'io_config.json')
        runtime_model_path = os.path.join(self._temp_dir.name, 'model_config.json')
        with open(runtime_io_path, 'w') as f:
            json.dump(io_config, f, indent=2)
        with open(runtime_model_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        self._ts_model = build_adaptive_track_sort_model(
            model_config_path=runtime_model_path,
            io_config_path=runtime_io_path,
            expected_max_num_tracks=expected_max_num_tracks,
            expected_max_total_num_timesteps=expected_max_total_num_timesteps,
            expected_max_diff_track_ids=expected_max_diff_track_ids,
            expected_max_length_tracks=expected_max_length_tracks,
            write_track_history=False,
            save_results=False,
            log_every_n_time_steps=1,
            verbosity_level=verbosity_level)

        self._feed_tracks_with_position_greater = feed_tracks_with_position_greater
        self._array_end = array_end
        self._area_width = area_width
        self._measurement_interval = measurement_interval
        self._feed_track_scores_to_controller = feed_track_scores_to_controller

        self._time_step = 0
        self._track_info = {
            "n_associated_meas": np.empty((0,), dtype=int),
            "last_meas_idx": np.empty((0,), dtype=int),
        }

    @property
    def track_info(self):
        """Auxiliary per-track info from the last MTT step (aligned with returned tracks).

        :returns: A dict with keys ``n_associated_meas`` and ``last_meas_idx`` (integer
            np.arrays of shape ``[num_particles]``; empty when there are no tracks).
        """
        return self._track_info

    def __call__(self, measurements):
        """Runs one MTT step via the functional TrackSort model.

        Unpacks the MTT ``states`` tuple from
        ``FunctionalLiveMultiTargetTrackingModel.run_one_mtt_step`` (see ``run_one_step``
        there for the full component documentation), optionally drops artificial tracks,
        converts velocities/accelerations from per-frame to physical time using
        ``measurement_interval``, and optionally filters by x-position. Auxiliary fields
        from ``states[-1]`` (``info``) are stored on :attr:`track_info`, aligned with the
        returned tracks (after the same filters). Used e.g. by PTCR partial-hit selection
        via ``n_associated_meas``.

        :param measurements: A np.array of shape [num_measurements, 3] with format [x, y, label], or an empty array
            with shape [0, 3].

        :returns:
            estimated_motion_state_mean: A np.array of shape [num_particles, state_length], the estimated motion
                state mean in physical units (CV: [x, v_x, y, v_y]; CA: [x, v_x, a_x, y, v_y, a_y]).
            estimated_motion_state_cov: A np.array of shape [num_particles, state_length, state_length], the
                estimated motion state covariance in physical units.
            particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
                particle") or 1 ("eject the particle") representing the particle classes (from most frequent label).
            particle_id: An integer np.array of shape [num_particles], the ids of the particles.
            track_score: A float np.array of shape [num_particles], the track scores. Used as estimated
                    existence probability for each particle. If ``feed_track_scores_to_controller`` is False,
                    only non-artificial tracks are returned and scores are set to 1.0.
        """
        if measurements is None:
            measurements = np.empty((0, 3))
        elif measurements.size == 0:
            measurements = np.empty((0, 3))

        in_fov = (measurements[:, 0] >= 0) & (measurements[:, 0] <= self._array_end) & \
                 (measurements[:, 1] >= 0) & (measurements[:, 1] <= self._area_width)
        measurements = measurements[in_fov]

        _, states = self._ts_model(measurements, time_step=self._time_step)
        if states is not None:
            # states = (means, covs, most_freq_label, track_ids, track_scores, is_artificial, info)
            mean, cov, most_freq_label, track_id, track_score, is_artificial, info = states
            info = dict(info or {})
            n_associated_meas = np.asarray(
                info.get("n_associated_meas", np.empty((mean.shape[0],), dtype=int)),
                dtype=int,
            )
            last_meas_idx = np.asarray(
                info.get("last_meas_idx", np.empty((mean.shape[0],), dtype=int)),
                dtype=int,
            )

            if not self._feed_track_scores_to_controller:
                # Only keep non-artificial tracks; treat them as certain (existence probability 1.0)
                is_not_artificial = np.logical_not(is_artificial)
                mean = mean[is_not_artificial]
                cov = cov[is_not_artificial]
                most_freq_label = most_freq_label[is_not_artificial]
                track_id = track_id[is_not_artificial]
                track_score = np.ones(np.count_nonzero(is_not_artificial), dtype=float)
                n_associated_meas = n_associated_meas[is_not_artificial]
                last_meas_idx = last_meas_idx[is_not_artificial]

            # Adaptive_Filtering denormalizes with L / a and L / a**k only (a = time_normalization_constant).
            # Remaining time unit is therefore "per frame"; convert to physical time with Δt = measurement_interval.
            # CV: [x, v_x, y, v_y] — v in length/frame → length/time via / Δt
            # CA: [x, v_x, a_x, y, v_y, a_y] — v via / Δt, a via / Δt**2
            dt = self._measurement_interval
            state_dim = mean.shape[1]
            if state_dim == 4:
                mean[:, [1, 3]] /= dt
                cov[:, [1, 3], [1, 3]] /= dt ** 2
                cov[:, [0, 1], [1, 0]] /= dt
                cov[:, [2, 3], [3, 2]] /= dt
            elif state_dim == 6:
                mean[:, [1, 4]] /= dt
                mean[:, [2, 5]] /= dt ** 2
                cov[:, [1, 4], [1, 4]] /= dt ** 2
                cov[:, [2, 5], [2, 5]] /= dt ** 4
                cov[:, [0, 1], [1, 0]] /= dt
                cov[:, [3, 4], [4, 3]] /= dt
                cov[:, [0, 2], [2, 0]] /= dt ** 2
                cov[:, [3, 5], [5, 3]] /= dt ** 2
                cov[:, [1, 2], [2, 1]] /= dt ** 3
                cov[:, [4, 5], [5, 4]] /= dt ** 3
            else:
                raise ValueError(
                    'Unsupported TrackSort state dimension {} (expected 4 for CV or 6 for CA).'.format(state_dim))

            if self._feed_tracks_with_position_greater is not None:
                # Filter tracks based on x-position
                mask = mean[:, 0] > self._feed_tracks_with_position_greater
                mean = mean[mask]
                cov = cov[mask]
                most_freq_label = most_freq_label[mask]
                track_id = track_id[mask]
                track_score = track_score[mask]
                n_associated_meas = n_associated_meas[mask]
                last_meas_idx = last_meas_idx[mask]

        else:
            mean = np.empty((0, 4))
            cov = np.empty((0, 4, 4))
            most_freq_label = np.empty((0,), dtype=int)
            track_id = np.empty((0,), dtype=int)
            track_score = np.empty((0,))
            n_associated_meas = np.empty((0,), dtype=int)
            last_meas_idx = np.empty((0,), dtype=int)

        self._track_info = {
            "n_associated_meas": n_associated_meas,
            "last_meas_idx": last_meas_idx,
        }

        self._time_step += 1
        return mean, cov, most_freq_label.astype(int), track_id.astype(int), track_score
