"""
###################################### Multi_Actor_Scheduling ##########################################
Authors: Marcel Reith-Braun (ISAS): marcel.reith-braun@kit.edu
########################################################################################################
Add description here.
usage:
 - run docker container - tested with tensorflow/twodarraysorting_snmpc:2.16.1-gpu image:
    $ docker run -u $(id -u):$(id -g) \\
            -- gpus="device=0" \\
            -it --rm \\
            -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \\
            -v </path/to/repo>:/mnt \\
            tensorflow/twodarraysorting_snmpc:2.16.1-gpu
 - within container:
     $   python3 /mnt/main.py
requirements:
  - Required packages: See corresponding dockerfile.
  - Volume mounts: Specify a path </path/to/repo/> that points to the repo.
"""
import sys

from absl import app, flags, logging

import numpy as np
from scenario_helpers import calculate_x_offset, avg_num_time_steps_particle_in_range

import json
from datetime import datetime
from importlib.metadata import version

from file_utils import copy_file
from simulator import ActorSimulator, AreaParticleSimulator, AreaSortingSimulator
from heuristics import FirstActorFirstHeuristicController

from ptcr_controller import PTCRController
from ptcr_config import pop_ptcr_nested_config
from mtt_trackers import TrackSortMTTTracker, physical_noise_to_tracksort_s_w_s_v

FLAGS = flags.FLAGS
FLAG_NAMES_SET = set()
TF_VERSION = version('tensorflow')


def define_flags(flag_names_set):
    """Defines the possible arguments you can give to main function.

    :param flag_names_set: A set of strings containing the names of the flags defined so far (e.g., by another module).
        Keeps track of the flags defined so far to avoid be able to detect when flags would be defined twice. Important
        for the tests since we use the same FlagValues instance (flags.FLAGS) for all tests.
    """
    flags.DEFINE_enum('controller',
                      default='first_actor_first',
                      enum_values=['first_actor_first', 'ptcr'],
                      help='The controller to be used to compute actuator activations.')
    flags.DEFINE_string('controller_config_path',
                        default=None,
                        help="The path where the controller config file is stored.")
    flags.DEFINE_enum('grid_size',
                      default='full',
                      enum_values=['full', '2x2', '3x3', '4x4'],
                      help='The size of the actor grid.')
    flags.DEFINE_enum('configuration',
                      default='A',
                      enum_values=['A', 'B'],
                      help='The configuration for the sorter (A is a toy example, B is similar to ISAS-TableSort).')
    flags.DEFINE_float('simulation_noise',
                       default=0.0,
                       help='The noise level for the particle simulator, i.e., its system noise power spectral density.')
    flags.DEFINE_float('measurement_noise',
                       default=0.0,
                       help='Measurement noise variance for particle simulator measure_particles(...). For 2D, the '
                            'same variance is used in x- and y-direction. Units are length**2 (e.g. mm**2).')
    flags.DEFINE_bool('use_mtt_tracker',
                      default=False,
                      help='If True, use TrackSortMTTTracker as the controller MTT. If False, use the particle '
                           'simulator as an oracle tracker.')
    flags.DEFINE_bool('match_tracker_noise',
                      default=False,
                      help='If True and use_mtt_tracker is True, override TrackSort KF s_w/s_v so they match the '
                           'simulator process noise (simulation_noise) and measurement noise (measurement_noise). '
                           'Assumes a CV model for both simulator and TrackSort KF.')
    flags.DEFINE_bool('feed_track_scores_to_controller',
                      default=False,
                      help='If True and use_mtt_tracker is True, feed TrackSort track scores to the controller as '
                           'initial existence probabilities (including artificial tracks). If False, only '
                           'non-artificial tracks are passed with existence probability 1.0.')
    flags.DEFINE_bool('feed_only_tracks_close_to_array_to_controller',
                      default=False,
                      help='If True and use_mtt_tracker is True, only feed TrackSort tracks with x-position past '
                           'tracking_start (= first actor row minus one actuator cycle lead) to the controller. '
                           'MTT still tracks the full FOV (including the mtt_x_offset warmup). Also shortens the '
                           'default control horizon N to cover [tracking_start, array_end].')
    flags.DEFINE_bool('save_results',
                      default=False,
                      help='Whether to save the results.')
    flags.DEFINE_string('result_dir',
                        default='/mnt/results/',
                        help='The directory where to save the results.')
    flags.DEFINE_bool('no_show_animation',
                      default=False,
                      help='Whether to display an interactive animation of the simulation.')
    flags.DEFINE_bool('save_animation',
                      default=False,
                      help='Whether to save the animation of the simulation as video frames to directory.')
    flags.DEFINE_string('animation_dir',
                        default='/mnt/images/',
                        help='The directory where the animation frames are saved to.'
                        )
    flags.DEFINE_enum('verbosity_level',
                      default='INFO',
                      enum_values=['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
                      help='Verbosity options.')
    flags.DEFINE_integer(
        'particle_simulator_seed',
        default=12345,
        help='RNG seed for the particle simulator (reproducible scenarios).',
    )
    flags.DEFINE_integer(
        'total_num_particles',
        default=100,
        help='Total number of particles created by the area particle simulator over the run.',
    )
    flags.DEFINE_enum(
        'mass_flow',
        default='medium',
        enum_values=['medium', 'high'],
        help='Occupation density / mass flow: medium uses the grid-size default '
             'avg_particles_in_array; high doubles it (and thus the birth rate).',
    )
    flag_names_set.update([fl.name for fl in FLAGS.get_flags_for_module(sys.modules[__name__])])


def main(args):
    del args

    # Get the current time
    current_time = datetime.today().strftime("%Y%m%d_%H:%M:%S")

    # Setup logging
    logging.set_verbosity(FLAGS.verbosity_level)

    # Print the TensorFlow version
    logging.info(f'TensorFlow version: {TF_VERSION}')

    # Load the controller config if provided
    if FLAGS.controller_config_path is not None:
        # import controller config to json tree
        with open(FLAGS.controller_config_path) as f:
            controller_config = json.load(f)

        if FLAGS.save_results:
            # copy the controller config to the results directory
            copy_file(FLAGS.controller_config_path, FLAGS.result_dir)
        if FLAGS.save_animation:
            # copy the controller config to the animation directory
            copy_file(FLAGS.controller_config_path, FLAGS.animation_dir)

    else:
        controller_config = {}

    # Define the actor times and geometry (configuration)
    if FLAGS.configuration == "A":
        T = 0.5
        actor_dict = {'t_activate': 1,
                      't_up': 1,
                      't_hit': 2,
                      't_down': 1,
                      't_reset': 4}
        v0_x = 10.0
        frame_factor = 5  # every 5 time steps, we receive a new frame
    else:
        # we use some reasonable values similar to the ones of the real system (ISAS-TableSort)
        T = 2.5  # in ms
        actor_dict = {'t_activate': 10,  # in ms, actually ~10ms (see Master's thesis Leo Schön, pg. 89)
                      't_up': 0,  # actually ~0-1ms
                      't_hit': 2.5,  # actually ~3ms
                      't_down': 5,  # actually ~6ms
                      't_reset': 5}  # actually ~5ms
        v0_x = 1.0  # in mm/ms
        frame_factor = 8  # every 8 time steps, we receive a new frame, i.e., each frame is 20ms, or 50 fps
        # frame_factor = 4  # this is the true value of ISAS-TableSort (100 fps), but for full grid, results in a too
        # long control horizon

    # Extra x-offset when using TrackSort so that ~20 MTT measurements are available before particles reach the
    # actor array (measurement interval ≈ frame_factor * T).
    mtt_x_offset = 0.0
    if FLAGS.use_mtt_tracker:
        mtt_x_offset = float(np.ceil((20 * frame_factor * T * v0_x) / 5.0) * 5.0)

    if FLAGS.grid_size == "full":
        # full grid
        actor_positions = np.array(  # in mm, positions correspond to the positions of the ISAS-TableSort with the
            # origin of the coordinate system fixed in the lower left corner of the (physical) actuator array assembly
            [[25., 17.5], [25., 33.5], [25., 49.5], [25., 65.5], [25., 81.5], [25., 97.5], [25., 113.5],
             [45., 11.5], [45., 27.5], [45., 43.5], [45., 59.5], [45., 75.5], [45., 91.5], [45., 107.5],
             [65., 21.5], [65., 37.5], [65., 53.5], [65., 69.5], [65., 85.5], [65., 101.5], [65., 117.5],
             [85., 15.5], [85., 31.5], [85., 47.5], [85., 63.5], [85., 79.5], [85., 95.5], [85., 111.5],
             [105., 9.5], [105., 25.5], [105., 41.5], [105., 57.5], [105., 73.5], [105., 89.5], [105., 105.5],
             [125., 19.5], [125., 35.5], [125., 51.5], [125., 67.5], [125., 83.5], [125., 99.5], [125., 115.5],
             [145., 13.5], [145., 29.5], [145., 45.5], [145., 61.5], [145., 77.5], [145., 93.5], [145., 109.5],
             [165., 7.5], [165., 23.5], [165., 39.5], [165., 55.5], [165., 71.5], [165., 87.5], [165., 103.5]])
        array_start = np.min(actor_positions[:, 0])
        actor_cycle_x_offset = calculate_x_offset(actor_dict, v0_x, array_start)
        offset = actor_cycle_x_offset + mtt_x_offset
        actor_positions[:, 0] += offset
        array_end = 180 + offset
        array_start += offset
        tracking_start = array_start - actor_cycle_x_offset if FLAGS.use_mtt_tracker and FLAGS.feed_only_tracks_close_to_array_to_controller else 0.0
        area_width = 125
        avg_particles_in_array = 20  # average number of particles (visible & invisible, i.e., ejected) that should be
        # in the sorter, defines the occupation density/mass flow of the sorter

    elif FLAGS.grid_size == "2x2":
        # 2x2 actor grid
        actor_positions = np.array([[45., 13.5], [45., 29.5],  # in mm
                                    [65., 7.5], [65., 23.5]])
        array_start = np.min(actor_positions[:, 0])
        actor_cycle_x_offset = calculate_x_offset(actor_dict, v0_x, array_start)
        offset = actor_cycle_x_offset + mtt_x_offset
        actor_positions[:, 0] += offset
        array_end = 80 + offset
        array_start += offset
        tracking_start = array_start - actor_cycle_x_offset if FLAGS.use_mtt_tracker and FLAGS.feed_only_tracks_close_to_array_to_controller else 0.0
        area_width = 37
        avg_particles_in_array = 1  # average number of particles (visible & invisible, i.e., ejected) that should be
        # in the sorter, defines the occupation density/mass flow of the sorter

    elif FLAGS.grid_size == "3x3":
        # 3x3 actor grid
        actor_positions = np.array(  # in mm
            [[45., 19.5], [45., 35.5], [45., 51.5],
             [65., 13.5], [65., 29.5], [65., 45.5],
             [85., 7.5], [85., 23.5], [85., 39.5]])
        array_start = np.min(actor_positions[:, 0])
        actor_cycle_x_offset = calculate_x_offset(actor_dict, v0_x, array_start)
        offset = actor_cycle_x_offset + mtt_x_offset
        actor_positions[:, 0] += offset
        array_end = 100 + offset
        array_start += offset
        tracking_start = array_start - actor_cycle_x_offset if FLAGS.use_mtt_tracker and FLAGS.feed_only_tracks_close_to_array_to_controller else 0.0
        area_width = 59
        avg_particles_in_array = 5  # average number of particles (visible & invisible, i.e., ejected) that should be
        # in the sorter, defines the occupation density/mass flow of the sorter

    else:
        # 4x4 actor grid
        actor_positions = np.array(  # in mm, positions correspond to the positions of the ISAS-TableSort with the
            # origin of the coordinate system fixed in the lower left corner of the (physical) actuator array assembly
            [[25., 17.5], [25., 33.5], [25., 49.5], [25., 65.5],
             [45., 11.5], [45., 27.5], [45., 43.5], [45., 59.5],
             [65., 21.5], [65., 37.5], [65., 53.5], [65., 69.5],
             [85., 15.5], [85., 31.5], [85., 47.5], [85., 63.5]])
        array_start = np.min(actor_positions[:, 0])
        actor_cycle_x_offset = calculate_x_offset(actor_dict, v0_x, array_start)
        x_offset = actor_cycle_x_offset + mtt_x_offset
        y_offset = -4  # we need to shift the array down to avoid that particles miss the actors
        actor_positions[:, 0] += x_offset
        actor_positions[:, 1] += y_offset
        array_end = 100 + x_offset
        array_start += x_offset
        tracking_start = array_start - actor_cycle_x_offset if FLAGS.use_mtt_tracker and FLAGS.feed_only_tracks_close_to_array_to_controller else 0.0
        area_width = 78 + y_offset
        avg_particles_in_array = 6  # average number of particles (visible & invisible, i.e., ejected) that should be
        # in the sorter, defines the occupation density/mass flow of the sorter

    if FLAGS.mass_flow == "high":
        avg_particles_in_array *= 2

    total_birth_rate_mean = avg_particles_in_array / avg_num_time_steps_particle_in_range(
        (array_start, array_end), v0_x, T)

    # Define some default values, these are no environment settings, but good values (from experience) for the
    # learned controllers
    default_num_particles = int(array_end / (array_end - array_start) * avg_particles_in_array) + 1
    default_N = avg_num_time_steps_particle_in_range((tracking_start, array_end), v0_x, frame_factor * T) + 1

    length = 4
    width = 9
    disturbance_area_offset = np.array([-4, 0])
    disturbance_area_length = 8
    disturbance_area_width = 10
    phys_offset = (-6, 0)
    phys_length = 18
    phys_width = 10

    # Define the actors for the simulator
    actors = ActorSimulator(pos=actor_positions,
                            length=length,
                            width=width,
                            T=T,
                            **actor_dict)

    # Define the scenario, and set up the particles, the particle simulator, and the simulator
    particle_simulator_dict = {'total_num_particles': FLAGS.total_num_particles,
                               'process_parameters': 2 * ({'type': 'CV',
                                                           'pos_0': (-2.5, area_width / 2),
                                                           'pos0_stddev': (0.0, 3 * area_width),
                                                           'v_0': (v0_x, 0.0),
                                                           'v0_stddev': (v0_x / 10.0, v0_x / 10.0),
                                                           'S_w': (FLAGS.simulation_noise, FLAGS.simulation_noise),
                                                           },),
                               'birth_rate_mean_and_stddev': ((total_birth_rate_mean / 2, 0.3),
                                                              (total_birth_rate_mean / 2, 0.3)),
                               'class_labels': (0, 1),
                               'area_width': area_width,
                               'particle_r': 3,
                               }
    particle_simulator = AreaParticleSimulator(T=T,
                                               **particle_simulator_dict,
                                               S_v=(FLAGS.measurement_noise, FLAGS.measurement_noise),
                                               seed=FLAGS.particle_simulator_seed,
                                               )
    spatial_tolerance = v0_x * T / 2  # should be ~ v0x * T / 2 when using CV model
    simulator = AreaSortingSimulator(particle_simulator=particle_simulator,
                                     actors=actors,
                                     contact_simulator_dict={'spatial_tolerance': (spatial_tolerance, 0.0),
                                                             'disturbance_area_offset': disturbance_area_offset,
                                                             'disturbance_area_extent': (
                                                                 disturbance_area_length, disturbance_area_width),
                                                             },
                                     array_end=array_end,
                                     physical_actor_offset_length_width=(*phys_offset, phys_length, phys_width),
                                     )
    time_out = 30

    # Define the default particle and contact models for the controllers
    default_particle_model_dict = {'p_detect': 0.95,
                                   'S_w': (0.0001, 0.0001),
                                   'S_v': (0.01, 0.01),
                                   }
    default_contact_model_dict = {'p_eject': [0.0, 0.9, 0.2, 0.1],
                                  'use_sum_approximation': False,
                                  }

    def build_mtt_tracker(controller_rate):
        """Builds the MTT passed to the controller: oracle particle simulator or TrackSortMTTTracker.

        :param controller_rate: An integer, simulator steps per controller / MTT update (measurement interval factor).

        :returns: An AbstractParticleSimulator or TrackSortMTTTracker instance.
        """
        if not FLAGS.use_mtt_tracker:
            return particle_simulator

        measurement_interval = controller_rate * T
        avg_num_time_steps_per_particle = avg_num_time_steps_particle_in_range(
            (0.0, array_end), v0_x, measurement_interval)
        s_w_override = None
        s_v_override = None
        if FLAGS.match_tracker_noise:
            # this assumes both a CV model in simulator and the KF in TrackSort
            s_w_override, s_v_override = physical_noise_to_tracksort_s_w_s_v(
                S_w=(FLAGS.simulation_noise, FLAGS.simulation_noise),
                S_v=(FLAGS.measurement_noise, FLAGS.measurement_noise),
                x_normalization_constant=array_end,
                y_normalization_constant=area_width,
                time_normalization_constant=avg_num_time_steps_per_particle,
                measurement_interval=measurement_interval,
            )
            logging.info('Matching TrackSort KF noise to simulator: s_w=%s, s_v=%s', s_w_override, s_v_override)

        feed_pos_gt = tracking_start if FLAGS.feed_only_tracks_close_to_array_to_controller else None
        if feed_pos_gt is not None:
            logging.info(
                'Feeding only tracks with x > %.1f to controller (array_start=%.1f); '
                'TrackSort still tracks full FOV [0, %.1f]',
                feed_pos_gt, array_start, array_end,
            )

        return TrackSortMTTTracker(
            array_end=array_end,
            area_width=area_width,
            avg_num_time_steps_per_particle=avg_num_time_steps_per_particle,
            measurement_interval=measurement_interval,
            feed_tracks_with_position_greater=feed_pos_gt,
            feed_track_scores_to_controller=FLAGS.feed_track_scores_to_controller,
            s_w=s_w_override,
            s_v=s_v_override,
            verbosity_level=FLAGS.verbosity_level,
        )

    # Build the controller
    if FLAGS.controller == "first_actor_first":

        # Build the controller
        controller_rate = frame_factor
        controller = FirstActorFirstHeuristicController(
            build_mtt_tracker(controller_rate),
            actors=actors,
            T=controller_rate * T,
            actor_model_dict=actor_dict,
            **controller_config,
        )

    elif FLAGS.controller == "ptcr":

        # Build the controller
        controller_rate = frame_factor  # for this controller, we use the frame factor as the controller rate
        internal_T = controller_config.pop('internal_T', None)
        default_N = avg_num_time_steps_particle_in_range((tracking_start, array_end), v0_x,
                                                         internal_T if internal_T is not None else actors.t_cycle) + 1

        accept_cost_weight = controller_config.pop("accept_cost_weight", 0.5)
        controller = PTCRController(
            mtt_tracker=build_mtt_tracker(controller_rate),
            actors=actors,
            T=controller_rate * T,
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
        raise ValueError(f"Controller {FLAGS.controller} is not supported.")

    # Start the simulation
    simulator.reset(restore_original_scenario=True)  # reset the simulator to the initial state / fixed seed
    simulator.evaluate(controller,
                       time_out=time_out,
                       no_show_animation=FLAGS.no_show_animation,
                       save_animation=FLAGS.save_animation,
                       animation_dir=FLAGS.animation_dir,
                       result_path=FLAGS.result_dir + "/sorting_result.csv" if FLAGS.save_results else None,
                       )
    if hasattr(controller, "finalize_adf_mc_cost_eval"):
        controller.finalize_adf_mc_cost_eval()


if __name__ == "__main__":
    # get the definition of the flags
    define_flags(FLAG_NAMES_SET)

    app.run(main)
