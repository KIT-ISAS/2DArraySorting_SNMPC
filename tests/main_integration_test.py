import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from absl import flags, logging

import shutil

import tensorflow as tf

from main import main, define_flags, FLAG_NAMES_SET

FLAGS = flags.FLAGS

FLAGS(sys.argv)  # need to explicitly to tell flags library to parse argv before you can access FLAGS.xxx (this needs
# to be done if not using absl.app.run(main))


class MainIntegrationTest(tf.test.TestCase):
    """Test cases for main.py.

    These tests use some of the settings also used in README.md. So it should be ensured that the examples in README.md
    are working.
    """

    @staticmethod
    def get_results_from_log_string(string, sum_disturbed_particles=False):
        """Extract the results from the log string.

        Example string:

            Evaluation result: TNs: 10, TPs: 5, FNs: 0, FPs: 4, Disturbed Ns: 0, Disturbed Ps: 0, TNR: 71.42%, TPR: 100%.

        :param string: A string from the logs
        :param sum_disturbed_particles: Whether to sum the disturbed negatives and positives particles to one number of
            disturbed particles.

        :returns: A list with the number of TNs, TPs, FNs, FPs, disturbed negatives and positives particles
            (if sum_disturbed_particle is False) or the total number of disturbed particle (if sum_disturbed_particles
            is True), and the TNR and TPR (in %).
        """
        result_string = string[-2]
        result_string = result_string[:-1]  # remove the last character (a dot)
        result_string_ls = result_string.replace(",", ":").split(": ")
        result = [int(result_string_ls[idx]) for idx in [2, 4, 6, 8, 10, 12]] + [float(result_string_ls[idx][:-2]) for
                                                                                 idx in [14, 16]]

        if sum_disturbed_particles:
            result[4] = result[4] + result[5]
            del result[5]

        return result

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Delete all FLAGS defined by other main files as we only want to keep the flags relevant for this file
        for name in list(flags.FLAGS):
            if name in FLAG_NAMES_SET:
                delattr(flags.FLAGS, name)
        define_flags(FLAG_NAMES_SET)

        cls.save_dir = '/mnt/tests/temporary_test_files/'

    def setUp(self):
        super().setUp()
        FLAGS.grid_size = "3x3"
        FLAGS.configuration = "A"
        FLAGS.simulation_noise = 0.0
        FLAGS.no_show_animation = True
        FLAGS.save_animation = False
        FLAGS.save_results = False

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_first_actor_first(self):
        FLAGS.controller = "first_actor_first"
        FLAGS.controller_config_path = None

        with self.subTest(name='Test for 3x3 without system noise'):
            with self.assertLogs(logger=logging.get_absl_logger(), level='INFO') as cm:
                main('args')
                results_outp = self.get_results_from_log_string(cm.output, sum_disturbed_particles=True)[:-2]

            exp_results = [39, 35,  4, 11, 11]
            self.assertAllEqual(exp_results, results_outp)

        with self.subTest(name='Test for 3x3 with system noise'):
            FLAGS.simulation_noise = 0.00001

            with self.assertLogs(logger=logging.get_absl_logger(), level='INFO') as cm:
                main('args')
                results_outp = self.get_results_from_log_string(cm.output, sum_disturbed_particles=True)[:-2]

            exp_results = [40, 34,  5,  9, 12]
            self.assertAllEqual(exp_results, results_outp)

        with self.subTest(name='Test for 2x2 without system noise'):
            FLAGS.simulation_noise = 0.0
            FLAGS.grid_size = "2x2"

            with self.assertLogs(logger=logging.get_absl_logger(), level='INFO') as cm:
                main('args')
                results_outp = self.get_results_from_log_string(cm.output, sum_disturbed_particles=True)[:-2]

            exp_results = [43, 34,  2, 12,  9]
            self.assertAllEqual(exp_results, results_outp)

        with self.subTest(name='Test for full grid without system noise'):
            FLAGS.simulation_noise = 0.0
            FLAGS.grid_size = "full"

            with self.assertLogs(logger=logging.get_absl_logger(), level='INFO') as cm:
                main('args')
                results_outp = self.get_results_from_log_string(cm.output, sum_disturbed_particles=True)[:-2]

            exp_results = [45, 37,  2,  1, 15]
            self.assertAllEqual(exp_results, results_outp)

    def test_ptcr(self):
        FLAGS.controller = "ptcr"
        FLAGS.controller_config_path = "/mnt/controller_configs/ptcr_config.json"

        with self.subTest(name='Test for full grid with system noise'):
            FLAGS.grid_size = "full"
            FLAGS.simulation_noise = 0.0000001

            with self.assertLogs(logger=logging.get_absl_logger(), level='INFO') as cm:
                main('args')
                results_outp = self.get_results_from_log_string(cm.output, sum_disturbed_particles=True)[:-2]

            exp_results = [47, 41, 2, 1, 9]
            self.assertAllEqual(exp_results, results_outp)


if __name__ == "__main__":
    # get the definition of the flags
    define_flags(FLAG_NAMES_SET)

    tf.test.main()
