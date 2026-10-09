from absl import logging

import matplotlib.pyplot as plt
import sys
import time


def flush_figure_generator(plot_func, mode='flush'):

    mode = mode

    def flush_figure(*args, **kwargs):
        nonlocal mode

        # We show the plot like a video stream, when it is updated, the new plot overwrites the old in the same figure.
        # see https://www.delftstack.com/de/howto/matplotlib/how-to-automate-plot-updates-in-matplotlib/
        plt.ion()  # activates interactive mode
        plt.gca().cla()  # clears axis of current, open figure

        # body of the plot function
        plot_func(*args, **kwargs)

        old_mode = mode  # mode from outer scope either press or flush (use nonlocal if required)
        if mode != 'flush':
            verbosity = logging.get_verbosity()
            logging.set_verbosity(logging.ERROR)  # surpresses logging from plt.waitforbuttonpress()
            plt.waitforbuttonpress()  # non specific key press, wait
            # while not plt.waitforbuttonpress():
            #     pass  # this might be a good idea when other interactions, e.g. zooming are required
            logging.set_verbosity(verbosity)
        plt.gcf().canvas.draw()
        plt.gcf().canvas.flush_events()

        def on_press(event):
            nonlocal mode

            sys.stdout.flush()
            if event.key == 'x':
                mode = 'press'
            if event.key == 'y':
                mode = 'flush'

        plt.gcf().canvas.mpl_connect('key_press_event', on_press)
        if old_mode == 'flush' and mode == 'press':
            logging.info(
                "Entering press mode. Press 'y' to exit press mode and enter flush mode. Press any other key to show next frame.")
        if old_mode == 'press' and mode == 'flush':
            logging.info("Entering flush mode. Press 'x' to exit flush mode and enter press mode.")
        time.sleep(0.1)

    return flush_figure
