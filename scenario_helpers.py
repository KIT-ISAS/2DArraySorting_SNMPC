"""Lightweight scenario geometry helpers shared by mains and tests.

Kept free of heavy dependencies (TensorFlow, sklearn, deep_iic, …) so unit /
integration tests can import these helpers without loading ``main``.
"""

import numpy as np


def calculate_x_offset(actor_dict, v, first_row_position, safety_factor=1.2, prec=5.0):
    """Calculates the x offset that the actors need to be shifted to the right in the simulation so that the controller
    has a thorough view on all relevant particles.

    The offset is based on the following observation:

                     v t_lead _________________
                       <---->|
                             |
           x           x     |
                             |
           <----------->     |
              v t_cycle      |
                             |__________________
                   first row of actors
        ----> x-axis (position)

         i) For a particle to be ejected on the first row of actors, it must be visible to the controller for the first
            time at a distance greater ~ v * t_lead, where t_lead is the time it takes to activate the actor so that it
            is able to hit the particle when it arrives at the first row of actors. This condition is checked in
            simulator.py and raises a warning if it is not fulfilled.
        ii) However, condition i) does not take into account that the motion of other particles influences the decision
            for an activation. For this, we consider two particles and require that the controller can see, at time
            t_lead before the first particle arrives, what happens in the subsequent time interval of length t_cycle,
            where t_cycle is the time it takes for a complete actor cycle, i.e., the time after which the actor is able
            to eject another particle after it has ejected one. Therefore, at time t_lead, the controller is able to
            make an informed decision whether it is better to activate the actor for the first particle or to wait and
            activate it for the second particle (or none of them).

    :param actor_dict: A dictionary containing the actor times.
    :param v: A float, the expected x-velocity of the particles.
    :param first_row_position: A float, the x-position of the first row of actors.
    :param safety_factor: A safety factor to account for that particles might be faster than expected.
    :param prec: A float, the precision for rounding up the result to the next multiple of prec for better readability
        of the plots.

    :returns: A float, the x-offset that the particles need to be shifted to the right in the simulation. The result is
        rounded up to the next multiple of 5 for better readability of the plots.
    """
    t_lead = actor_dict['t_activate'] + actor_dict['t_up'] + actor_dict['t_hit'] / 2
    t_cycle = actor_dict['t_activate'] + actor_dict['t_up'] + actor_dict['t_hit'] + actor_dict['t_down'] + actor_dict[
        't_reset']
    return float(np.ceil((safety_factor * v * (t_lead + t_cycle) - first_row_position) / prec) * prec)


def avg_num_time_steps_particle_in_range(start_end, v, T):
    """Calculates the number of time steps from start to end.

    :param start_end: A tuple (start, end) containing the start and end position of the array.
    :param v: A float, the expected x-velocity of the particles.
    :param T: A float representing the time interval between two consecutive time steps.

    :returns: An integer, the average number of time steps a particle is in the range, i.e., the number of time steps
        from start to end.
    """
    return int(np.ceil((start_end[1] - start_end[0]) / (v * T)))
