from abc import ABC, abstractmethod

from enum import Enum, auto

import numpy as np


class ActorStatus(Enum):
    """The (discrete) status of an actor."""
    ACTIVATE = auto()
    UP = auto()
    HIT = auto()
    DOWN = auto()
    RESET = auto()
    READY = auto()


class Rectangle:
    """A rectangle in 2D space.
    
    The rectangle is defined by its lower left (x1, y1) and upper right (x2, y2) corner.
    
    From https://stackoverflow.com/questions/25068538/intersection-and-difference-of-two-rectangles
    """

    def __init__(self, x1, y1, x2, y2):
        """Initializes the rectangle.

        :param x1: A float, the x-coordinate of the lower left corner.
        :param y1: A float, the y-coordinate of the lower left corner.
        :param x2: A float, the x-coordinate of the upper right corner.
        :param y2: A float, the y-coordinate of the upper right corner.
        """
        self._x1, self._y1, self._x2, self._y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def __and__(self, other):
        """Calculates the intersection of two rectangles.

        :param other: A Rectangle object, the other rectangle.

        :returns: None or a Rectangle, the intersection of the two rectangles.
        """
        x1, y1, x2, y2 = max(self.x1, other.x1), max(self.y1, other.y1), \
            min(self.x2, other.x2), min(self.y2, other.y2)
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2, y2)

    @property
    def x1(self):
        """The x-coordinate of the lower left corner.

        :returns: A float, the x-coordinate of the lower left corner.
        """
        return self._x1

    @property
    def y1(self):
        """The y-coordinate of the lower left corner.

        :returns: A float, the y-coordinate of the lower left corner.
        """
        return self._y1

    @property
    def x2(self):
        """The x-coordinate of the upper right corner.

        :returns: A float, the x-coordinate of the upper right corner.
        """
        return self._x2

    @property
    def y2(self):
        """The y-coordinate of the upper right corner.

        :returns: A float, the y-coordinate of the upper right corner.
        """
        return self._y2

    intersection = __and__


class AbstractActorModel(ABC):
    """Abstract class for all actor models of the controller and simulator (either for 1D or 2D optical sorters).

    Along with the particle and the contact model, the actor model forms the system model of the stochastic model
    predictive controller.

    Assumptions:

        - The actor model is deterministic.
        - The state is fully observable.
        - The actors dynamic behavior is identical for all actors and does not change with time (no abrasion, etc.).

    Dynamic model:

        There exist different versions depending on whether the model is formulated in discrete or continuous time.

        With going back to READY status:

                    Actor
                   status
                s_{j,k}^act ^
                            |
                   HIT      |                    ___________
                            |                   |           |
                   UP/DOWN  |              _____|           |_____     slide/belt surface
                            | ------------|-----------------------|-------------------------------
                   ACTIVATE |        _____|                       |
                   READY    | ______|                             |                  _____________
                   RESET    |                                     |_________________|
                            --------------------------------------------------------|--------------->
                                    |––>  t_{j,k}^act                           t_cycle^act       Time t


        With direct reactivation:

                    Actor
                   status
                s_{j,k}^act ^
                            |
                   HIT      |                    ___________                                     __
                            |                   |           |                                   |
                   UP/DOWN  |              _____|           |_____     slide/belt surface  _____|
                            | ------------|-----------------------|-----------------------|--------
                   ACTIVATE |        _____|                       |                  _____|
                   READY    | ______|                             |                 |
                   RESET    |                                     |_________________|
                            --------------------------------------------------------|--------------->
                                    |––>  t_{j,k}^act                           t_cycle^act       Time t

    """

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
                 ):
        """Initializes the actor model.

        :param pos: A np.array of shape [num_actor] or [num_actors, 2] representing the positions of the actors. If
            the shape is [num_actors], an 1D optical sorter (groove sorter) is assumed and the positions need to be
            sorted from smallest to largest position. If the shape is [num_actors, 2], a 2D optical sorter (belt or
            chute) is assumed. In this case, the position format is [x, y], with the x-axis corresponding the transport
            direction, and pos needs to be sorted from smallest x-position to largest x-position. Actors with the same
            x-position need to be sorted from smallest y-position to largest y-position.
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
        """
        if np.shape(pos) != (np.shape(pos)[0], 2) and np.shape(pos) != (np.shape(pos)[0],):
            raise ValueError('Actor positions must be of shape [num_actors] or [num_actors, 2].')
        if np.ndim(pos) == 1 and width is not None or np.ndim(pos) != 1 and width is None:
            raise ValueError('If width is given, pos must be of shape [num_actors, 2] and vice versa.')
        if np.any(np.asarray(length) < 0):
            raise ValueError('Actor lengths must be greater equal zero.')
        if not np.isscalar(length) and np.shape(pos)[0] != np.shape(length)[0]:
            raise ValueError('If length is an array, it needs num_actors elements.')
        if width is not None and np.any(np.asarray(width) < 0):
            raise ValueError('Actor width must be greater equal zero.')
        if width is not None and not np.isscalar(width) and np.shape(pos)[0] != len(width):
            raise ValueError('If width is an array, it needs num_actors elements.')
        if np.ndim(pos) == 1 and np.any(np.diff(pos) <= 0):
            raise ValueError('Actor positions must be ordered so that pos is a strictly increasing array.')
        if np.ndim(pos) == 2:
            for index in range(np.shape(pos)[0] - 1):
                if pos[index + 1][0] < pos[index][0]:
                    raise ValueError('Actor positions must be ordered from smallest x-position to largest x-position')
                elif pos[index + 1][0] == pos[index][0]:
                    if pos[index + 1][1] < pos[index][1]:
                        raise ValueError(
                            'Actor positions must be ordered from smallest y-position to largest y-position'
                            'when the x-position stays the same')
        if not np.isscalar(T) or T <= 0:
            raise ValueError('The time interval between two consecutive time steps T must be a positive scalar.')
        if np.any(np.array([t_activate, t_up, t_hit, t_down, t_reset]) < 0):
            raise ValueError('Actor time intervals must be greater equal zero.')

        self._pos = np.asarray(pos)  # np.array of length num_actors
        self._length = length * np.ones(self._pos.shape[0])  # np.array of length num_actors
        self._width = width * np.ones(self._pos.shape[0]) if width is not None else None  # None or np.array of
        # width num_actors

        # check if the actors are non-overlapping
        if np.ndim(self._pos) == 1:
            self._check_one_dimensional_actor_grid()
        else:
            self.check_two_dimensional_actor_grid(self._pos, self._length, self._width)

        self._T = float(T)
        self._t_activate = float(t_activate)
        self._t_up = float(t_up)
        self._t_hit = float(t_hit)
        self._t_down = float(t_down)
        self._t_reset = float(t_reset)

    @property
    def pos(self):
        """The position of the actors.

         Format position:

            [x] if 1D optical sorter (groove sorter), [x, y] if 2D optical sorter (belt or chute)

        If the shape is [num_actors], the positions are sorted from smallest to largest position. If the shape is
        [num_actors, 2], it is sorted from smallest x-position to largest x-position. Actors with the same x-position
        need to be sorted from smallest y-position to largest y-position.

        :returns: A np.array of shape [num_actors] or [num_actors, 2] representing the positions of the actors.
        """
        return self._pos

    @property
    def length(self):
        """The lengths (operating ranges in x-direction) of the actors.

        :returns: A np.array of shape [num_actors] representing the lengths (operating ranges in x-direction) of the
            actors.
        """
        return self._length

    @property
    def width(self):
        """The widths (operating ranges in y-direction) of the actors.

        :returns: None or a np.array of shape [num_actors] representing the widths (operating ranges in y-direction) of
            the actors.
        """
        return self._width

    @property
    def num_rows(self):
        """The number of rows of the actors.

        :returns: An integer, the number of rows of the actors.
        """
        return len(np.unique(self._pos[:, 0] if self._pos.ndim == 2 else self._pos))

    @property
    def actors_in_row(self):
        """The number of columns of the actors.

        :returns: A np.array of shape [num_rows], the number of columns of the actors.
        """
        return np.unique(self._pos[:, 0] if self._pos.ndim == 2 else self._pos, return_counts=True)[1]

    @property
    def actor_grid(self):
        """The positions, lengths (and widths) of the actors in grid format.

         Format actor_grid:

            [x, length] if 1D optical sorter (groove sorter), [x, y, length, width] if 2D optical sorter (belt or chute)

        :returns: A list of length num_rows of np.array of shape [actors_in_row, 2 or 4] representing the positions
            and lengths (and widths) of the actors in grid format.
        """
        return self._transform_to_actor_grid_data(self._pos, self._length, self._width)

    @property
    def row_pos(self):
        """The x-positions of the actors in grid format.

        :returns: A np.array of shape [num_rows] representing the x-positions of the actors in grid format.
        """
        return np.array([pos_row[0, 0] for pos_row in self.actor_grid])

    @property
    def column_pos(self):
        """The y-positions of the actors in grid format.

        :returns: None or a list of length num_rows of np.array of shape [actors_in_row] representing the y-positions of
            the actors in grid format.
        """
        return [pos_row[:, 1] for pos_row in self.actor_grid] if self._pos.ndim == 2 else None

    @property
    def T(self):
        """The sampling time difference.

        :returns: A float representing the time interval between two consecutive time steps.
        """
        return self._T

    @property
    def t_activate(self):
        """The time interval after an activation until the actor reaches the surface.

        :returns: A float, the time interval after an activation until the actor reaches the surface.
        """
        return self._t_activate

    @property
    def t_up(self):
        """The time interval above the surface until reaching the optimal hitting time. The particle is not ejected but
        disturbed or only ejected with a reduced probability when hit during this time period.

        :returns: A float, the time interval above the surface until reaching the optimal hitting time. The particle
                is not ejected but disturbed when hit during this time period.
        """
        return self._t_up

    @property
    def t_hit(self):
        """The time interval when a hitting is optimal, i.e., the particles is ejected.

        :returns: A float, the time interval when a hitting is optimal, i.e., the particles is ejected.
        """
        return self._t_hit

    @property
    def t_down(self):
        """The time interval above the surface after t_hit until reaching the surface again. The particle is not ejected
        but disturbed or only ejected with a reduced probability when hit during this time period.

        :returns: A float, the time interval above the surface after t_hit until reaching the surface again. The particle
            is not ejected but disturbed when hit during this time period.
        """
        return self._t_down

    @property
    def t_reset(self):
        """The reset time. The time interval after an activation before a new activation is possible.

        :returns: A float, the reset time. The time interval after an activation before a new activation is possible.
        """
        return self._t_reset

    @property
    def t_cycle(self):
        """The time for one actor cycle, i.e., the time that is required by the actor before the next ejection process
        can be started.

        :returns: A float, the time for one actor cycle.
        """
        return self._t_activate + self._t_up + self._t_hit + self._t_down + self._t_reset

    @property
    def t_lead(self):
        """The smallest possible time interval before time step k at which the actuator must be activated so that it is
        able to hit at time step k.

        :returns: A float, the smallest possible time interval before time step k for hitting at k.
        """
        return self._t_activate + self._t_up + self._t_hit / 2

    @property
    def end_array(self):
        """The end coordinate of the actuator array, i.e., the position of the last actor in transport direction.

        :returns: A float, the end coordinate of the actuator array.
        """
        return self.pos[-1] if self._pos.ndim == 1 else float(np.max(self.pos[:, 0]))

    @abstractmethod
    def predict_actor_state(self, *args):
        """Calculates the actors' internal states at the next time step.

        :param args: Additional arguments that are required to calculate the actors' internal states at the next time
            step. See the subclasses for more information.

        :returns: A np.array of shape [num_actors_to_consider], the updated actor states.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def _transform_to_actor_grid_data(self, pos, length, width=None):
        """Transforms the positions, lengths (and widths) of the actors in grid format.

         Format actor_grid:

            [x, length] if 1D optical sorter (groove sorter), [x, y, length, width] if 2D optical sorter (belt or chute)

        :param pos: A np.array of shape [num_actors] or [num_actors, 2] representing the positions of the actors. If
            the shape is [num_actors], an 1D optical sorter (groove sorter) is assumed and the positions need to be
            sorted from smallest to largest position. If the shape is [num_actors, 2], a 2D optical sorter (belt or
            chute) is assumed. In this case, the position format is [x, y], with the x-axis corresponding the transport
            direction, and pos needs to be sorted from smallest x-position to largest x-position. Actors with the same
            x-position need to be sorted from smallest y-position to largest y-position.
        :param length: A float or a np.array of shape [num_actors] representing the lengths (operating ranges in
            x-direction) of the actors.
        :param width: None or a float or a np.array of shape [num_actors] representing the with (operating ranges in
            y-direction) of the actors. If None, an 1D optical sorter (groove sorter) is assumed. If given, a 2D
            optical sorter (belt or chute) is assumed.

        :returns: A list of length num_rows of np.array of shape [actors_in_row, 2 or 4] representing the positions
            and lengths (and widths) of the actors in grid format.
        """
        length_width = np.column_stack((length, width)) if width is not None else length
        grid_data = np.column_stack((pos, length_width))
        return np.split(grid_data, np.cumsum(self.actors_in_row)[:-1])

    def _check_one_dimensional_actor_grid(self):
        """Checks if the actors are not overlapping.

        If the actors are overlapping, a ValueError is raised.
        """
        for row_idx, (pos, length) in enumerate(zip(self._pos[:-1], self._length[:-1])):
            # check if the actors are not overlapping
            next_pos, next_length = self._pos[row_idx + 1], self._length[row_idx + 1]
            if pos + length / 2 > next_pos - next_length / 2:
                raise ValueError(
                    f'The operating ranges of the actors {row_idx + 1} and {row_idx + 2} must not overlap.')

    def check_two_dimensional_actor_grid(self, pos, length, width):
        """Checks if the actors are not overlapping within the rows and between the rows.

        Each actor is compared with the next actor in the same row and with all actors in the next row. If the actors
        are overlapping (the intersection is not None), a ValueError is raised.

        :param pos: A np.array of shape [num_actors, 2] representing the positions of the actors. pos needs to be sorted
            from smallest x-position to largest x-position. Actors with the same x-position need to be sorted from
            smallest y-position to largest y-position.
        :param length: A float or a np.array of shape [num_actors] representing the lengths (operating ranges in
            x-direction) of the actors.
        :param width: A float or a np.array of shape [num_actors] representing the with (operating ranges in
            y-direction) of the actors.
        """
        actor_grid = self._transform_to_actor_grid_data(pos, length, width)
        for row_idx, actor_row in enumerate(actor_grid):
            # check if the actors are not overlapping within the row
            for column_idx in range(len(actor_row) - 1):
                actor = actor_row[column_idx]
                next_actor = actor_row[column_idx + 1]
                actor_rect = Rectangle(actor[0] - actor[2] / 2,
                                       actor[1] - actor[3] / 2,
                                       actor[0] + actor[2] / 2,
                                       actor[1] + actor[3] / 2)
                next_actor_rect = Rectangle(next_actor[0] - next_actor[2] / 2,
                                            next_actor[1] - next_actor[3] / 2,
                                            next_actor[0] + next_actor[2] / 2,
                                            next_actor[1] + next_actor[3] / 2)
                intersection = actor_rect & next_actor_rect
                if intersection is not None:
                    raise ValueError(
                        f'The operating ranges of the actors "row {row_idx + 1}, column {column_idx + 1} and '
                        f'{column_idx + 2}" must not overlap.')
            # check if the actors are not overlapping with the actors in the next row
            if row_idx + 1 < len(actor_grid):
                next_row = actor_grid[row_idx + 1]
                for column_idx in range(len(actor_row)):
                    actor = actor_row[column_idx]
                    actor_rect = Rectangle(actor[0] - actor[2] / 2,
                                           actor[1] - actor[3] / 2,
                                           actor[0] + actor[2] / 2,
                                           actor[1] + actor[3] / 2)
                    for next_column_idx in range(len(next_row)):
                        next_actor = next_row[next_column_idx]
                        next_actor_rect = Rectangle(next_actor[0] - next_actor[2] / 2,
                                                    next_actor[1] - next_actor[3] / 2,
                                                    next_actor[0] + next_actor[2] / 2,
                                                    next_actor[1] + next_actor[3] / 2)
                        intersection = actor_rect & next_actor_rect
                        if intersection is not None:
                            raise ValueError(
                                f'The operating ranges of the actors "row {row_idx + 1}, column {column_idx + 1} and '
                                f'row {row_idx + 2}, column {next_column_idx + 1}" must not overlap.')


class ActorModelVariantB2(AbstractActorModel):
    """Actor model for variant B2 (continuous-time with intermediate time steps) of the controllers and for heuristics
    (either for 1D or 2D optical sorters).

    Along with the particle and the contact model, the actor model forms the system model of the stochastic model
    predictive controller.

    Variant B2 of the controller maps particles to activation times, i.e., an actor is activated so that at the time
    when a particle is expected to pass the actor, the actor is supposed to hit the particle. The controllers then only
    decide on the association of particles to actors.

    Assumptions:

        - The actor model is deterministic.
        - The state is fully observable.
        - The actors dynamic behavior is identical for all actors and does not change with time (no abrasion, etc.).

    Dynamic model:

        This actor model simply predicts the actor movement in continuous time without any notion of time steps.

        With going back to READY status:

                    Actor
                   status
                s_{j,k}^act ^
                            |
                   HIT      |                    ___________
                            |                   |           |
                   UP/DOWN  |              _____|           |_____     slide/belt surface
                            | ------------|-----------------------|-------------------------------
                   ACTIVATE |        _____|                       |
                   READY    | ______|                             |                  _____________
                   RESET    |                                     |_________________|
                            --------------------------------------------------------|--------------->
                                    |––>  t_{j,k}^act                           t_cycle^act       Time t


        With direct reactivation:

                    Actor
                   status
                s_{j,k}^act ^
                            |
                   HIT      |                    ___________                                     __
                            |                   |           |                                   |
                   UP/DOWN  |              _____|           |_____     slide/belt surface  _____|
                            | ------------|-----------------------|-----------------------|--------
                   ACTIVATE |        _____|                       |                  _____|
                   READY    | ______|                             |                 |
                   RESET    |                                     |_________________|
                            --------------------------------------------------------|--------------->
                                    |––>  t_{j,k}^act                           t_cycle^act       Time t

    """

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
                 ):
        """Initializes the actor model.

        :param pos: A np.array of shape [num_actor] or [num_actors, 2] representing the positions of the actors. If
            the shape is [num_actors], an 1D optical sorter (groove sorter) is assumed and the positions need to be
            sorted from smallest to largest position. If the shape is [num_actors, 2], a 2D optical sorter (belt or
            chute) is assumed. In this case, the position format is [x, y], with the x-axis corresponding the transport
            direction, and pos needs to be sorted from smallest x-position to largest x-position. Actors with the same
            x-position need to be sorted from smallest y-position to largest y-position.
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
        """
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

    def predict_actor_state(self, actors_next_time_ready, delta_toa, actor_indices, particle_indices):
        """Predicts the actors' next time in ready status given actor activations.

        Here, it is assumed that actors are activated such that they hit the particles at the respective expected
        arrival time at the actors.

        :param actors_next_time_ready: A np.array of shape [num_actors] containing the time when the actor will be
            in READY status for the first time again as delta w.r.t. the current time step.
        :param delta_toa: A np.array of shape [num_actors, num_particles], the particles' time of arrival as delta
            w.r.t. the current time step.
        :param actor_indices: An integer np.array or an integer, index or indices of the actors to be activated.
        :param particle_indices: An integer np.array or an integer, index or mask of the particles to be ejected.

        :returns: actors_next_time_ready: A np.array of shape [num_actors] containing the time when the actor will be
            in READY status for the first time again as delta w.r.t. the current time step.
        """
        # we check the type of particle indices since the required type was recently changed
        # TODO: Remove this again in future versions
        assert isinstance(particle_indices, int) or issubclass(particle_indices.dtype.type, np.integer), \
            "particle_indices must be an integer or an integer np.array."
        actors_next_time_ready[actor_indices] = delta_toa[actor_indices, particle_indices] + self.t_cycle - self.t_lead
        return actors_next_time_ready


class ActorModelVariantA(AbstractActorModel):
    """Actor model for variant A (discrete actor activation time) of the controller (either for 1D or 2D optical
    sorters).

    Along with the particle and the contact model, the actor model forms the system model of the stochastic model
    predictive controller.

    Variant A of the controller in each time steps uses the controls

        u_k = [ u_{1,k}   ...   u_{N_A,k} ]^T

    with binary components u_{j,k} in {0, 1} where u_{j,k} = 1 ≙ “activate actor 𝑗 at time step 𝑘” and N_A being the
    number of actors in the setup.

    Assumptions:

        - The actor model is deterministic.
        - The state is fully observable.
        - The actors dynamic behavior is identical for all actors and does not change with time (no abrasion, etc.).

    Dynamic model:

        With going back to READY status:

                    Actor
                   status
                s_{j,k}^act ^
                            |
                   HIT      |                    ___________
                            |                   |           |
                   UP/DOWN  |              _____|           |_____     slide/belt surface
                            | ------------|-----------------------|-------------------------------
                   ACTIVATE |        _____|                       |
                   READY    | ______|                             |                  _____________
                   RESET    |                                     |_________________|
                            --------------------------------------------------------|--------------->
                                    |––>  t_{j,k}^act                           t_cycle^act       Time t

                   Example    0     0     1     2     3     4     5     6     7     8     0     0
                   (with T=1)


        With direct reactivation:

                    Actor
                   status
                s_{j,k}^act ^
                            |
                   HIT      |                    ___________                                     __
                            |                   |           |                                   |
                   UP/DOWN  |              _____|           |_____     slide/belt surface  _____|
                            | ------------|-----------------------|-----------------------|--------
                   ACTIVATE |        _____|                       |                  _____|
                   READY    | ______|                             |                 |
                   RESET    |                                     |_________________|
                            --------------------------------------------------------|--------------->
                                    |––>  t_{j,k}^act                           t_cycle^act       Time t

                   Example    0     0     1     2     3     4     5     6     7     8     1     2
                   (with T=1)


        The dynamics of actor j is given by

                        {  t_{j,k}^act + T  , if t_{j,k}^act not in {0, t_cycle^act}
        t_{j,k+1}^act = {  T                , if u_{j,k} = 1 and t_{j,k}^act in {0, t_cycle^act}
                        {  0                , else ,

        where t_{j,k}^act describes the actor's internal time since its last activation (measured at the right edge of
        the discrete time interval) and t_cycle^act is the time for one actor cycle, i.e., the time that is required by
        the actor before the next ejection process can be started.

        The discrete status of the actor s_{j,k}^act is given by the corresponding output equation that maps
        t_{j,k}^act to s_{j,k}^act according to

                      {  ACTIVATE  , if 0 < t_{j,k}^act <= t_activate^act
                      {  UP        , if t_activate^act < t_{j,k}^act <= t_activate^act + t_up^act
        s_{j,k}^act = {  HIT       , if t_activate^act + t_up^act < t_{j,k}^act <= t_activate^act + t_up^act + t_hit^act
                      {  DOWN      , if t_activate^act + t_up^act + t_hit^act < t_{j,k}^act
                                            <= t_activate^act + t_up^act + t_hit^act + t_down^act
                      {  RESET     , if t_activate^act + t_up^act + t_hit^act + t_down^act < t_{j,k}^act
                                            <= t_activate^act + t_up^act + t_hit^act + t_down^act + t_reset^act
                      {  READY     , else .

        Here, t_activate^act, t_up^act, t_hit^ac, t_down^act, t_reset^act with t_activate^act + t_up^act + t_hit^act +
        t_down^act + t_reset^act = t_cycle^act are time intervals. s_{j,k}^act models the relevant stages in which the
        actor interacts with its environment, in particular, with the particles in its operating range. See the
        contact model for an explanation.
    """

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
                 ):
        """Initializes the actor model.

        :param pos: A np.array of shape [num_actor] or [num_actors, 2] representing the positions of the actors. If
            the shape is [num_actors], an 1D optical sorter (groove sorter) is assumed and the positions need to be
            sorted from smallest to largest position. If the shape is [num_actors, 2], a 2D optical sorter (belt or
            chute) is assumed. In this case, the position format is [x, y], with the x-axis corresponding the transport
            direction, and pos needs to be sorted from smallest x-position to largest x-position. Actors with the same
            x-position need to be sorted from smallest y-position to largest y-position.
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
        """
        if np.any(np.mod(np.array([t_activate, t_up, t_hit, t_down, t_reset]), T) != 0):
            raise ValueError('Actor time intervals must be multiples of the time interval T.')

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

    def predict_actor_state(self, t_act, u_act):
        """Calculates the actors' internal time and status at the next time step.

        :param t_act: A np.array of shape [num_actors] representing the actors' internal time t_k^act at time step
            k.
        :param u_act: A Boolean np.array of shape [num_actors] representing the control inputs u_k at time step k, where
            True indicates that the actor should be activated in the current time step.

        :returns:
            t_act: A np.array of shape [num_actors] representing the actors' internal time t_{k+1}^act at time step k+1.
            s_act: An object np.array of shape [num_actors] representing the actors' status s_{k+1}^act at time step
                k+1.
        """
        # increment the internal actor time (dynamic equation)
        t_act = np.where(np.logical_not(
            np.isin(t_act, [0, self._t_activate + self._t_up + self._t_hit + self._t_down + self._t_reset])),
            t_act + self._T, 0.0)
        t_act[np.logical_and(u_act, t_act == 0)] = self._T  # because it was already set to 0 if
        # t_act == self.t_cycle - self._T by the previous line

        s_act = self.output_actor_status(t_act)

        return t_act, s_act

    def output_actor_status(self, t_act):
        """Calculates the actors' status given the current actor time.

        This function constitutes the output equation of the actor model.

        :param t_act: A np.array of shape [num_actors] representing the actors' internal time t_k^act at time step k.

        :returns:
            s_act: An object np.array of shape [num_actors] representing the actors' status s_k^act at time step k.
        """
        activate_mask = np.logical_and(0 < t_act, t_act <= self._t_activate)
        up_mask = np.logical_and(self._t_activate < t_act, t_act <= self._t_activate + self._t_up)
        hit_mask = np.logical_and(self._t_activate + self._t_up < t_act,
                                  t_act <= self._t_activate + self._t_up + self._t_hit)
        down_mask = np.logical_and(self._t_activate + self._t_up + self._t_hit < t_act,
                                   t_act <= self._t_activate + self._t_up + self._t_hit + self._t_down)
        reset_mask = np.logical_and(self._t_activate + self._t_up + self._t_hit + self._t_down < t_act,
                                    t_act <= self._t_activate + self._t_up + self._t_hit + self._t_down + self._t_reset)

        s_act = np.array(len(t_act) * [ActorStatus.READY], dtype=object)
        s_act[activate_mask] = ActorStatus.ACTIVATE
        s_act[up_mask] = ActorStatus.UP
        s_act[hit_mask] = ActorStatus.HIT
        s_act[down_mask] = ActorStatus.DOWN
        s_act[reset_mask] = ActorStatus.RESET

        return s_act


class ActorModelVariantB(AbstractActorModel):
    """Actor model for variant B (continuous-time with intermediate time steps) of the controller (either for 1D or 2D
    optical sorters).

    Along with the particle and the contact model, the actor model forms the system model of the stochastic model
    predictive controller.

    Variant B of the controller in each time steps uses the controls

        u_k = [ u_{1,k}   ...   u_{N_A,k} ]^T

    with N_A being the number of actors in the setup and u_{j,k} in R is such that it coincides with the time
    t_activate^act + t_up^act + t_hit^act / 2, i.e., it is the center time of the HIT-status.
    # TODO: Potentially change the inputs from hitting to activation times (however, no issues are currently known)

    Assumptions:

        - The actor model is deterministic.
        - The state is fully observable.
        - The actors dynamic behavior is identical for all actors and does not change with time (no abrasion, etc.).

        - The sampling time interval 𝑇 is smaller than or equal to the duration of one actor cycle.
            ===> At each time step, an actor can be activated at most once.

    Additional conventions:

        - All times t are given as delta times w.r.t. the current time step k, i.e., t in [0, T].
        - We exclude u_k^act from the last convention and allow u_k^act in R to enable the use of unconstrained
            optimization (although it should be constrained if possible)

    Dynamic model:

                         ^
          Internal actor |
            time t^act   |                   |-------t_cycle^act--------|
                         | ......  ............ |  ................ | ................. | ......
           (t_cycle^act) |                      |                   |  /                |
                         (t^act_{k}(T))....  .. | ................. |/                  |
                         |                      |                  /|                   |      /
           RESET         |                      |           ____ /  |                   |    /
                         |                      |        1 |   /    |                   |  /
           DOWN          |                      |          | /      |                   |/
                         |                      |          /        |                  /|
           HIT           |-------  -------------|--------/----------|----------------/--|-------
               (t_hit_B) | ......  ............ | ...  / |  ....... | ..........   / .. | ......
           UP            |                      |    /              |            / |    |
                         |                      |  /     |          |          /        |
           ACTIVATE      (t^act_{k-1}(T))...  ..|/                  |        /     |    |
                         |                     /|        |          |   ___/            |
           READY     (0) |          _________/  |                   |      |       |    |
                         |                   |  |      u_k^act      |       u_{k+1}^act |
                         --------  ----------|--|--------|----------|------|-------|----|--------------->
                                         c(k-1) |––> t_k^act              c(k+1)                      Time t
                                                k                  k+1                  k+2         Time step
                                                |---------T---------|

    I) Forward actor dynamics

    The dynamics of an actor j are given by a mapping from time t to internal time t^act. In general, this a
    function

          t^act = f(t, t^act_{k-1}(T), u_k^act) :  [0, T] x [0, t_cycle^act) x R -> [0, t_cycle^act)

    of time t, the actor status at the end of the previous time step t^act_{k-1}(T) and the control input u_k^act.
    The function itself has a sawtooth profile with linear and constant (only t^act=0) segments. Note that the line has
    a slope of 1 since time passes equally for both coordinate systems (time and internal time coordinate system). If
    the actor moves the coordinate systems are shifted according to the equation

           t^act = t - c . (we use a minus here for convenience.)

    The shifting constant c can be determined by looking at the point

        t^act_{k-1}(T)) = 0 - c . (see first window of the sketch above.)

    or

        t_activate^act + t_up^act + t_hit^act / 2 = u_k^act - c(k)  , (see second window of the sketch above)

            -->   c(k) = u_k^act - (t_activate^act + t_up^act + t_hit^act / 2),

            (i.e., c(k) = the actor's activation time)


    There are three cases to consider:

        i)  the actor movement is due to a previous activation (due to t^act_{k-1}(T)),
       ii)  the actor movement is due to an activation in the current time step (due to u_k^act), or
      iii)  there is no actor movement.

        Case i) Actor movement due to a previous activation:

            t^act = t + t^act_{k-1}(T)   if  t^act_{k-1}(T) > 0                   , i.e., there is a previous activation
                                               and
                                             t + t^act_{k-1}(T) < t_cycle^act     , i.e., t is still influenced by
                                                                                        t^act_{k-1}(T).

        Case ii) Actor movement due to an activation in the current time step:

            t^act = t - c(k)             if  0 <= c(k) < T                        , i.e., there is an activation
                                               and
                                             c(k) >= t_cycle^act - t^act_{k-1}(T) , i.e., the activation is not within
                                                    if t^act_{k-1}(T) > 0               a previous actor movement
                                               and
                                             t >= t_cycle^act - t^act_{k-1}(T)    , i.e., t is not influenced by
                                                    if t^act_{k-1}(T) > 0               t^act_{k-1}(T)
                                               and
                                             t >= c(k)                            , i.e., t is influenced by u_k^act.

            Note that the first two requirements are independent of t and only check if u_k^act is a valid activation
            while the latter two describe the time range in which the line segment is valid.

            Furthermore, note that if t^act_{k-1}(T) > 0 and if t >= c(k) >= t_cycle^act - t^act_{k-1}(T) (fourth and
            second condition), the third condition, t >= t_cycle^act - t^act_{k-1}(T), is automatically fulfilled.
            (Also the fourth condition ensures the part c(k) < T of the first condition if t < T.)

        Case iii) There is no actor movement:

            t^act = 0                    if not in case i) or case ii).

        Note that this case results in ambiguities since t^act = 0 for a wide range of t, t^act_{k-1}(T), and u_k^act.

    II) Inverse actor dynamics

    Additionally to the forward actor dynamics, we require an inverse actor model outputting time given an internal
    actor time (e.g., the time when a HIT-phase begins, depicted by t_hit_B in the sketch above), i.e., we need to
    invert f(.). This results in a function

        t = g(t^act, t^act_{k-1}(T), u_k^act) :  (0, t_cycle^act) x [0, t_cycle^act) x R -> {[0, T) , inf}

    that maps the internal actor time t^act, the actor status at the end of the previous time step t^act_{k-1}(T) and
    the control input u_k^act to the time t.

    Note that for t^act = 0, the continuous actor dynamics is not bijective while for all other t^act in
    (0, t_cycle^act), the function is (because of the assumption that the sampling time interval 𝑇 is smaller than or
    equal to the duration of one actor cycle t_cycle^act). For this reason, we restrict the domain of g(.) to the range
    (0, t_cycle^act) where the inversion is properly (without ambiguities) defined.

    We again need to consider the three cases defined before:

        Case i) Actor movement due to a previous activation:

            t = t^act - t^act_{k-1}(T)   if  t^act_{k-1}(T) > 0                   , i.e., there is a previous activation
                                               and
                                             t^act_{k-1}(T) <= t^act              , this comes from the requirement that
                                                                                      t^act(0) = t^act_{k-1}(T) + 0
                                                                                                           <= t^act(t)
                                                                                      (otherwise t^act(t) cannot be
                                                                                       caused by t^act_{k-1}(T) ).

                                               and
                                             t^act - t^act_{k-1}(T) < T           , this comes from the requirement that
                                                                                      t^act(T) = t^act_{k-1}(T) + T
                                                                                                            > t^act(t)
                                                                                      (otherwise t^act(t) cannot be
                                                                                       reached within the current time
                                                                                       step).

        Case ii) Actor movement due to an activation in the current time step:

            t = t^act + c(k)             if  0 <= c(k) < T                        , i.e., there is an activation
                                               and
                                             c(k) >= t_cycle^act - t^act_{k-1}(T) , i.e., the activation is not within
                                                    if t^act_{k-1}(T) > 0             a previous actor movement
                                               and
                                             t^act < t^act_{k-1}(T)               , i.e., t^act is not influenced by
                                                    if t^act_{k-1}(T) > 0             t^act_{k-1}(T)
                                               and
                                             t^act < T - c(k)                     , this comes from the requirement that
                                                                                      t^act(T) = T - c(k) > t^act(t)
                                                                                      (otherwise t^act(t) cannot be
                                                                                       reached within the current time
                                                                                       step).

            Note that the first two requirements are the same as before (check if u_k^act is a valid activation).

            Furthermore, note that if t^act_{k-1}(T) > 0 and if t^act < T - c(k) <= t_cycle^act - c(k) <= t^act_{k-1}(T)
            (fourth condition, T <= t_cycle^act by the assumptions, and second condition), the third condition
            t^act < t^act_{k-1}(T) is automatically satisfied. (Also the fourth condition ensures the part c(k) < T of
            the first condition since T - t^act <= T.)

        Case iii) There is no actor movement:

            t = np.inf                   if not in case i) or case ii).
    """

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
                 ):
        """Initializes the actor model.

        :param pos: A np.array of shape [num_actor] or [num_actors, 2] representing the positions of the actors. If
            the shape is [num_actors], an 1D optical sorter (groove sorter) is assumed and the positions need to be
            sorted from smallest to largest position. If the shape is [num_actors, 2], a 2D optical sorter (belt or
            chute) is assumed. In this case, the position format is [x, y], with the x-axis corresponding the transport
            direction, and pos needs to be sorted from smallest x-position to largest x-position. Actors with the same
            x-position need to be sorted from smallest y-position to largest y-position.
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
        """
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

        if self.t_cycle < T:
            raise ValueError('For this model to be applicable, the sampling time interval T must be '
                             'smaller than or equal to the time for one actor cycle.')

    def predict_actor_state(self, t_act_k_minus_one_at_T, u_act):
        """Calculate the internal actor time at the end of the current time step.

        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns: t_act_k_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle)
            representing the internal actor times t^act of the actors at the end of the current time step (at time T).
        """
        t_prev = np.asarray(t_act_k_minus_one_at_T, dtype=float)
        u = np.asarray(u_act, dtype=float)
        try:
            from adf.adf_numba_kernels import numba_enabled, predict_actor_state_numba
        except ImportError:
            numba_enabled = lambda: False  # noqa: E731
            predict_actor_state_numba = None

        if numba_enabled() and predict_actor_state_numba is not None and t_prev.ndim == 1:
            return predict_actor_state_numba(
                np.ascontiguousarray(t_prev, dtype=np.float64),
                np.ascontiguousarray(u, dtype=np.float64),
                float(self._T),
                float(self._T),
                float(self.t_cycle),
                float(self._t_activate),
                float(self._t_up),
                float(self._t_hit),
            )
        return self._continuous_actor_dynamics(
            t=self._T, t_act_k_minus_one_at_T=t_prev, u_act=u
        )

    def output_phase_time_intervals(self, t_act_k_minus_one_at_T, u_act):
        """HIT / UP / DOWN begin–end times within the current stage (fused).

        Prefer this over three separate ``output_t_*_interval`` calls on the ADF
        hot path (one Numba pass).

        :returns: ``(t_hit, t_up, t_down)``, each ``(K, 2)``.
        """
        t_prev = np.asarray(t_act_k_minus_one_at_T, dtype=float)
        u = np.asarray(u_act, dtype=float)
        try:
            from adf.adf_numba_kernels import actor_phase_intervals_numba, numba_enabled
        except ImportError:
            numba_enabled = lambda: False  # noqa: E731
            actor_phase_intervals_numba = None

        if numba_enabled() and actor_phase_intervals_numba is not None and t_prev.ndim == 1:
            return actor_phase_intervals_numba(
                np.ascontiguousarray(t_prev, dtype=np.float64),
                np.ascontiguousarray(u, dtype=np.float64),
                float(self._T),
                float(self.t_cycle),
                float(self._t_activate),
                float(self._t_up),
                float(self._t_hit),
                float(self._t_down),
            )
        return (
            self.output_t_hit_interval(t_prev, u),
            self.output_t_up_interval(t_prev, u),
            self.output_t_down_interval(t_prev, u),
        )

    def output_t_up_interval(self, t_act_k_minus_one_at_T, u_act):
        """Calculates the start and end time of the UP-actor status within the current time step.

        The start and end time interval is defined as:

            [t_up_begin, t_up_end] , if both t_up_begin and t_up_end are reached within the current time step,
            [t_up_begin, T]        , if only t_up_begin is reached within the current time step,
            [0, t_up_end]          , if only t_up_end is reached within the current time step, i.e., the UP-status
                                        was already started but not yet finished at previous time step.
            [inf, inf]             , if neither t_up_begin nor t_up_end is reached within the current time step (because
                                        the actor was not activated or activated too late to reach the UP-status).

        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns: A np.array of shape [num_actors_to_consider, 2] with elements in [0, T] or np.inf, where T is the
            sampling time interval, representing the time as delta w.r.t. the current time step k when the beginning and
            the end of the UP-status is reached.
        """
        return self._output_actor_status_time_interval(t_act_status_begin=self._t_activate,
                                                       duration=self._t_up,
                                                       t_act_k_minus_one_at_T=t_act_k_minus_one_at_T,
                                                       u_act=u_act)

    def output_t_hit_interval(self, t_act_k_minus_one_at_T, u_act):
        """Calculates the start and end time of the HIT-actor status within the current time step.

        The start and end time interval is defined as:

            [t_hit_begin, t_hit_end] , if both t_hit_begin and t_hit_end are reached within the current time step,
            [t_hit_begin, T]         , if only t_hit_begin is reached within the current time step,
            [0, t_hit_end]           , if only t_hit_end is reached within the current time step, i.e., the HIT-status
                                        was already started but not yet finished at previous time step.
            [inf, inf]               , if neither t_hit_begin nor t_hit_end is reached within the current time step 
                                        (because the actor was not activated or activated too late to reach the 
                                        HIT-status).

        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns: A np.array of shape [num_actors_to_consider, 2] with elements in [0, T] or np.inf, where T is the
            sampling time interval, representing the time as delta w.r.t. the current time step k when the beginning and
            the end of the HIT-status is reached.
        """
        return self._output_actor_status_time_interval(t_act_status_begin=self._t_activate + self._t_up,
                                                       duration=self._t_hit,
                                                       t_act_k_minus_one_at_T=t_act_k_minus_one_at_T,
                                                       u_act=u_act)
    
    def output_t_down_interval(self, t_act_k_minus_one_at_T, u_act):
        """Calculates the start and end time of the DOWN-actor status within the current time step.

        The start and end time interval is defined as:

            [t_down_begin, t_down_end] , if both t_down_begin and t_down_end are reached within the current time step,
            [t_down_begin, T]          , if only t_down_begin is reached within the current time step,
            [0, t_down_end]            , if only t_down_end is reached within the current time step, i.e., the 
                                          DOWN-status was already started but not yet finished at previous time step.
            [inf, inf]                 , if neither t_down_begin nor t_down_end is reached within the current time step 
                                          (because the actor was not activated or activated too late to reach the 
                                          DOWN-status).

        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns: A np.array of shape [num_actors_to_consider, 2] with elements in [0, T] or np.inf, where T is the
            sampling time interval, representing the time as delta w.r.t. the current time step k when the beginning and
            the end of the DOWN-status is reached.
        """
        return self._output_actor_status_time_interval(t_act_status_begin=self._t_activate + self._t_up + self._t_hit,
                                                       duration=self._t_down,
                                                       t_act_k_minus_one_at_T=t_act_k_minus_one_at_T,
                                                       u_act=u_act)

    def _continuous_actor_dynamics(self, t, t_act_k_minus_one_at_T, u_act):
        """The continuous actor dynamics within the current time step.

        A function

            t^act = f(t, t^act_{k-1}(T), u_k^act) :  [0, T] x [0, t_cycle^act) x R -> [0, t_cycle^act)

        mapping the time t, the actor status at the end of the previous time step t^act_{k-1}(T) and the control input
        u_k^act to the internal actor time t^act.

        :param t: A float in [0, T] where T is the sampling time interval, the current time as delta w.r.t. the current
            time step k.
        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns: t_act_k_at_t: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle)
            representing the internal actor times t^act of the actors at time t.
        """
        if 0 > t or t > self._T:
            raise ValueError('The difference time t must be within [0, T].')
        if np.any(np.logical_or(0 > t_act_k_minus_one_at_T, t_act_k_minus_one_at_T >= self.t_cycle)):
            raise ValueError('The internal actor time at the end of the last time step t_act_k_minus_one_at_T t must be '
                             'within [0, self.t_cycle).')

        # We need to distinguish three cases: i) The actor movement is due to a previous activation, ii) the movement is
        # due to u_act, iii) there is no actor movement

        # iii) there is no actor movement
        t_act_k_at_t = np.zeros_like(t_act_k_minus_one_at_T, dtype=float)

        # i) actor movement is due to a previous activation
        # this part is independent of u_act
        still_active_from_last_time_step_mask = np.logical_and(0 < t_act_k_minus_one_at_T,
                                                               t + t_act_k_minus_one_at_T < self.t_cycle)
        t_act_k_at_t[still_active_from_last_time_step_mask] = t + t_act_k_minus_one_at_T[
            still_active_from_last_time_step_mask]
        # note that it jumps to (actually stays at) 0 if t + t_act_k_minus_one_at_T >= t_cycle

        # iii) actor movement is due to u_act
        # this part is dependent on u_act, u_act needs to satisfy three requirements checked below
        valid_u_act_mask = self._check_for_valid_u_act(t_act_k_minus_one_at_T, u_act)
        # additionally, t must be such that an actor movement because of t_act_k_minus_one_at_T is not present, and t
        # must be such that the actor activation caused the current t_act_k_at_t (latter is fulfilled automatically, see
        # the docstring of the class)
        start_activation_time = u_act - self._t_activate - self._t_up - self._t_hit / 2
        t_in_movement_from_activation_mask = t >= start_activation_time

        # bring it all together
        activated_in_current_time_step_mask = np.logical_and(valid_u_act_mask, t_in_movement_from_activation_mask)
        t_act_k_at_t[activated_in_current_time_step_mask] = t - start_activation_time[
            activated_in_current_time_step_mask]
        # note that if self.T < self.t_cycle, it's not possible to activate an actor and finish the actor cycle (arrive
        # at t_cycle) within the same time step, that's why in this case, we do not need to ensure that t_act_k_at_t is
        # 0 when t_act_k_at_t >= t_cycle. However, for the edge case self.T == self.t_cycle, we need to check that.
        if self._T == self.t_cycle:
            cycle_finished_mask = t_act_k_at_t == self.t_cycle
            t_act_k_at_t[np.logical_and(cycle_finished_mask, activated_in_current_time_step_mask)] = 0.0

        assert np.all(np.logical_and(t_act_k_at_t >= 0, t_act_k_at_t < self.t_cycle))
        return t_act_k_at_t

    def _inverse_continuous_actor_dynamics(self, t_act_k_at_t, t_act_k_minus_one_at_T, u_act):
        """The inverse continuous actor dynamics within the current time step.

        A function

            t = g(t^act, t^act_{k-1}(T), u_k^act) :  (0, t_cycle^act) x [0, t_cycle^act) x R -> {[0, T) , inf}

        mapping the internal actor time t^act, the actor status at the end of the previous time step t^act_{k-1}(T) and
        the control input u_k^act to the time t.

        Note that for t^act = 0, the continuous actor dynamics is not bijective while for all other t^act in
        (0, t_cycle^act), the function is (because of the assumption that the sampling time interval 𝑇 is smaller than
        one actor cycle t_cycle^act). For this reason, we restrict the domain of g(.) to the range (0, t_cycle^act)
        where the inversion is properly (without ambiguities) defined.

        If t_act_k_at_t is not reached by an actor within the current time step (e.g., because it was not activated or
        activated too late), np.inf is returned for this actor.

        :param t_act_k_at_t: A np.array of shape [num_actors_to_consider] with elements in (0, self.t_cycle)
            representing the internal actor times t^act of the actors at time t.
        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns t: A np.array of shape [num_actors_to_consider] with elements in [0, T) or np.inf, where T is the
            sampling time interval, representing the time as delta w.r.t. the current time step k when the internal
            actor time t_act_k_at_t is reached.
        """
        if np.any(np.logical_or(0 >= t_act_k_at_t, t_act_k_at_t >= self.t_cycle)):
            raise ValueError(
                'The internal actor time t_act_k_at_t must be within (0, self.t_cycle). This is partly because for '
                't_act_k_at_t=0 the function is not bijective, so that there are ambiguous solutions.')
        if np.any(np.logical_or(0 > t_act_k_minus_one_at_T, t_act_k_minus_one_at_T >= self.t_cycle)):
            raise ValueError(
                'The internal actor time at the end of the last time step t_act_k_minus_one_at_T t must be '
                'within [0, self.t_cycle).')

        t_act_k_at_t = t_act_k_at_t * np.ones_like(t_act_k_minus_one_at_T) if np.isscalar(
            t_act_k_at_t) else t_act_k_at_t

        # We need to distinguish two cases: i) The actor movement is due to a previous activation, ii) the movement is
        # due to u_act, iii) there is no actor movement

        # iii) there is no actor movement
        t = np.empty_like(u_act, dtype=float)
        t[:] = np.inf

        # i) actor movement is due to a previous activation
        # this part is independent of u_act
        # For t_act_k_at_t caused by a previous activation, the actor must be in movement and the beginning of the
        # t_act_k_at_t must not be already reached in the previous time step, and it must be reached within the current
        # time step
        t_act_k_at_t_from_previous_activation_mask = np.logical_and.reduce((
            0 < t_act_k_minus_one_at_T,
            t_act_k_at_t - self._T < t_act_k_minus_one_at_T,
            t_act_k_minus_one_at_T <= t_act_k_at_t))
        t[t_act_k_at_t_from_previous_activation_mask] = t_act_k_at_t[t_act_k_at_t_from_previous_activation_mask] - \
                                                        t_act_k_minus_one_at_T[
                                                            t_act_k_at_t_from_previous_activation_mask]

        # ii) actor movement is due to u_act
        # this part is dependent on u_act, u_act needs to satisfy three requirements checked below
        valid_u_act_mask = self._check_for_valid_u_act(t_act_k_minus_one_at_T, u_act)
        # additionally, u must be such that the actor reaches t_act_k_at_t within the current time step
        start_activation_time = u_act - self._t_activate - self._t_up - self._t_hit / 2
        reaching_t_act_k_at_t_from_activation_within_current_time_step_mask = start_activation_time + t_act_k_at_t < self._T
        # bring it all together
        t_act_k_at_t_from_activation_mask = np.logical_and(valid_u_act_mask,
                                                           reaching_t_act_k_at_t_from_activation_within_current_time_step_mask)
        t[t_act_k_at_t_from_activation_mask] = start_activation_time[t_act_k_at_t_from_activation_mask] + t_act_k_at_t[
            t_act_k_at_t_from_activation_mask]

        assert np.all(np.logical_or(np.isinf(t), np.logical_and(t >= 0, t < self._T)))
        return t

    def _check_for_valid_u_act(self, t_act_k_minus_one_at_T, u_act):
        """Checks if the control inputs satisfy the constraints.

        The constraints for the control inputs are:

              i)  c(k) >= 0                              , i.e., the activation starts not in the previous time step,
             ii)  c(k) < T                               , i.e., the activation starts not in subsequent time step,
            iii)  c(k) >= t_cycle^act - t^act_{k-1}(T)   , i.e., the activation is not within a previous actor movement,

        where c(k) = u_k^act - (t_activate^act + t_up^act + t_hit^act / 2)

        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns: valid_u_act_mask: A Boolean np.array of shape [num_actors_to_consider], a mask for valid (True)
            control inputs satisfying the constraints.
        """
        # u_act needs to satisfy three requirements:
        start_activation_time = u_act - self._t_activate - self._t_up - self._t_hit / 2
        # 1) activation is not in previous time step
        activation_not_too_early_mask = start_activation_time >= 0
        # 2) activation is not in the subsequent time step
        activation_not_too_late_mask = start_activation_time < self._T
        # 3) activation is not within a previous actor movement
        activation_not_in_previous_activation_mask = np.ones_like(t_act_k_minus_one_at_T, dtype=bool)
        activation_not_in_previous_activation_mask[0 < t_act_k_minus_one_at_T] = self.t_cycle - t_act_k_minus_one_at_T[
            0 < t_act_k_minus_one_at_T] <= start_activation_time[0 < t_act_k_minus_one_at_T]

        valid_u_act_mask = np.logical_and.reduce(
            (activation_not_too_early_mask, activation_not_too_late_mask, activation_not_in_previous_activation_mask))
        return valid_u_act_mask

    def _output_actor_status_time_interval(self, t_act_status_begin, duration, t_act_k_minus_one_at_T, u_act):
        """Calculates the start and end time of the actor status within the current time step.

        The start and end time interval is defined as:

            [t_status_begin, t_status_end] , if both t_status_begin and t_status_end are reached within the current time
                                                step,
            [t_status_begin, T]            , if only t_status_begin is reached within the current time step,
            [0, t_status_end]              , if only t_status_end is reached within the current time step, i.e., the
                                                status was already started but not yet finished at previous time step.
            [inf, inf]                     , if neither t_status_begin nor t_status_end is reached within the current
                                                time step (because the actor was not activated or activated too late to
                                                reach the status).

        :param t_act_status_begin: A float with elements in (0, self.t_cycle) representing the internal actor time when
            the status begins.
        :param duration: A float representing the duration of the status.
        :param t_act_k_minus_one_at_T: A np.array of shape [num_actors_to_consider] with elements in [0, self.t_cycle).
            The internal actor time t^act at the end of the previous time step k-1.
        :param u_act: A np.array of shape [num_actors_to_consider] representing the control inputs u_k at time step k.
            The elements represent the time at which the actor should be in the center of its HIT-status as delta
            w.r.t. the current time step.

        :returns: A np.array of shape [num_actors_to_consider, 2] with elements in [0, T] or np.inf, where T is the
            sampling time interval, representing the time as delta w.r.t. the current time step k when the beginning and
            the end of the respective status is reached.
        """
        t_prev = np.asarray(t_act_k_minus_one_at_T, dtype=float)
        u = np.asarray(u_act, dtype=float)
        try:
            from adf.adf_numba_kernels import actor_status_interval_numba, numba_enabled
        except ImportError:
            numba_enabled = lambda: False  # noqa: E731
            actor_status_interval_numba = None

        if numba_enabled() and actor_status_interval_numba is not None and t_prev.ndim == 1:
            return actor_status_interval_numba(
                float(t_act_status_begin),
                float(duration),
                np.ascontiguousarray(t_prev, dtype=np.float64),
                np.ascontiguousarray(u, dtype=np.float64),
                float(self._T),
                float(self.t_cycle),
                float(self._t_activate),
                float(self._t_up),
                float(self._t_hit),
            )

        # first, second, and fourth case
        t_status_int = np.empty((len(t_prev), 2))
        t_status_int[:, 0] = self._inverse_continuous_actor_dynamics(
            t_act_k_at_t=t_act_status_begin,
            t_act_k_minus_one_at_T=t_prev,
            u_act=u,
        )
        t_status_int[:, 1] = t_status_int[:, 0] + duration
        t_status_int[
            np.logical_and(np.isfinite(t_status_int[:, 0]), t_status_int[:, 1] > self._T), 1
        ] = self._T

        # third case
        in_status_mask = np.logical_and(
            t_act_status_begin <= t_prev,
            t_prev < t_act_status_begin + duration,
        )
        t_status_int[in_status_mask, 0] = 0
        t_status_int[in_status_mask, 1] = (
            t_act_status_begin + duration - t_prev[in_status_mask]
        )
        return t_status_int
