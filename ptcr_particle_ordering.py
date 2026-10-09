from abc import ABC, abstractmethod
from inspect import isabstract

import numpy as np

from scipy.spatial import distance_matrix


class AbstractParticleOrderingScheme(ABC):
    """Abstract class for particle ordering schemes.

    Particle ordering scheme define the order in which particles are considered in the search of the PTCR controller.  
    Particles being on top of the order/search tree are considered first, and thus they dominate the search. Note that 
    in particular, only the first N_R particles are considered for the search tree while the remaining particles are 
    assigned by a heuristic.
    """

    particle_ordering_scheme_registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            if not hasattr(cls, 'name'):
                raise TypeError(
                    f"Can't instantiate class {cls.__name__} without the 'name' attribute defined.")
            cls.particle_ordering_scheme_registry[getattr(cls, 'name')] = cls

    def __init__(self, **kwargs):
        """
        Initializes the particle ordering scheme.

        :param kwargs: Unused keyword arguments (accepted for subclass compatibility).
        """
        self.sort_order = None

    def __call__(self, **kwargs):
        """
        Sorts the particles and stores the result in ``self.sort_order``.

        :param kwargs: Keyword arguments forwarded to :meth:`sort` (scheme-dependent).

        :returns: An integer np.array of shape ``[num_particles]``, the indices for
            sortation (same as ``self.sort_order``).
        """
        self.sort_order = self.sort(**kwargs)
        return self.sort_order

    @abstractmethod
    def sort(self, **kwargs):
        """
        Sorts the particles based on the provided criteria.

        :param kwargs: Scheme-dependent keyword arguments (particle ids, states, …).

        :returns: An integer np.array of shape ``[num_particles]``, the indices for
            sortation.
        """
        raise NotImplementedError('Call to abstract method.')


class ParticleOrderingById(AbstractParticleOrderingScheme):
    """Sorts the particles by their IDs (ascending).

    """

    name = 'id'

    def sort(self, particle_id, **kwargs):
        """
        Sorts the particles by their IDs (ascending).

        :param particle_id: An integer np.array of shape ``[num_particles]``, the ids
            of the particles.
        :param kwargs: Unused extra keyword arguments.

        :returns: An integer np.array of shape ``[num_particles]``, the indices for
            sortation.
        """
        return np.argsort(particle_id)


class ParticleOrderingByLeavingTime(AbstractParticleOrderingScheme):
    """Sort particles by the time they reach the end of the actor array.

    """

    name = 'leaving_time'

    def __init__(self, array_end, **kwargs, ):
        """
        Initializes the leaving-time particle ordering scheme.

        :param array_end: A float, the end coordinate of the actuator array.
        :param kwargs: Unused extra keyword arguments.
        """
        if not np.isscalar(array_end) or array_end <= 0:
            raise ValueError("The end coordinate of the actuator array array_end must be a positive scalar.")

        super().__init__(**kwargs)
        self._array_end = float(array_end)

    def sort(self, particle_states, **kwargs):
        """
        Sorts particles by the time they reach the end of the actor array.

        Format ``particle_states``:

            [estimated_motion_state_mean, estimated_motion_state_cov, track_score]

        with ``estimated_motion_state_mean`` a np.array of shape ``[num_particles, 2 or 4]``.

        :param particle_states: A tuple of length ``>= 1`` as above.
        :param kwargs: Unused extra keyword arguments.

        :returns: An integer np.array of shape ``[num_particles]``, the indices for
            sortation (earliest leaving time first).
        """
        x_positions = particle_states[0][:, 0]
        x_speeds = particle_states[0][:, 1]

        # determine how long the particles still need to travel to reach the end of the array
        delta_pos = np.subtract(self._array_end, x_positions, dtype=np.float32)
        # determine the time it takes to reach the end of the array
        times_to_end = np.divide(delta_pos, x_speeds, dtype=np.float32)
        return np.argsort(times_to_end)


class ParticleOrderingByCurrentPosition(AbstractParticleOrderingScheme):
    """Sorts particles by their current position, with particles further in the array being sorted first.

    """

    name = 'current_position'

    def sort(self, particle_states, **kwargs):
        """
        Sorts particles by current streamwise position (furthest first).

        Format ``particle_states``:

            [estimated_motion_state_mean, estimated_motion_state_cov, track_score]

        :param particle_states: A tuple of length ``>= 1`` as above.
        :param kwargs: Unused extra keyword arguments.

        :returns: An integer np.array of shape ``[num_particles]``, the indices for
            sortation.
        """
        x_positions = particle_states[0][:, 0]
        # sort by the particle positions in reverse order, eject the particles furthest in the array first
        return np.argsort(-x_positions)


class ParticleOrderingByProximityToKeepParticle(AbstractParticleOrderingScheme):
    """Sorts the particles by their distance to the next particles to keep.

    """

    name = 'proximity_to_keep'

    def sort(self, particle_states, particle_class, **kwargs):
        """
        Sorts eject particles by distance to the nearest keep particle.

        Format ``particle_states``:

            [estimated_motion_state_mean, estimated_motion_state_cov, track_score]

        :param particle_states: A tuple of length ``>= 1`` as above.
        :param particle_class: An integer np.array of shape ``[num_particles]`` with
            ``0`` (keep) or ``1`` (eject).
        :param kwargs: Unused extra keyword arguments.

        :returns: An integer np.array of shape ``[num_particles]``, eject particles
            ordered by ascending distance to the nearest keep particle, followed by
            keep particles.
        """
        positions = particle_states[0]
        keep_indices = np.nonzero(particle_class == 0)[0]
        reject_indices = np.nonzero(particle_class == 1)[0]

        if len(reject_indices) == 0:
            # if there are no eject particles, we are done
            return np.arange(len(particle_class), dtype=int)
        if len(keep_indices) == 0:
            # using sorting by position as fallback if there are no keep particles
            return np.argsort(-positions[:, 0])

        # Calculate minimal distance for each keep particle to any non-keep particle
        dist_matrix = distance_matrix(positions[reject_indices], positions[keep_indices])
        # shape [num_reject, num_keep]
        min_distances = np.min(dist_matrix, axis=1)

        # Sort keep particles by their minimal distance to non-keep particles (ascending)
        sorted_reject_indices = reject_indices[np.argsort(min_distances)]
        return np.concatenate([sorted_reject_indices, keep_indices]).flatten()
