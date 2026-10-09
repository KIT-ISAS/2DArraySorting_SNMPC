from abc import ABC, abstractmethod

import numpy as np

from actor_model import AbstractActorModel


class AbstractContactModel(ABC):
    """Abstract class for all contact models of the controller (either for 1D or 2D optical sorters).

    Along with the particle and the actor model, the contact model forms the system model of the stochastic model
    predictive controller.

    Contact model:

    The contact model describes P(ej_k^i = 1 | x_k^{i, motion}, t_k^act, u_k) respectively
    P(ej_k^i = 1 | t_k^act, u_k), i.e., the conditional and unconditional ejection probability of particle i, which
    additionally depends deterministically on the actors and the control input.

    Since a particle can eject an actor only if it was not ejected by an arbitrary other actor (we will later see that
    this holds no matter if the particle passed the other actor previously or will pass it in future), the distribution
    of ej_k^i can be calculated with the help of the additional binary random variable eu_j^i in {0, 1} (omitting time
    indices k for brevity) with eu_j^i = 0 ≙ “particle 𝑖 not ejected until actor 𝑗 (including actor j)” that describes
    whether a particle i was not ejected by all actors up to j.

    The evolution of eu^i_j is given by the actor-variant Markov chain

          [ P(eu_{j+1}^i = 0) ]  =  [ 1 - P(ej_{j+1}^i = 1 | t_{j+1}^act, u_{j+1} )   0 ]  *  [ P(eu_j^i = 0) ]
          [ P(eu_{j+1}^i = 1) ]     [ P(ej_{j+1}^i = 1 | t_{j+1}^act, u_{j+1} )       1 ]     [ P(eu_j^i = 1) ] ,

    with initial value [ P(eu_0^i = 0) , P(eu_0^i = 1) ]^T = [1, 0]^T and P(ej_j^i = 1 | t_{j+1}^act, u_{j+1} )
    being the probability that particle i is ejected at actor j. Then

        P(ej^i = 1 | t^act, u) = P(eu^i_N_A = 1)  ,

    with N_A being the number of actors in the setup.

    Note that in particular,

        P(ej^i = 1 | t^act, u) = 1 - Prod_{j=1}^{N_A} ( 1 - P(ej_j^i = 1 | t_j^act, u_j ) )

    as clearly visible from the first row of the Markov chain and by the initial condition. In other words, the
    probability that a particle is ejected is one minus the probability that it is not ejected by any actor.
    Since the product is associative, the order of the actors does not matter.

    Note that when P(ej_j^i = 1 | t_j^act, u_j) ≈ 0 for at most one actor j, then

        P(ej_j^i = 1 | t^act, u) ≈ Sum_{j=1}^N_A P(ej_j^i = 1 | t_j^act, u_j)  ,

    for which it is easier to calculate gradients and Hessians.

    There exist different versions on the definition of P(ej_j^i = 1 | t_j^act, u_j) depending on whether the model is
    formulated in discrete or continuous time and whether we consider 1D or 2D optical sorters.
    """

    def __init__(self,
                 actor_model,
                 # p_eject=np.array([0.0, 0.9, 0.5, 0.3]),  # this may be more realistic, but results in a controller
                 # that sometimes tries to eject two particle with one actor by starting the actors a time step later
                 # (reducing only p_eject,down should solve this problem).
                 p_eject=np.array([0.0, 0.9, 0.2, 0.1]),
                 use_sum_approximation=False,
                 seed=None,
                 ):
        """Initializes the contact model.

        :param actor_model: A child instance of AbstractActorModel, the actor model to be used to calculate the contacts.
        :param p_eject: A np.array of shape [4] or [2, 4], the probabilities of ejecting the particle when the actor
            status is [not in {HIT, UP, DOWN} , HIT, UP, DOWN], i.e., the parameters
            [0, p_eject, p_eject,up, p_eject,down]. Different probabilities can be used for particles to be ejected
            and particles to be kept by specifying a 2D array.
        :param use_sum_approximation: A Boolean, whether to approximate the actor-dependent Markov chain with a sum of
            the ejection probabilities of all actors, i.e., assume independence of the actor ejections.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        """
        if not isinstance(actor_model, AbstractActorModel):
            raise ValueError('actor_model must be a child instance of AbstractActorModel.')
        if np.asarray(p_eject).shape != (4,) and np.asarray(p_eject).shape != (2, 4):
            raise ValueError(
                'p_eject must be either a np.array of shape [4] or of shape [2, 4].')
        if np.any(np.asarray(p_eject) < 0) or np.any(np.asarray(p_eject) > 1):
            raise ValueError(
                'p_eject must be a valid probabilities, i.e., its components must be within [0, 1].')

        self._actor_model = actor_model
        self._hit_success_probs = np.asarray(p_eject).T if np.asarray(p_eject).ndim == 2 else np.asarray(
            (p_eject, p_eject)).T

        self._calculate_ejection_probability = self._calculate_ejection_probability_by_sum if \
            use_sum_approximation else self._calculate_ejection_probability_by_product

        # set the seed and create the random number generator
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def predict_ejection(self, motion_state_mean, motion_state_cov, particle_class, *args):
        """Calculates the (unconditioned) ejection probability P(ej_k^i = 1 | t_k^act, u_k) for all particles.

        Note that the ejection probability is not conditioned on the motion state x_k^{i, motion} of the particles,
        i.e., a marginalization over the given motion state distribution is performed.

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        This function may support a batch shape, i.e., some subclasses may allow the first dimension of the input
        arrays to be a batch or sample size.

        :param motion_state_mean: A np.array of shape [num_samples, num_particles, motion_state_length] or
            [num_particles, motion_state_length], the mean of x_k^motion.
        :param motion_state_cov: A np.array of shape [num_samples, num_particles, motion_state_length,
            motion_state_length] or [num_particles, motion_state_length, motion_state_length], the covariance
            of x_k^motion.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param args: Additional arguments that are required to calculate the ejection probabilities. See the subclasses
            for more information.

        :returns:
            p_eject: A np.array of shape [num_samples, num_particles] or [num_particles] containing the probabilities
                P(ej_k^i = 1 | x_k^{i, motion}) for each particle at the current time step.
        """
        # To be overwritten by subclass
        raise NotImplementedError('Call to abstract method.')

    def conditional_predict(self, sampled_motion_state, particle_class, *args):
        """Calculates the (conditional) ejection probability P(ej_k^i = 1 | x_k^{i, motion}, t_k^act, u_k) for all
        particles.

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        This function may support a batch shape, i.e., some subclasses may allow the first dimension of the input
        arrays to be a batch or sample size.

        :param sampled_motion_state: A np.array of shape [num_samples, num_particles, motion_state_length] or
            [num_particles, motion_state_length], samples of x_k^motion.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param args: Additional arguments that are required for sampling. See the subclasses for more information.

        :returns:
            p_eject: A np.array of shape [num_particles, num_particles] or [num_particles] containing the probabilities
                P(ej_k^i = 1 | x_k^{i, motion}, t_k^act, u_k) for each particle at the current time step.
        """
        motion_state_cov = np.zeros((*sampled_motion_state.shape, sampled_motion_state.shape[-1]))
        return self.predict_ejection(sampled_motion_state, motion_state_cov, particle_class, *args)

    def sample_ejection(self, sampled_motion_state, particle_class, *args):
        """Samples from the contact model to determine whether a particle is ejected by an actor, i.e., samples from
        the (conditional) distribution p(ej_k^i | x_k^{i, motion}, t_k^act, u_k).

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        This function may support a batch shape, i.e., some subclasses may allow the first dimension of the input
        arrays to be a batch or sample size.

        :param sampled_motion_state: A np.array of shape [num_samples, num_particles, motion_state_length] or
            [num_particles, motion_state_length], samples of x_k^motion.
        :param particle_class: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes.
        :param args: Additional arguments that are required for sampling. See the subclasses for more information.

        :returns: An integer np.array of shape [num_samples, num_particles] or [num_particles] with components either
            0 or 1., the samples of the ejection probabilities.
        """
        p_ejects = self.conditional_predict(sampled_motion_state, particle_class, *args)
        return self._rng.binomial(1, p=p_ejects, size=sampled_motion_state.shape[:-1])

    @staticmethod
    def _calculate_ejection_probability_by_product(p_eject_i_at_j):
        """Calculates the (true) ejection probabilities by evaluating the Markov chain of passed actors.

        This function supports a batch shape, i.e., the first dimension of the input arrays can be a batch or sample
        size.

        :param p_eject_i_at_j: A np.array of shape [num_samples, num_actors, num_particles] or [num_actors,
            num_particles], the probabilities that particle i is ejected at actor j, regardless of if it is ejected
            by another actor.

        :returns:
            p_eject: A np.array of shape [num_samples, num_particles] or [num_particles] containing the
                probabilities P(ej_k^i = 1 | t_k^act, u_k, ...) for each particle at the current time step.
        """
        return 1 - np.prod(1 - p_eject_i_at_j, axis=-2)

    @staticmethod
    def _calculate_ejection_probability_by_sum(p_eject_i_at_j):
        """Calculates approximate ejection probabilities by assuming that the conditional dependencies between the
        ejection probabilities on different actors can be neglected.

        This function supports a batch shape, i.e., the first dimension of the input arrays can be a batch or sample
        size.

        :param p_eject_i_at_j: A np.array of shape [num_samples, num_actors, num_particles] or [num_actors,
            num_particles], the probabilities that particle i is ejected at actor j, regardless of if it is ejected
            by another actor. 

        :returns:
            p_eject: A np.array of shape [num_samples, num_particles] or [num_particles] containing the
                probabilities P(ej_k^i = 1 | t_k^act, u_k, ...) for each particle at the current time step.
        """
        p_eject = np.sum(p_eject_i_at_j, axis=-2)
        p_eject[p_eject > 1] = 1  # restrict it to 1 because sum does not give the true probability
        return p_eject

