import numpy as np

from scipy.stats import multivariate_normal as mvn

from math_utils import batch_matvec


class ParticleModel:
    """General particle system and measurement model for the controller (either for 1D or 2D optical sorters).

    I: System Model:

    Along with the actor and the contact model, the particle model forms the system model of the stochastic model
    predictive controller.

    Models the particles'

        - motion,
            via the (random variable) motion state x_k^motion (defined on the real plane, k being the time step),

        - existence,
            via the binary existence random variable ex_k^i in {0, 1}, with ex_k^i = 1 ≙ “particle 𝑖 not yet ejected”,
            and

        - class
            via the (deterministic) binary class variable c^i in {acc, rej}, with c^i = rej ≙ “particle 𝑖 should
            be removed from the particle stream (reject particle i)”. Note that c_i is a storage variable only that does
            not change with time.

    It is assumed that all particle are independent and that a particle i moves independent of existence and class,
    i.e., the distribution of the particles' state is given by

        p(x_k) = Prod_{i=1}^{N_P} p(x_k^i) ,

            with

                x_k = [x_k^1 ... x_k^{N_P}]^T   (N_P being the number of particles) ,

                x_k^i = [x_k^{i,motion} ex_k^i]^T ,

    and its transition is given by

        p(x_{k+1}^motion , ex_{k+1} | x_k^motion , ex_k+1)
            = p(x_{k+1}^motion | x_k^motion) p( ex_{k+1} | ex_k , x_k^motion) ,

            with

                x_k^motion = [x_k^{1, motion} ... x_k^{N_P, motion}]^T ,

                ex_k = [ex^1 ... ex^{N_P}]^T .

    p(x_{k+1}^motion | x_k^motion) constitutes the motion model, whereas p( ex_{k+1} | ex_k , x_k^motion), the existence
    model, depends on the contact and actor model.

    Motion model:

        We assume time-invariant constant-velocity (CV) motion behavior for all particles here, i.e., the motion state
        x_k^{i,motion} has format [x, v_x] for 1D motion, or [x, v_x, y, v_y] for 2D motion, respectively, with x and y
        being the position and v_x and v_y being the velocity in x and y direction, respectively.

    Existence model:

        We model the existence with a time-variant hidden Markov chain

          [ P(ex_{k+1}^i = 0) ]  =  [1     P(ej_k^i = 1 | x_k^{i, motion})     ]  *  [ P(ex_k^i = 0) ]
          [ P(ex_{k+1}^i = 1) ]     [0     1 - P(ej_k^i = 1 | x_k^{i, motion}) ]     [ P(ex_k^i = 1) ] ,

          where the binary ejection random variable ej_k^i in {0, 1} with ≙ “particle 𝑖 ejected at time step 𝑘“ is given
          by the contact and actor model.



    II: Measurement Model:

        Assumptions for 𝑛 > 0 (excluding 𝑛 = 0 !), i.e., for the control horizon:

            i)   In each time step and for each particle i, we get exactly one measurement for the position z_k^{i, pos}
                    and exactly one measurement for the existence of the particle z_k^{i, ex}.
                    -> There are no missing measurements and no multiple detections.
            ii)  There is no clutter (measurements without a particle).
            iii) The correspondence between measurements and particles is known
            iv)  There are no track births

            i) - iii) imply that there is no need to solve a data association problem

    Position measurement:

        Only the position component of a particles motion state x_k^{i, motion} is observable. The measurement noise is
        additive and time-invariant with power spectral density S_v.

    Existence measurement:

        The measurement equation is

          [ P(z_k^{i, ex} = 0) ]  =  [1 - p_clutter   1 - p_detect ]  *  [ P(ex_k^i = 0) ]
          [ P(z_k^{i, ex} = 1) ]     [p_clutter         p_detect   ]     [ P(ex_k^i = 1) ] ,

         where p_detect is the probability of correctly detecting an existing particle and p_clutter is the probability
         of detecting a non-existing particle as existing (both time-invariant).
    """

    def __init__(self,
                 T,
                 p_detect=0.95,
                 p_clutter=0.2,
                 S_w=(0.0001, 0.0001),
                 S_v=(0.01, 0.01),
                 motion_model_type='CV',
                 boundaries=None,
                 seed=None,
                 ):
        """Initializes the particle model.

        :param T: A float representing the time interval between two consecutive time steps.
        :param p_detect: A float in [0, 1] representing the probability of (correctly) detecting an existing particle as
            existing.
        :param p_clutter: A float in [0, 1] representing the probability of detecting a non-existing particle as
            existing (clutter). May also be used to avoid overly confident existence states in cas of CLF control, i.e.,
            to avoid that virtual existence measurements influences the existence state distribution too much.
        :param S_w: A float or a tuple of two floats representing the power spectral density of the motion model's
            system noise. If a float, an 1D optical sorter (groove sorter) is assumed. If a tuple, a 2D optical sorter
            (belt or chute) is assumed and the values represent the power spectral densities in x- and y-direction,
            respectively.
        :param S_v: A float or a tuple of two floats representing the power spectral density of the motion model's
            measurement noise. Must be of same type as S_w. If a float, an 1D optical sorter (groove sorter) is assumed.
            If a tuple, a 2D optical sorter (belt or chute) is assumed and the values represent the power spectral
            densities in x- and y-direction, respectively.
        :param motion_model_type: A string, either 'CV' or 'CA', representing the type of motion model to be used
            (either constant velocity (CV) or constant acceleration (CA)).
        :param boundaries: None or a tuple of two floats (lower_boundary, upper_boundary) representing the boundaries
            in y direction in case of a chute/belt sorter. If given, a simple wall collision model is applied in the
            predict_motion step.
        :param seed: None or an integer, the seed for the random number generator. Fix the seed for reproducibility.
        """
        if not np.isscalar(T) or T <= 0:
            raise ValueError('The time interval between two consecutive time steps T must be a positive scalar.')
        if p_detect < 0 or p_detect > 1:
            raise ValueError('p_detect must be a valid probability, i.e., 0 <= p_detect <= 1.')
        if p_clutter < 0 or p_clutter > 1:
            raise ValueError('p_clutter must be a valid probability, i.e., 0 <= p_clutter <= 1.')
        if not ((np.isscalar(S_w) or len(np.asarray(S_w)) == 2) and (
                np.isscalar(S_v) or len(np.asarray(S_w)) == 2)) or type(S_w) is not type(S_v):
            raise ValueError(
                'The power spectral densities S_w and S_v must be both either scalars or tuples of two scalars.')
        if np.any(np.asarray(S_w) < 0):
            raise ValueError('The power spectral density S_w must be greater than or equal to zero.')
        if np.any(np.asarray(S_v) < 0):
            raise ValueError('The power spectral density S_v must be greater than or equal to zero.')
        if not motion_model_type in ['CV', 'CA']:
            raise ValueError("motion_model_type must be either 'CV' or 'CA'.")
        if boundaries is not None and np.isscalar(S_w):
            raise ValueError('Boundaries can only be applied for 2D optical sorters (belt or chute).')
        if boundaries is not None and (np.asarray(boundaries).shape != (2,) or boundaries[0] >= boundaries[1]):
            raise ValueError('Boundaries must be given as a tuple (lower_boundary, upper_boundary) with '
                             'lower_boundary < upper_boundary.')

        # define the motion model (CV model, motion state [x, v_x] or [x, v_x, y, v_y])
        if np.isscalar(S_w):
            # 1D optical sorter (groove sorter)

            if motion_model_type == 'CV':
                self._A = np.array([[1, T],
                                    [0, 1]], dtype=float)
                self._C_w = S_w * np.array([[T ** 3 / 3, T ** 2 / 2],
                                            [T ** 2 / 2, T]], dtype=float)
                self._H = np.array([[1, 0]], dtype=float)
            else:
                self._A = np.array([[1, T, T ** 2 / 2, ],
                                    [0, 1, T, ],
                                    [0, 0, 1]], dtype=float)
                self._C_w = S_w * np.array(
                    [[T ** 5 / 20, T ** 4 / 8, T ** 3 / 6],
                     [T ** 4 / 8, T ** 3 / 3, T ** 2 / 2],
                     [T ** 3 / 6, T ** 2 / 2, T]], dtype=float)
                self._H = np.array([[1, 0, 0, ]], dtype=float)

            self._C_v = S_v * np.array([[1]], dtype=float)

        else:
            # 2D optical sorter (belt or chute)

            if motion_model_type == 'CV':

                self._A = np.array([[1, T, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, T],
                                    [0, 0, 0, 1]], dtype=float)
                self._C_w = np.diag([S_w[0], S_w[0], S_w[1], S_w[1]]) @ np.array([[T ** 3 / 3, T ** 2 / 2, 0, 0],
                                                                                  [T ** 2 / 2, T, 0, 0],
                                                                                  [0, 0, T ** 3 / 3, T ** 2 / 2],
                                                                                  [0, 0, T ** 2 / 2, T]], dtype=float)
                self._H = np.array([[1, 0, 0, 0],
                                    [0, 0, 1, 0]], dtype=float)
            else:
                self._A = np.array([[1, T, T ** 2 / 2, 0, 0, 0],
                                    [0, 1, T, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, T, T ** 2 / 2],
                                    [0, 0, 0, 0, 1, T],
                                    [0, 0, 0, 0, 0, 1]],
                                   dtype=float)
                self._C_w = np.diag([S_w[0], S_w[0], S_w[0], S_w[1], S_w[1], S_w[1]]) @ np.array(
                    [[T ** 5 / 20, T ** 4 / 8, T ** 3 / 6, 0, 0, 0],
                     [T ** 4 / 8, T ** 3 / 3, T ** 2 / 2, 0, 0, 0],
                     [T ** 3 / 6, T ** 2 / 2, T, 0, 0, 0],
                     [0, 0, 0, T ** 5 / 20, T ** 4 / 8, T ** 3 / 6],
                     [0, 0, 0, T ** 4 / 8, T ** 3 / 3, T ** 2 / 2],
                     [0, 0, 0, T ** 3 / 6, T ** 2 / 2, T]], dtype=float)
                self._H = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0]], dtype=float)

            self._C_v = np.diag([S_v[0], S_v[1]]) @ np.eye(2, dtype=float)

        self._S_w = S_w

        # define the existence measurement model, state [P(ex = 0), P(ex = 1)] , i.e., [P(exist not), P(exist)]
        self._H_exists = np.array([[1 - p_clutter, 1 - p_detect],
                                   [p_clutter, p_detect]], dtype=float)  # time invariant

        self._boundaries = np.asarray(boundaries, dtype=float) if boundaries is not None else None

        # for properties
        self._particle_class = None

        # set the seed and create the random number generator
        self._rng = np.random.default_rng(seed)

    @property
    def particle_class(self):
        """The classes of the particles.

        :returns: An integer np.array of shape [num_particles] where each component is either 0 ("keep the particle") or
            1 ("eject the particle") representing the particle classes (the class storage variable of the particle
            model).
        """
        return self._particle_class

    @particle_class.setter
    def particle_class(self, value):
        """Sets the classes of the particles.

        :param value: An integer np.array of shape [num_particles] where each component is either 0 ("keep the
            particle") or 1 ("eject the particle") representing the particle classes (the class storage variable of the
            particle model).
        """
        if not np.all(np.logical_or(value == 0, value == 1)):
            raise ValueError('particle_class must be an binary variable consisting only of zeros or ones.')
        self._particle_class = np.asarray(value, dtype=int)  # shape [num_particles], 0 keep, 1 eject

    @property
    def motion_state_length(self):
        """The length (dimension) of the motion state vector.

        :returns: An integer, the length (dimension) of the motion state vector.
        """
        return len(self._A)

    @property
    def H(self):
        """The measurement matrix of the motion model.

        :returns: A np.array of shape [motion_state_length, motion_state_length], the measurement matrix of the motion
            model.
        """
        return self._H

    @property
    def S_w(self):
        """The power spectral density of the motion model's system noise.

        :returns: A float or a tuple of two floats representing the power spectral density of the motion model's system
            noise.
        """
        return self._S_w

    @property
    def C_v(self):
        """The measurement noise covariance of the motion model.

        :returns: A np.array of shape [1, 1] or [2, 2], the measurement noise covariance of the motion model.
        """
        return self._C_v

    @property
    def H_exits(self):
        """The measurement matrix of the existence model.

        The existence state is defined as [P(ex = 0), P(ex = 1)] , i.e., [P(exist not), P(exist)].

        :returns: A np.array of shape [2, 2], the measurement matrix of the existence model.
        """
        return self._H_exists

    @staticmethod
    def A_exists(p_eject):
        """The transition matrix of the existence model.

        Note that the transition matrix is time-variant and depends on the ejection probability p_eject.

        The existence state is defined as [P(ex = 0), P(ex = 1)] , i.e., [P(exist not), P(exist)].

        :param p_eject: A float in [0, 1], the probability of ejecting a particle.

        :returns: A np.array of shape [2, 2], the transition matrix of the existence model.
        """
        return np.array([[1, p_eject],
                         [0, 1 - p_eject]])

    def predict_motion(self, motion_state_mean, motion_state_cov):
        """Calculates the (unconditional) next motion state distribution p( x_{k+1}^motion ) from p( x_k^motion ) using
        a linear Gaussian state space model as transition model.

        Note that p( x_{k+1}^motion ) is not conditioned on the motion state x_k^motion, i.e., a marginalization
        over the given motion state distribution p( x_k^motion ) is performed.

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        :param motion_state_mean: A np.array of shape [num_particles, motion_state_length], the mean of x_k^motion.
        :param motion_state_cov: A np.array of shape [num_particles, motion_state_length, motion_state_length], the
            covariance of x_k^motion.

        :returns:
            motion_state_mean: A np.array of shape [num_particles, motion_state_length], the mean of
                x_{k+1}^motion given x_k^motion.
            motion_state_cov: A np.array of shape [num_particles, motion_state_length, motion_state_length], the
                covariance of x_{k+1}^motion given x_k^motion.
        """
        mu = np.asarray(motion_state_mean, dtype=float)
        P = np.asarray(motion_state_cov, dtype=float)

        try:
            from adf.adf_numba_kernels import numba_enabled, predict_motion_numba
        except ImportError:
            numba_enabled = lambda: False  # noqa: E731
            predict_motion_numba = None

        if numba_enabled() and predict_motion_numba is not None and mu.ndim == 2:
            apply_walls = self._boundaries is not None
            y_ind = int(mu.shape[1] / 2) if apply_walls else 0
            y_lo = float(self._boundaries[0]) if apply_walls else 0.0
            y_hi = float(self._boundaries[1]) if apply_walls else 1.0
            return predict_motion_numba(
                np.ascontiguousarray(mu, dtype=np.float64),
                np.ascontiguousarray(P, dtype=np.float64),
                np.ascontiguousarray(self._A, dtype=np.float64),
                np.ascontiguousarray(self._C_w, dtype=np.float64),
                apply_walls,
                y_lo,
                y_hi,
                y_ind,
            )

        motion_state_mean = batch_matvec(self._A, mu)
        motion_state_cov = np.matmul(np.matmul(self._A, P), self._A.T) + self._C_w[None, :, :]

        if self._boundaries is not None:
            # apply the wall collision model to the mean of the motion state distribution.
            motion_state_mean = self._wall_collisions_model(motion_state_mean)

        return motion_state_mean, motion_state_cov

    def _wall_collisions_model(self, motion_state_mean):
        """Simple collision model for collisions with walls orientated along the transport direction (x).

        The model assumes reflection and energy preservation.

         Format state:

            [x, v_x, ... , y, v_y, ...]
        """
        # TODO: Note that the model does not consider the particle extent. It therefore also deviates from the one in
        #  simulator.py
        y_ind = int(motion_state_mean.shape[1] / 2)
        y_pred = motion_state_mean[:, y_ind]

        beyond_upper_mask = y_pred > self._boundaries[1]
        motion_state_mean[beyond_upper_mask, y_ind] = 2 * self._boundaries[1] - y_pred[beyond_upper_mask]

        below_lower_mask = y_pred < self._boundaries[0]
        motion_state_mean[below_lower_mask, y_ind] = 2 * self._boundaries[0] - y_pred[below_lower_mask]

        return motion_state_mean

    @staticmethod
    def predict_existence(existence_probs, p_eject):
        """Calculates the (unconditional) next existence state distribution p( ex_{k+1} ) from p( ex_k , x_k^motion )
        using the transition model

          [ P(ex_{k+1}^i = 0) ]  =  [1     P(ej_k^i = 1 | x_k^{i, motion})     ]  *  [ P(ex_k^i = 0) ]
          [ P(ex_{k+1}^i = 1) ]     [0     1 - P(ej_k^i = 1 | x_k^{i, motion}) ]     [ P(ex_k^i = 1) ]

        for each particle i.

        Note that p( ex_{k+1} ) is not conditioned on ex_k, x_k^motion, i.e., an (approximate) marginalization over
        the given distribution of ex_k and x_k^motion is performed.

        :param existence_probs: A np.array of shape [num_particles, 2], the distribution of ex_k, i.e., the
            probabilities [ P(ex_k = 0) , P(ex_k = 1) ], respectively [ P(exist not), P (exist) ].
        :param p_eject: A np.array of shape [num_particles] with values in [0, 1], the probability
            P(ej_k^i = 1 | x_k^motion ).

        :returns:
            predicted_existence_probs: A np.array of shape [num_particles, 2], the distribution of ex_{k+1}, i.e., the
                probabilities [ P(ex_{k+1} = 0) , P(ex_{k+1} = 1) ], respectively [ P(exist not), P (exist) ] given the
                previous state.
        """
        A_exists = np.ones((*existence_probs.shape, existence_probs.shape[-1]))
        A_exists[:, 1, 0] = 0
        A_exists[:, 0, 1] = p_eject
        A_exists[:, 1, 1] = 1 - p_eject
        return batch_matvec(A_exists, existence_probs)

    def measure_position(self, motion_state_mean, motion_state_cov):
        """Calculates the (unconditional) measurement likelihood p( z_k^pos ) from p( x_k^motion ).

        Note that p( z_k^pos ) is not conditioned on the motion state x_k^motion, i.e., a marginalization over the given
        motion state distribution p( x_k^motion ) is performed.

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        :param motion_state_mean: A np.array of shape [num_particles, motion_state_length], the mean of x_k^motion.
        :param motion_state_cov: A np.array of shape [num_particles, motion_state_length, motion_state_length], the
            covariance of x_k^motion.

        :returns:
            pos_meas_mean: A np.array of shape [num_particles, 1 or 2], the mean of z_k^pos given x_k^motion.
            pos_meas_cov: A np.array of shape [num_particles, 1, 1] or [num_particles, 2, 2], the covariance of
                z_k^pos given x_k^motion.
        """
        pos_meas_mean = batch_matvec(self._H, motion_state_mean)
        pos_meas_cov = np.matmul(np.matmul(self._H, motion_state_cov), self._H.T) + self._C_v[None, :, :]
        return pos_meas_mean, pos_meas_cov

    def measure_existence(self, existence_probs):
        """Calculates the measurement likelihood p( z_k^ex ) from p ( ex_k ) using the measurement equation

          [ P(z_k^{i, ex} = 0) ]  =  [1     1 - p_detect ]  *  [ P(ex_k^i = 0) ]
          [ P(z_k^{i, ex} = 1) ]     [0       p_detect   ]     [ P(ex_k^i = 1) ] ,

        for each particle i.

        Note that p( z_k^ex ) is not conditioned on ex_k, i.e., a marginalization over the given distribution p( ex_k )
        is performed.

        :param existence_probs: A np.array of shape [num_particles, 2], the distribution of ex_k, i.e., the
            probabilities [ P(ex_k = 0) , P(ex_k = 1) ], respectively [ P(exist not), P (exist) ].

        :returns:
            existence_meas_probs: A np.array of shape [num_particles, 2], the distribution of z_k^ex, i.e., the
                probabilities [ P(z_k^ex = 0) , P(z_k^ex = 1) ] given the current state.
        """
        return batch_matvec(self._H_exists, existence_probs)

    def sample_motion_transition(self, sampled_motion_state):
        """Propagates samples of the particle's motion state, i.e., outputs samples from the (unconditional) next motion
        state distribution p( x_{k+1}^motion ) by propagating the original samples from p( x_k^motion ) through the
        transition model p( x_{k+1}^motion | x_k^motion).

        Note that p( x_{k+1}^motion ) is not conditioned on the motion state x_k^motion, i.e., a marginalization
        over the given motion state distribution p( x_k^motion ) is performed.

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        :param sampled_motion_state: A np.array of shape [num_particles, motion_state_length], samples of x_k^motion.

        :returns:
            sampled_motion_state: A np.array of shape [num_particles, motion_state_length], the samples of
                x_{k+1}^motion after the transition.
        """
        sampled_motion_state = batch_matvec(self._A, sampled_motion_state)
        w_t = self._rng.multivariate_normal(mean=np.zeros(sampled_motion_state.shape[1]),
                                            cov=self._C_w,
                                            size=len(sampled_motion_state))
        sampled_motion_state += w_t
        return sampled_motion_state

    def sample_existence_transition(self, sampled_existence_state, sampled_ejections):
        """Propagates samples the particles' existence, i.e, outputs samples from the next existence state distribution
        p( ex_{k+1} ) by propagating the original samples of ex_k, x_k^motion through the transition model
        p( ex_{k+1} | ex_k , x_k^motion).

        Note that p( ex_{k+1} ) is not conditioned on ex_k, x_k^motion, i.e., an (approximate) marginalization over
        the given distribution of ex_k and x_k^motion is performed.

        Note that sampled_existence_state is assumed to be in one-hot encoding, i.e., the elements sum to 1 for each
        particle.

        :param sampled_existence_state: An integer np.array of shape [num_particles, 2] with elements either 0 or 1,
            samples of ex_k. The elements must sum to 1 for each particle.
        :param sampled_ejections: An integer np.array of shape [num_particles] with components either 0 or 1., the
            samples of the ejection probabilities.

        :returns:
            sampled_existence_state: An integer np.array of shape [num_particles, 2] with elements either 0 or 1, the
                samples of ex_{k+1} after the transition. The elements must sum to 1 for each particle.
        """
        return self.predict_existence(sampled_existence_state, sampled_ejections).astype(int)

    def sample_position_measurement(self, sampled_motion_state):
        """Samples the position measurements, i.e., creates a measurement of the position of the particles by
        propagating the original samples from p( x_k^motion ) through the continuous measurement model
        p(z_k^pos | x_k^motion).

        Note that p( z_k^pos ) is not conditioned on the motion state x_k^motion, i.e., a marginalization
        over the given motion state distribution p( x_k^motion ) is performed.

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        :param sampled_motion_state: A np.array of shape [num_particles, motion_state_length], samples of x_k^motion.

        :returns: A np.array of shape [num_particles, 1 or 2], the samples of z_k^pos.
        """
        return batch_matvec(self._H, sampled_motion_state) + self._rng.multivariate_normal(
            np.zeros(self._H.shape[0]),
            cov=self._C_v,
            size=len(sampled_motion_state))

    def sample_existence_measurement(self, sampled_existence_state):
        """Samples the existence measurements, i.e., creates a measurement of the existence of the particles by
        propagating the original samples of ex_k through the binary measurement model p(z_k^ex | ex_k).

        Note that p( z_k^ex ) is not conditioned on ex_k, i.e., a marginalization over the given distribution p( ex_k )
        is performed.

        Note that sampled_existence_state is assumed to be in one-hot encoding, i.e., the elements sum to 1 for each
        particle.

        :param sampled_existence_state: An integer np.array of shape [num_particles, 2] with elements either 0 or 1,
            samples of ex_k. The elements must sum to 1 for each particle.

        :returns: An integer np.array of shape [num_particles] with elements either 0 or 1, the samples of z_k^ex.
        """
        meas_probs = self.measure_existence(sampled_existence_state)
        return self._rng.binomial(1, p=meas_probs[:, 1], size=sampled_existence_state.shape[0])

    def calculate_motion_transition_log_likelihood(self, next_motion_state, motion_state):
        """Calculates the log likelihood of the next motion state distribution p( x_{k+1}^motion | x_k^motion ) given
        the current motion state distribution p( x_k^motion ).

         Format motion_state:

            [position, velocity],
                in case of a 1D optical sorter (groove sorter), or

            [position_x, velocity_x, position_y, velocity_y]
                in case of a 2D optical sorter (grid sorter).

        :param next_motion_state: A np.array of shape [num_particles, motion_state_length], the next value for the
            motion state.
        :param motion_state: A np.array of shape [num_particles, motion_state_length], the current value for the motion
            state.

        :returns:
            log_likelihoods: A np.array of shape [num_particles], the log likelihoods of the motion state distribution
                p( x_{k+1}^motion | x_k^motion ).
        """
        diff = next_motion_state - batch_matvec(self._A, motion_state)
        if np.all(self._C_w == 0):
            # Dirac distribution
            return np.where(np.all(np.isclose(diff, 0.0), axis=1), 0.0, -np.inf)
        return mvn.logpdf(diff, cov=self._C_w)

    @staticmethod
    def calculate_existence_transition_log_likelihood(next_existence, existence, p_eject):
        """Calculates the log likelihood of the next existence state distribution p( ex_{k+1} | ex_k , x_k^motion )
        given the current existence state distribution p( ex_k , x_k^motion ).

        :param next_existence: An integer np.array of shape [num_particles, 2] with elements either 0 or 1, the next
            value for the existence state.
        :param existence: An integer np.array shape [num_particles, 2] with elements either 0 or 1, the current value
            for the existence state.
        :param p_eject: A np.array of shape [num_particles], the ejection probabilities P(ej_k^i = 1 | x_k^motion ).

        :returns:
            log_likelihoods: A np.array of shape [num_particles], the log likelihoods of the existence state
                distribution p( ex_{k+1} | ex_k , x_k^motion ).
        """
        A_exists = np.ones((*existence.shape, existence.shape[-1]))
        A_exists[:, 1, 0] = 0
        A_exists[:, 0, 1] = p_eject
        A_exists[:, 1, 1] = 1 - p_eject
        with np.errstate(divide='ignore'):
            log_l = np.log(A_exists[np.arange(len(existence)), next_existence[:, 1], existence[:, 1]])
        return log_l
