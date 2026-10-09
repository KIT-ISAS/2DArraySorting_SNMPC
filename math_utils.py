import numpy as np


def batch_matvec(A, b):
    """A batched matrix-vector multiplication.

    :param A: A np.array of shape [batch_size, n, m] or [n, m], the batch of matrices.
    :param b: A np.array of shape [batch_size, m], the batch of vectors.

    :returns: A np.array of shape [batch_size, n], the batch of matrix-vector products.
    """
    # return tf.linalg.matvec(A, b).numpy()  # slowest
    # if A.ndim == 2:
    #     return np.einsum('Ni,Bi ->BN', A, b)
    # return np.einsum('BNi,Bi ->BN', A, b)
    return np.matmul(A, b[:, :, None]).squeeze(-1)  # fastest
