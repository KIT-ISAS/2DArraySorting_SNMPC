import os
import sys
import inspect

# import parent directory (see https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np

import tensorflow as tf

from math_utils import batch_matvec


class MathUtilsTest(tf.test.TestCase):
    """Test cases for functions in math_utils.py"""

    def test_batch_matvec(self):
        A1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        A2 = np.array([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
        A = np.stack([A1, A2], axis=0)  # shape [2, 3, 4]

        b1 = np.array([1, 2, 3, 4])
        b2 = np.array([5, 6, 7, 8])
        b = np.stack([b1, b2], axis=0)  # shape [2, 4]

        c = batch_matvec(A, b)
        self.assertAllEqual((2, 3), c.shape)
        cc = np.empty((2, 3))
        for i, bi in enumerate(b):
            cc[i] = A[i] @ bi
        self.assertAllClose(cc, c)

        c = batch_matvec(A1, b)
        self.assertAllEqual((2, 3), c.shape)
        cc = np.empty((2, 3))
        for i, bi in enumerate(b):
            cc[i] = A1 @ bi
        self.assertAllClose(cc, c)


if __name__ == "__main__":
    tf.test.main()
