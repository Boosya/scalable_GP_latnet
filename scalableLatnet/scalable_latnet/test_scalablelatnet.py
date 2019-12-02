import unittest
from scalablelatnet import get_t, get_z, get_norm
import numpy as np
import tensorflow as tf


class ScalableGPTest(unittest.TestCase):
    sess = tf.InteractiveSession()

    def test_get_t(self):
        t = get_t(2, float)
        assert np.array_equal(t.eval(), np.array([[-1.], [1.]], np.int32))

    def test_get_z(self):
        # testing using 1 everywhere
        # thus z = sin 1 + cos 1 = 1.3817732
        n_mc = 5
        n_signals = 2
        n_rf = 1
        n_nodes = 3
        dim = 1
        t = np.array([[1.], [1.]], np.float32)
        some_omega = tf.ones((n_mc, n_rf, dim), dtype=float)
        some_gamma = tf.ones((n_mc, n_nodes, 2 * n_rf), dtype=float)
        log_variance = tf.log(tf.constant(1, dtype=float))
        z = get_z(n_signals=n_signals, n_mc=n_mc, n_rf=n_rf, dim=dim, g=some_gamma, o=some_omega,
                  log_variance=log_variance, t=t, n_nodes=3)
        result = np.array([[[1.3817732, 1.3817732], [1.3817732, 1.3817732], [1.3817732, 1.3817732]],
                           [[1.3817732, 1.3817732], [1.3817732, 1.3817732], [1.3817732, 1.3817732]],
                           [[1.3817732, 1.3817732], [1.3817732, 1.3817732], [1.3817732, 1.3817732]],
                           [[1.3817732, 1.3817732], [1.3817732, 1.3817732], [1.3817732, 1.3817732]],
                           [[1.3817732, 1.3817732], [1.3817732, 1.3817732], [1.3817732, 1.3817732]]], np.float32)
        assert np.array_equal(z.eval(), result)

    def test_get_norm(self):
        # testing using 1
        # norm = sqrt(1^2 + 1^2) = 1.4142135
        real_y = np.array([[2.], [2.]], np.float32)
        exp_y = np.array([[1.], [1.]], np.float32)
        norm = get_norm(real_y=real_y, exp_y=exp_y)
        assert np.array_equal(norm.eval(), np.array(1.4142135, np.float32))


if __name__ == '__main__':
    unittest.main()
