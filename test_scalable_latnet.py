import unittest
from scalable_latnet.scalablelatnet import ScalableLatnet
from flags import Flags
import numpy as np
import tensorflow as tf


class ScalableGPTest(unittest.TestCase):

    def test_get_z(self):
        # testing using 1 everywhere
        # thus z = sin 1 + cos 1 = 1.3817732906760363
        T = 2
        N = 3
        new_flags = {'n_mc': 5, 'n_rff': 1, 'd': 1}
        myflags = Flags(T, None, None, None)
        myflags.set_flags(new_flags)
        t = np.matrix([[1], [1]])
        y = np.matrix([[1, 1, 1], [1, 1, 1]])
        sess = tf.InteractiveSession()
        self.myLatnet = ScalableLatnet(flags=myflags, s=None, d=1, t=t, y=y, logger=None, session=sess)
        t_tf = tf.cast(t, self.myLatnet.FLOAT)
        some_omega = tf.ones((myflags.get_flag('n_mc'), myflags.get_flag('n_rff'), myflags.get_flag('d')),
                             dtype=self.myLatnet.FLOAT)
        some_gamma = tf.ones((myflags.get_flag('n_mc'), N, 2 * myflags.get_flag('n_rff')), dtype=self.myLatnet.FLOAT)
        z = self.myLatnet.get_z(gamma=some_gamma, omega=some_omega, t=t_tf,
                                log_variance=tf.log(tf.constant(1, dtype=self.myLatnet.FLOAT)))
        assert z.shape[0] == myflags.get_flag('n_mc')
        assert z.shape[1] == N
        assert z.shape[2] == T
        assert tf.reduce_max(z).eval() == 1.3817732906760363
        assert tf.reduce_min(z).eval() == 1.3817732906760363


if __name__ == '__main__':
    unittest.main()
