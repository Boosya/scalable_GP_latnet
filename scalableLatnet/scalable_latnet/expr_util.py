import numpy as np
import os
import logging
from io import StringIO as StringBuffer
import io


# from StringIO import StringIO as StringBuffer


class ExprUtil:

    @staticmethod
    def check_dir_exists(dir_name):
        """
        Checks if folder ``dir_name`` exists, and if it does not exist, it will be created.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def write_to_file(mu_w, sigma2_w, alpha, mu_g, sigma2_g, mu_o, sigma2_o, hyp, path, logger):
        """
        Writes parameters (variational and hyper-parameters) to CSV files.

        Args:
            alpha: numpy array of size N x N.
            mu: numpy array of size N x N.
            sigma2: numpy array of size N x N.
            hyp: numpy array.
            path: string. Path to directory in which result files will be saved.
        """

        logger.debug("writing results to the file")
        np.savetxt(path + '/mu_w' + '.csv', mu_w, delimiter=',', comments='')
        np.savetxt(path + '/sigma2_w' + '.csv', sigma2_w, delimiter=',', comments='')
        np.savetxt(path + '/mu_g' + '.csv', mu_g, delimiter=',', comments='')
        np.savetxt(path + '/sigma2_g' + '.csv', sigma2_g, delimiter=',', comments='')
        np.savetxt(path + '/mu_o' + '.csv', mu_o, delimiter=',', comments='')
        np.savetxt(path + '/sigma2_o' + '.csv', sigma2_o, delimiter=',', comments='')
        np.savetxt(path + '/hyper' + '.csv', hyp, delimiter=',', comments='')
        w = mu_w / sigma2_w
        w_max = np.amax(w)
        w = w / w_max
        p = alpha / (1.0 + alpha)
        b = p * w
        np.savetxt(path + '/p_new' + '.csv', b, delimiter=',', comments='')
        np.savetxt(path + '/p' + '.csv', p, delimiter=',', comments='')
        logger.debug("finished writing results to the file")

    @staticmethod
    def write_to_file_callback(path, logger):
        def toFile(mu_w, sigma2_w, alpha, mu_g, sigma2_g, mu_o, sigma2_o, sigma2, variance, lengthscale):
            ExprUtil.write_to_file(mu_w, sigma2_w, alpha, mu_g, sigma2_g, mu_o, sigma2_o,  np.array([sigma2, lengthscale, variance]), path, logger)

        return toFile
