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
    def write_to_file( mu, sigma2, path,logger):
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
        np.savetxt(path + '/mu' + '.csv', mu, delimiter=',', comments='')
        np.savetxt(path + '/sigma2' + '.csv', sigma2, delimiter=',', comments='')

        W = mu/sigma2
        np.savetxt(path + '/p_new' + '.csv',W,delimiter=',',comments='')

        logger.debug("finished writing results to the file")


    @staticmethod
    def write_to_file_callback(path, logger):
        def toFile( mu, sigma2):
            ExprUtil.write_to_file( mu, sigma2, path, logger)
        return toFile