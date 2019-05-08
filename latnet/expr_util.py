import numpy as np
import os
import logging
from io import StringIO as StringBuffer
import io
# from StringIO import StringIO as StringBuffer


class Logger:
    """
    This is a logger class used for logging the output of the algorithm.
    """


    @staticmethod
    def init_logger(path=None, name=None, logger_name=None):
        if logger_name is None:
            logger_name = __name__
        Logger.logger = logging.getLogger(logger_name)
        Logger.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # formatter = logging.Formatter('%(message)s')

        if path is not None:
            fh = logging.FileHandler(path + '/' + name + '.log')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            Logger.logger.addHandler(fh)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        Logger.logger.addHandler(ch)

        Logger.logger.debug("logging started")

    @staticmethod
    def remove_handlers():
        Logger.logger.handlers = []


    @staticmethod
    def get_string_logger():
        log_capture_string = StringBuffer()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        Logger.logger.addHandler(ch)
        return log_capture_string


class ExprUtil:

    @staticmethod
    def check_dir_exists(dir_name):
        """
        Checks if folder ``dir_name`` exists, and if it does not exist, it will be created.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def write_to_file(alpha, mu, sigma2, hyp, path,logger):
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
        np.savetxt(path + '/alpha' + '.csv', alpha, delimiter=',', comments='')
        np.savetxt(path + '/p' + '.csv', alpha / (1.0 + alpha), delimiter=',', comments='')
        np.savetxt(path + '/hyp' + '.csv', hyp, delimiter=',', comments='')
        logger.debug("finished writing results to the file")


    @staticmethod
    def write_to_file_callback(path, logger):
        def toFile(alpha_, mu, sigma2_, sigma2_n_, sigma2_g_, lengthscale, variance):
            if sigma2_g_:
                ExprUtil.write_to_file(alpha_, mu, sigma2_, np.array([sigma2_n_, sigma2_g_, lengthscale, variance]), path, logger)
            else:
                ExprUtil.write_to_file(alpha_, mu, sigma2_, np.array([sigma2_n_, lengthscale, variance]), path, logger)
        return toFile

