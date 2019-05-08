import numpy as np
import os


class ExprUtil:

    @staticmethod
    def check_dir_exists(dir_name):
        """
        Checks if folder ``dir_name`` exists, and if it does not exist, it will be created.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def write_to_file(alpha, mu, sigma2, hyp, path, logger):
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
        np.savetxt(path + '/sigma2' + '.csv', sigma2,
                   delimiter=',', comments='')
        np.savetxt(path + '/alpha' + '.csv', alpha, delimiter=',', comments='')
        np.savetxt(path + '/p' + '.csv', alpha /
                   (1.0 + alpha), delimiter=',', comments='')
        np.savetxt(path + '/hyp' + '.csv', hyp, delimiter=',', comments='')
        logger.debug("finished writing results to the file")

    @staticmethod
    def write_to_file_callback(path, logger):
        def toFile(alpha_, mu, sigma2_, sigma2_n_, lengthscale, variance):
            ExprUtil.write_to_file(alpha_, mu, sigma2_, np.array(
                [sigma2_n_, lengthscale, variance]), path, logger)
        return toFile
