import csv

import pandas
import time
import logging

from latnet.latnet import Latnet
from latnet.expr_util import ExprUtil
import numpy as np

import sys
import traceback
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor


class ThreadPoolExecutorStackTraced(ThreadPoolExecutor):

    def submit(self, fn, *args, **kwargs):
        """Submits the wrapped function instead of `fn`"""

        return super(ThreadPoolExecutorStackTraced, self).submit(
            self._function_wrapper, fn, *args, **kwargs)

    def _function_wrapper(self, fn, *args, **kwargs):
        """Wraps `fn` in order to preserve the traceback of any kind of
        raised exception

        """
        try:
            return fn(*args, **kwargs)
        except Exception:
            raise sys.exc_info()[0](traceback.format_exc())  # Creates an
            # exception of the
            # same type with the
            # traceback as
            # message


RESULTS = 'results/'
DATA = 'data/'
N_OBJECTS = 50


def functional_connectivity_group(config):
    """
    Args:
            output_folder: the folder in which the results of running LATNET will be saved in.
            input_file: the name of input file that contains observations from nodes.
            Ti: number of observations to use for running the model (from 1...Ti)

    Returns: None
    """
    output_folder = config['output_folder']
    input_file = config['input_file']
    Ti = int(config['Ti'])
    s = int(config['s'])
    sims = config.get('sims')
    logger_name = 'latnet_'+str(sims)+'_'+str(Ti)+'_'+str(s)
    output_folder = output_folder + str(Ti) + '/'
    folder_name = output_folder + 'subject' + '_' + str(s)
    data = pandas.read_csv(DATA + input_file, header=None)
    Y = []
    l = int(data.values.shape[0]/N_OBJECTS)
    Y.append([])
    for i in range(data.shape[1]):
        Y[0].append(data.ix[:, i].values[s*l:(s+1)*l, np.newaxis][0:Ti, :])
    mean_ = np.hstack(Y[0]).mean()
    std_ = np.hstack(Y[0]).std()

    # for standardizing the inputs
    for i in range(data.shape[1]):
        Y[0][i] = (Y[0][i] - mean_) / std_
    functional_connectivity_sim(Y[0], folder_name, s, logger_name)
    return logger_name


def functional_connectivity_sim(Y, folder_name, s, logger_name):
    path = RESULTS + folder_name
    ExprUtil.check_dir_exists(path)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/run.log')
    fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    (T, D) = Y[0].shape
    t = [np.array(range(0, T))[:, np.newaxis]]
    start_time = time.time()
    norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
    Y_data = np.hstack(Y)

    # initialize hyperparameters
    init_lenthscale = 1. / np.sqrt(D)
    lambda_prior = 1.
    lambda_postetior = .15
    opt_targets = {'var': 10, 'hyp': 0}
    # opt_targets = {'var': 40}
    # n_total_iter = 2
    n_total_iter = 1
    init_sigma2_n = 0.31
    init_sigma2_g = 1e-4
    init_variance = 0.50
    init_p = 0.5
    var_lr = 0.01
    hyp_lr = 0.001
    n_samples = 200
    log_every = 10
    elbo_, sigma2_n_, sigma2_g_, mu, sigma2_, alpha_, lengthscale, variance = \
        Latnet.optimize(s, norm_t, Y_data, opt_targets.keys(), n_total_iter, opt_targets, logger, init_sigma2_g,
                        init_sigma2_n, init_lenthscale, init_variance, init_p, lambda_prior, lambda_postetior, var_lr, hyp_lr, n_samples, log_every=log_every,callback=None,seed=1)
    end_time = time.time()
    ExprUtil.write_to_file_callback(path, logger)(
        alpha_, mu, sigma2_, sigma2_n_, sigma2_g_, lengthscale, variance)
    csvdata = [start_time, end_time, end_time - start_time]
    with open(path + '/timing.csv', "w") as csvFile:
        Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
        Fileout.writerow(csvdata)


if __name__ == '__main__':
    
    N_OBJECTS = 50

    configs = []
    methods = ["latnet"]
    n_workers = 4
    for method in methods:
        # for sims in ['sim1','sim2','sim3']:
        for sims in ['sim2']:
            """sims: which simulation in the dataset """
            for Ti in [100]:
            # for Ti in [50, 100, 200]:
                """Ti: number of observations"""
                for s in range(1):
                    configs.append({'sims': sims, 'Ti': Ti, 's': s, 'output_folder': 'fmri/fmri_' +
                                    sims+'_'+method+'/', 'input_file': 'fmri_sim/ts_'+sims+'.csv'})


    with ThreadPoolExecutorStackTraced(max_workers=n_workers) as executor:
        futures = {executor.submit(functional_connectivity_group, config)
                for config in configs}
        for future in futures:
            try:
                future.result()
            except TypeError as e:
                print(e)
