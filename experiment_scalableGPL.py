import csv
# import sys
from flags import Flags
import pandas
import time


# export OMP_NUM_THREADS=1


# TODO save best state so far
# TODO fix sampling from standart normal in omega (prior fixed and var fixed)
# TODO use 1000 random features
# TODO set core settings - use one subject for one terminal
# TODO try remove legthscale from learnable 
# TODO random feature code - other implemention
# TODO add noise that is missing
# TODO fix the seed
# TODO histigram of p matrix and compare
# TODO refactor code
# TODO try small noise


from scalable_latnet.scalablelatnet import ScalableLatnet
from scalable_latnet.expr_util import ExprUtil
import numpy as np
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor

RESULTS = 'results/'
DATA = 'data/'
N_OBJECTS = 50


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
    logger_name = 'scalable_latnet_'+str(sims)+'_'+str(Ti)+'_'+str(s)
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


def functional_connectivity_sim(Y, folder_name, subject, logger_name):
    start_time = time.time()

    path = RESULTS + folder_name
    ExprUtil.check_dir_exists(path)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/run.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    (T, D) = Y[0].shape
    t = [np.array(range(0, T))[:, np.newaxis]]
    norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
    Y_data = np.hstack(Y)

    flags = Flags(T)
    logger.debug("Parameters of the model")
    flags.log_flags(logger)
    elbo_, sigma2_n_,  mu, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale, variance = \
        ScalableLatnet.optimize(flags, subject, D, norm_t, Y_data,  logger)
    end_time = time.time()
    ExprUtil.write_to_file_callback(path, logger)(
        alpha_, mu, sigma2_, sigma2_n_, lengthscale, variance)
    csvdata = [start_time, end_time, end_time - start_time]
    with open(path + '/timing.csv', "w") as csvFile:
        Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
        Fileout.writerow(csvdata)
    flags.del_all_flags()


if __name__ == '__main__':

    N_OBJECTS = 50

    configs = []
    # methods = ("super_scalableGPL","latnet","scalableGPL")
    methods = ["scalableGPL"]
    n_workers = 1
    for method in methods:
        for sims in ['sim2']:
        # for sims in ['sim1', 'sim2', 'sim3']:
            """sims: which simulation in the dataset """
            # for Ti in [50, 100, 200]:
            for Ti in [100]:
                """Ti: number of observations"""
                # for s in range(N_OBJECTS):
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
