import csv
# import sys
from flags import Flags
import pandas
import time
import argparse

# export OMP_NUM_THREADS=1

# TODO add tests to all the functions
# TODO add new way to parse dataset

# TODO porior fixed: no KL term (posterior = prior), generrte random smples from st normal ONCE L - variables,
#       var-fixed: sample st normal once, optimize parameters, L - variable
#       var-resamples: same as prev but sample every iteration

# TODO try running vanila the above 3
# TODO use optimal L from latent nd fix it in tf.Variable - should really work
# search for word "ident"
#

# TODO enable eager execution and track matrices
# TODO add noise that is missing, small one
# TODO try multiplying AW in the end - not working?
# TODO try other datasets
# TODO implement AUC check in the code and log the decreasing rate
# TODO calculate the time complexity

# TODO add noise that is missing, small one
# TODO histigram of p matrix and compare


# ? when calculating KL (Omega) prior over sigma - lengthscale from posterior or from flags?

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

    # parsing data for subject
    all_subjects_data = pandas.read_csv(DATA + input_file, header=None)
    n_sign_per_subjects = int(all_subjects_data.shape[0]/N_OBJECTS)
    data = all_subjects_data.iloc[s * n_sign_per_subjects:s * n_sign_per_subjects + Ti, :]

    # standardizing inputs
    mean_ = data.stack().mean()
    std_ = data.stack().std()
    data = (data - mean_) / std_

    # split test, train and validation
    n_validation_samples = int(Ti*0.2)
    n_test_samples = int(Ti*0.2)
    n_training_samples = Ti - n_validation_samples - n_test_samples

    train_data = data.iloc[:n_training_samples,:]
    validation_data = data.iloc[n_training_samples:n_training_samples+n_validation_samples, :]
    test_data = data.iloc[n_training_samples+n_validation_samples:, :]

    functional_connectivity_sim(train_data, validation_data, test_data, folder_name, s, logger_name,sims,Ti,s)
    return logger_name


def functional_connectivity_sim(train_data, validation_data, test_data, folder_name,subject,logger_name,sims,Ti,s):
    start_time = time.time()

    path = RESULTS + folder_name
    ExprUtil.check_dir_exists(path)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path + '/run.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    (T, _) = train_data.shape
    t = [np.array(range(0,T))[:,np.newaxis]]
    norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))

    myflags = Flags(T,sims,Ti,s)
    logger.debug("Parameters of the model")
    myflags.log_flags(logger)
    myLatnet = ScalableLatnet(myflags, subject=subject, dim=1, t=norm_t, train_data=train_data,  validation_data=validation_data, test_data=test_data, logger=logger)
    elbo_, sigma2_n_,  mu, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale, variance = \
        myLatnet.optimize()
    end_time = time.time()
    ExprUtil.write_to_file_callback(path, logger)(
        alpha_, mu, sigma2_, sigma2_n_, lengthscale, variance)
    csvdata = [start_time, end_time, end_time - start_time]
    with open(path + '/timing.csv', "w") as csvFile:
        Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
        Fileout.writerow(csvdata)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify the data to process.')
    parser.add_argument('--sim', help='which sim to use')
    parser.add_argument('--Ti', help='number of observations per node')
    parser.add_argument('--s', help='which oject')

    args = parser.parse_args()

    methods = ["scalableGPL"]
    config = {'sims': args.sim, 'Ti': args.Ti, 's': args.s, 'output_folder': 'fmri/fmri_' +
                                    args.sim+'_scalableGPL/', 'input_file': 'fmri_sim/ts_'+args.sim+'.csv'}
    functional_connectivity_group(config)