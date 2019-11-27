import csv
from flags_syd_prices import Flags
import pandas
import time
import argparse


from scalable_latnet.scalablelatnet import ScalableLatnet
from scalable_latnet.expr_util import ExprUtil
import logging

RESULTS = 'results/'
DATA = 'data/'


def syd_prices(input_file, output_folder, from_, to_):
    """
    Args:
        output_folder: the folder in which the results of running LATNET will be saved in.
        input_file: the name of input file that contains observations from nodes.
        sims: which simulation data to use (sim1 - 5 nodes, sim2 - 10 nodes, sim3 - 15 nodes)
        ti: number of observations to use for running the model (from 1...ti)
        s: which subject to run model on (each dataset has 50 subjects), each subject has signals from network nodes

    Returns: None
    """
    logger_name = 'syd_prices_scalable_latnet_' + str(from_) + '_' + str(to_)

    # parsing data for all dates
    all_dates_data = pandas.read_csv(DATA + input_file)
    data = all_dates_data.iloc[:,(4 * (from_ - 1995) + 1):(4 * (to_ - 1999) + 21)]

    # standardizing inputs
    mean_ = data.stack().mean()
    std_ = data.stack().std()
    data = (data - mean_) / std_
    sydney_prices_sim(data, output_folder,logger_name, 20)
    # return logger_name


def sydney_prices_sim(data, folder_name, logger_name, Ti):
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

    myflags = Flags()
    logger.debug("Parameters of the model")
    myflags.log_flags(logger)

    # split test, train and validation
    n_validation_samples = int(Ti * myflags.get_flag('validation_percent'))
    n_test_samples = int(Ti * myflags.get_flag('validation_percent'))
    n_training_samples = Ti - n_validation_samples - n_test_samples

    train_data = data.iloc[:n_training_samples, :]
    validation_data = data.iloc[n_training_samples:n_training_samples + n_validation_samples, :]
    test_data = data.iloc[n_training_samples + n_validation_samples:, :]

    myLatnet = ScalableLatnet(myflags, dim=1, train_data=train_data,  validation_data=validation_data,
                              test_data=test_data, true_conn = None, logger=logger)
    mu, sigma2, alpha = myLatnet.optimize()
    ExprUtil.write_to_file_callback(path, logger)(mu, sigma2, alpha)


if __name__ == '__main__':

    window_size = 4
    window_slide = 1
    start_pose = 1995
    end_pose = start_pose + window_size
    while end_pose <= 2014:
        file_name = str(start_pose) + '_' + str(end_pose)
        syd_prices('syd_prices/data.csv', 'syd_prices/' + file_name + '/', start_pose, end_pose)
        start_pose += window_slide
        end_pose = start_pose + window_size