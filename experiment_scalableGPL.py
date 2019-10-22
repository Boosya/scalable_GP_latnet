import csv
from flags import Flags
import pandas
import time
import argparse

# TODO try other datasets (employment...) - sydney
# TODO change p to calculate A, W
# TODO fix calculations of AUC to be faster
# TODO add tests to all the functions
# TODO comment all
# TODO double-check lengthscale and omega:
#       prior fixed: no KL term (posterior = prior), generrte random smples from st normal ONCE L - variables,
#       var-fixed: sample st normal once, optimize parameters, L - variable
#       var-resamples: same as prev but sample every iteration
# TODO add noise that is missing, small one
# TODO calculate the time complexity


# TODO ask what is X in random feature? why never use signals from other nodes? why use complete I-B if diagonal is removed?

from scalable_latnet.scalablelatnet import ScalableLatnet
from scalable_latnet.expr_util import ExprUtil
import logging

RESULTS = 'results/'
DATA = 'data/'
N_OBJECTS = 50


def functional_connectivity_group(config):
    """
    Args:
        output_folder: the folder in which the results of running LATNET will be saved in.
        input_file: the name of input file that contains observations from nodes.
        sims: which simulation data to use (sim1 - 5 nodes, sim2 - 10 nodes, sim3 - 15 nodes)
        ti: number of observations to use for running the model (from 1...ti)
        s: which subject to run model on (each dataset has 50 subjects), each subject has signals from network nodes

    Returns: None
    """
    output_folder = config['output_folder']
    input_file = config['input_file']
    Ti = int(config['Ti'])
    s = int(config['s'])
    sims = config.get('sims')
    logger_name = 'scalable_latnet_' + str(sims) + '_' + str(Ti) + '_' + str(s)
    output_folder = output_folder + str(Ti) + '/'
    folder_name = output_folder + 'subject' + '_' + str(s)

    # parsing data for subject
    all_subjects_data = pandas.read_csv(DATA + input_file, header=None)
    n_sign_per_subjects = int(all_subjects_data.shape[0] / N_OBJECTS)
    n_nodes = int(all_subjects_data.shape[1])
    data = all_subjects_data.iloc[s * n_sign_per_subjects:s * n_sign_per_subjects + Ti, :]

    # parsing network to use for testing and validatopn
    all_network_file = 'fmri_sim/net_' + sims + '.csv'
    all_network_data = pandas.read_csv(DATA + all_network_file, header=None)
    subject_true_connections_array = all_network_data.iloc[s, :100]
    subject_true_connections = subject_true_connections_array.as_matrix()

    # standardizing inputs
    mean_ = data.stack().mean()
    std_ = data.stack().std()
    data = (data - mean_) / std_

    functional_connectivity_sim(data, subject_true_connections, folder_name, s, logger_name, sims, Ti, s)
    return logger_name


def functional_connectivity_sim(data, subject_true_connections, folder_name, subject, logger_name, sims, Ti, s):
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

    myflags = Flags(sims, Ti, s)
    logger.debug("Parameters of the model")
    myflags.log_flags(logger)

    # split test, train and validation
    n_validation_samples = int(Ti * myflags.get_flag('validation_percent'))
    n_test_samples = int(Ti * myflags.get_flag('validation_percent'))
    n_training_samples = Ti - n_validation_samples - n_test_samples

    train_data = data.iloc[:n_training_samples, :]
    validation_data = data.iloc[n_training_samples:n_training_samples + n_validation_samples, :]
    test_data = data.iloc[n_training_samples + n_validation_samples:, :]

    myLatnet = ScalableLatnet(myflags, subject=subject, dim=1, train_data=train_data,  validation_data=validation_data,
                              test_data=test_data, true_conn = subject_true_connections, logger=logger)
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
    config = {'sims': args.sim, 'Ti': args.Ti, 's': args.s, 'output_folder': 'fmri/fmri_' + args.sim + '_scalableGPL/',
              'input_file': 'fmri_sim/ts_' + args.sim + '.csv'}
    functional_connectivity_group(config)
