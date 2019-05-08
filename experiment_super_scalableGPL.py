import csv
# import sys

import pandas
import time

from super_scalable_latnet.latnet import Latnet
from super_scalable_latnet.expr_util import Logger as super_scalable_latnet_Logger
from super_scalable_latnet.expr_util import ExprUtil
import numpy as np

RESULTS='results/'
DATA='data/'
N_OBJECTS =50

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

	output_folder = output_folder + str(Ti) + '/'
	data = pandas.read_csv(DATA + input_file, header=None)
	Y = []
	l = int(data.values.shape[0]/N_OBJECTS)
	for s in range(N_OBJECTS):
		"""s: subject number. Each file contains data from 50 subjects."""
		Y.append([])
		for i in range(data.shape[1]):
			Y[s].append(data.ix[:, i].values[s*l:(s+1)*l, np.newaxis][0:Ti,:])
		mean_ = np.hstack(Y[s]).mean()
		std_ = np.hstack(Y[s]).std()

		# for standardizing the inputs
		for i in range(data.shape[1]):
			Y[s][i] = (Y[s][i] - mean_) / std_

	for s in range(len(Y)):
		folder_name = output_folder + 'subject' + '_' + str(s)
		functional_connectivity_sim(Y[s], folder_name,s)

def functional_connectivity_sim(Y, folder_name, s):
	path = RESULTS + folder_name
	ExprUtil.check_dir_exists(path)
	super_scalable_latnet_Logger.init_logger(path, "run", folder_name)
	(T,D) = Y[0].shape

	# t is number of observations Ti written as array 1xT1
	'''
	[array([[ 0],
       [ 1],
       [ 2],
       [ 3]
	'''
	t = [np.array(range(0, T))[:, np.newaxis]]
	start_time = time.time()
	# normalize t : [[-1.7034199 ] [-1.64567685] ...
	norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
	Y_data = np.hstack(Y)

	# initialize hyperparameters
	init_lenthscale = 1. / np.sqrt(D)
	lambda_prior = 1.0
	lambda_postetior = .15
	opt_targets = {'var': 2000, 'hyp': 2000}
	# opt_targets = {'var': 20}
	n_total_iter = 7
	init_sigma2_n=0.31
	init_sigma2_g=1e-4
	init_variance=0.50
	init_p = 0.5
	var_lr=0.01
	hyp_lr=0.001
	N_rf = 500
	D = 1
	n_samples = 200
	approximate_kernel = True
	fix_kernel=False
	elbo_, sigma2_n_,  mu, sigma2_, mu_gamma, sigma2_gamma_,mu_omega, sigma2_omega_, alpha_, lengthscale, variance = \
		Latnet.optimize(s, D, N_rf, norm_t, Y_data , opt_targets.keys() ,n_total_iter, opt_targets, super_scalable_latnet_Logger.logger,init_sigma2_n ,init_sigma2_g, init_lenthscale,init_variance,init_p, lambda_prior,lambda_postetior,var_lr,hyp_lr, n_samples, approximate_kernel, fix_kernel = fix_kernel)
	end_time = time.time()
	ExprUtil.write_to_file_callback(path)(alpha_, mu, sigma2_,sigma2_n_,lengthscale,variance)
	csvdata = [start_time, end_time, end_time - start_time]
	with open(path + '/timing.csv', "w") as csvFile:
		Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
		Fileout.writerow(csvdata)
	for l in super_scalable_latnet_Logger.logger.handlers:
		l.close()
		
if __name__ == '__main__':
	configs = []
	method = "super_scalable_GPL"
	for sims in ['sim1']:
	# for sims in ['sim1','sim2','sim3']:
		"""sims: which simulation in the dataset """
		# for Ti in [200]:
		for Ti in [50, 100, 200]:
			"""Ti: number of observations"""
			configs.append({'sims': sims, 'Ti':Ti})
	for i in range(len(configs)):
		sims = configs[i]['sims']
		Ti = configs[i]['Ti']
		output_folder = 'fmri/fmri_'+sims+'_'+method+'/'
		input_file = 'fmri_sim/ts_'+sims+'.csv'
		functional_connectivity_group(output_folder,input_file,method,Ti=Ti)