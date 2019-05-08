import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import argparse
from scipy.spatial.distance import pdist, squareform
TEST_FLOAT = tf.float64

def pretty_matrix_print(mat, fmt="f"):
	# prints beautifylly 2D matrix
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def parse_args():
	# parse important variables
	parser = argparse.ArgumentParser()
	parser.add_argument('-nrf')
	parser.add_argument('-s')
	parameters = parser.parse_args()
	N_rf = int(parameters.nrf)
	S = int(parameters.s)
	return N_rf,S

def set_parameters():
	# initalize parameters that are not of interest
	N = 1
	T = 3
	D = 1
	variance = 0.5
	lengthscale = 1. / np.sqrt(D)
	# t = [np.array(range(0, T))[:, np.newaxis]]
	# norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
	# t = tf.cast(norm_t,dtype=TEST_FLOAT)
	return N,T,D,variance,lengthscale

def get_t(T):
	t_init = [np.array(range(0, T))[:, np.newaxis]]
	norm_t = (t_init[0] - np.mean(t_init[0])) / np.double(np.std(t_init[0]))
	t_init = tf.cast(norm_t,dtype=TEST_FLOAT)
	# t = tf.cast(tf.random_normal([T, D]),dtype=TEST_FLOAT)
	return t_init

def get_omega(S,N_rf,D,lengthscale):
	# get Omega
	S_by_Nrf_by_D = ((S,N_rf, D))
	Nrf_by_D = (N_rf, D)
	# draw N(0,1) as basis for omega
	random_normal_for_omega = tf.random_normal(S_by_Nrf_by_D, dtype=TEST_FLOAT)
	# initate mu and sigma for omega
	mu_omega = tf.zeros(Nrf_by_D, dtype=TEST_FLOAT)
	sigma2_omega = tf.ones(Nrf_by_D, dtype=TEST_FLOAT)/lengthscale/lengthscale
	# calculate omega
	Omega = tf.multiply(random_normal_for_omega, tf.sqrt(sigma2_omega)) + mu_omega
	return Omega

def get_gamma(S,N,N_rf):
	# get Gamma
	S_by_N_by_2Nrf = ((S, N, 2*N_rf))
	N_by_2Nrf = (N, 2*N_rf)
	random_normal_for_gamma = tf.random_normal(S_by_N_by_2Nrf,dtype=TEST_FLOAT)
	mu_gamma = tf.zeros(N_by_2Nrf,dtype=TEST_FLOAT)
	sigma2_gamma = tf.ones(N_by_2Nrf,dtype=TEST_FLOAT)
	Gamma = tf.multiply(random_normal_for_gamma, tf.sqrt(sigma2_gamma)) + mu_gamma
	# Gamma = random_normal_for_gamma
	return Gamma

def get_z(Omega,Gamma,N_rf,S,T,D,variance,t):
	Omega_temp = tf.reshape(Omega, [S*N_rf, D])
	# find omega*x
	Omega_X = tf.reshape(tf.matmul(Omega_temp, tf.transpose(t)),[S, N_rf, T])
	# get Ф
	Fi = tf.cast(tf.sqrt(tf.math.divide(variance, N_rf)),dtype=TEST_FLOAT)*tf.concat([tf.cos(Omega_X),tf.sin(Omega_X)], axis=1)
	# Z = Gamma * Ф
	Z = tf.matmul(Gamma, Fi)
	return tf.reduce_mean(Z,axis = 0), tf.reduce_mean(Fi,axis = 0)

def initalize_variables(t_init,Omega_init,Gamma_init,Z_init, Fi_init):
	t = tf.Variable(t_init, dtype=TEST_FLOAT) 
	Omega = tf.Variable(Omega_init, dtype=TEST_FLOAT) 
	Gamma = tf.Variable(Gamma_init, dtype=TEST_FLOAT)
	Fi = tf.Variable(Fi_init, dtype=TEST_FLOAT)
	Z = tf.Variable(Z_init, dtype=TEST_FLOAT)
	init = tf.initialize_all_variables()
	return t, Z, init, Fi

def kernel(t,variance,lengthscale):
	# get Kt - kernel
	# this code taken from latent class from github
	dist = tf.reduce_sum(tf.square(t), 1)
	dist = tf.reshape(dist, [-1, 1])
	_two = tf.cast(2.0, dtype = TEST_FLOAT)
	sq_dists = dist-_two*tf.matmul(t, tf.transpose(t))+tf.transpose(dist)
	Kt = variance*tf.exp(-tf.abs(sq_dists) / _two*tf.square(lengthscale))
	return Kt

def kernel_edward(X, lengthscale, variance):
  X = X / lengthscale
  Xs = tf.reduce_sum(tf.square(X), 1)
  X2 = X
  X2s = Xs
  square = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - \
      2 * tf.matmul(X, X2, transpose_b=True)
  output = variance * tf.exp(-square / 2)
  return output
	
if __name__ == '__main__':
# run the code by : python3 test_kernel_approximation.py -nrf 10000 -s 200
# parameters: nrf - number of random features, s - number of MC samples
	tf.logging.set_verbosity(tf.logging.ERROR)
	sess = tf.InteractiveSession()
	# parse important parameters
	N_rf,S = parse_args()
	# the rest of parameters assign the same way as in code in Latnet
	N,T,D,variance,lengthscale = set_parameters()
	print("\nparameters:\n\t# of random features {}\n\t# of MC samples for approximation {}\n\t# of nodes N {}\n\t# of observations per node T {}\n\t# of dimensions for\ data D {}".format(N_rf, S, N, T, D))
	t_init = get_t(T)
	# get all the random variables in the model
	Omega_init = get_omega(S,N_rf,D,lengthscale)
	Gamma_init = get_gamma(S,N,N_rf)
	# calculate Random feature approximation to kernel
	Z_init, Fi_init = get_z(Omega_init,Gamma_init,N_rf,S,T,D,variance,t_init)
	t, Z, init, Fi = initalize_variables(t_init,Omega_init,Gamma_init,Z_init, Fi_init)
	sess.run(init)
	# appr_Kt = tf.matmul(tf.transpose(Z),Z)
	appr_Kt = tf.matmul(tf.transpose(Fi),Fi)
	print("\napprox Kt")
	pretty_matrix_print(appr_Kt.eval())
	Kt = kernel(t,variance,lengthscale)
	print("\nlatnet Kt")
	pretty_matrix_print(Kt.eval())
	# kernel_edw = kernel_edward(t, lengthscale, variance)
	# print("\nedw Kt")
	# pretty_matrix_print(kernel_edw.eval())
	# print("\nbuilt-in Kt")
	# pretty_matrix_print(variance*rbf_kernel(t.eval(), gamma = 1))

	
