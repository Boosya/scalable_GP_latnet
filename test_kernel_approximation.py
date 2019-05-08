import tensorflow as tf
import numpy as np
import argparse
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
	print("\nparsed parameters: \n \tnumber of random features {} \n \tnumber of MC samples for approximation {}".format(N_rf, S))
	return N_rf,S

def initalize_variables():
	# initalize parameters that are not of interest
	N = 2
	T = 3
	D = 1
	variance = 0.50
	lengthscale = 1. / np.sqrt(T)
	t = [np.array(range(0, T))[:, np.newaxis]]
	norm_t = (t[0] - np.mean(t[0])) / np.double(np.std(t[0]))
	t = tf.cast(norm_t,dtype=TEST_FLOAT)
	p = 0.5
	posterior_lambda_ = .15
	sigma2_n=tf.cast(0.31,TEST_FLOAT)
	sigma2_g=tf.cast(1e-4,TEST_FLOAT)
	return N,T,D,variance,lengthscale,t,p,posterior_lambda_,sigma2_n,sigma2_g

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
	return Gamma

def get_w(S,N,p):
	# get W
	SbyNbyN = ((S, N, N))
	N_by_N = ((N, N))
	random_normal_for_w = tf.random_normal(SbyNbyN, dtype=TEST_FLOAT)
	mu = tf.zeros(N_by_N, dtype=TEST_FLOAT)
	sigma2 = tf.multiply(tf.cast(1. / (N * p), TEST_FLOAT), tf.ones(N_by_N, dtype=TEST_FLOAT))
	W = tf.multiply(random_normal_for_w, tf.sqrt(sigma2)) + mu
	W = tf.matrix_set_diag(W, tf.zeros((S, N), dtype=TEST_FLOAT))
	return W

def get_a(S,N,p,posterior_lambda_):
	# get A
	SbyNbyN = ((S, N, N))
	N_by_N = ((N, N))
	random_normal_for_a = tf.random_uniform(SbyNbyN, minval=0, maxval=1, dtype=TEST_FLOAT)
	eps = tf.constant(1e-20, dtype=TEST_FLOAT)
	alpha = tf.multiply(tf.cast(p / (1. - p), TEST_FLOAT), tf.ones(N_by_N, dtype=TEST_FLOAT))
	X = tf.math.divide(
		tf.add(tf.log(alpha), tf.subtract(tf.log(random_normal_for_a + eps), tf.log(tf.constant(1.0, dtype=TEST_FLOAT) - random_normal_for_a + eps))),
		posterior_lambda_)
	A = tf.sigmoid(X)
	A = tf.matrix_set_diag(A, tf.zeros((S, N), dtype=TEST_FLOAT))
	return A

def calc_ELL_approx(Omega,Gamma,W,A,N_rf,S,N,T,D,variance,t,Y,sigma2_n,sigma2_g):
	Omega_temp = tf.reshape(Omega, [S*N_rf, D])
	# find omega*x
	Omega_X = tf.reshape(tf.matmul(Omega_temp, tf.transpose(t)),[S, N_rf, T])
	# get Ф
	Fi = tf.cast(tf.sqrt(tf.math.divide(variance, N_rf)),dtype=TEST_FLOAT)*tf.concat([tf.cos(Omega_X),tf.sin(Omega_X)], axis=1)
	# Z = Gamma * Ф
	Z = tf.matmul(Gamma, Fi)
	B = tf.multiply(A, W)

	# find B*noise 
	noise_g = tf.random_normal(((N,T)), dtype=TEST_FLOAT) * tf.sqrt(sigma2_g)
	B_temp = tf.reshape(B, [S*N,N])
	B_noise = tf.matmul(B_temp,noise_g)
	B_noise = tf.reshape(B_noise,[S,N,T])
	I = tf.constant(np.identity(N), dtype=TEST_FLOAT)
	IB = tf.subtract(I, B)
	IB_inverse = tf.matrix_inverse(IB)
	# expected_Y: (I - B)^(-1)( ГФ + B*noise), size (S,N,T)
	expected_Y = tf.matmul(IB_inverse,Z + B_noise)
	real_Y = tf.expand_dims(tf.transpose(Y), 0)
	# get a second norm by N
	norm = tf.norm(expected_Y-real_Y, ord = 2, axis = 1)
	# print("ELL matrix before summation for approximation")
	# pretty_matrix_print(norm.eval())
	# find sum by T
	norm_sum_by_T = tf.reduce_sum(norm, axis = 1)
	# avg over MC samples
	norm_sum_by_T_avg_by_S = tf.reduce_mean(norm_sum_by_T)
	_half = tf.constant(0.5, dtype=TEST_FLOAT)
	_two = tf.constant(2.0,dtype=TEST_FLOAT)
	_pi = np.pi
	_N = tf.constant(N,dtype=TEST_FLOAT)
	res = -_half* N*T*tf.log(_two*_pi*sigma2_n) -_half/sigma2_n*norm_sum_by_T_avg_by_S
	return res

def kernel(t,variance,lengthscale):
	# get Kt - kernel
	# this code taken from latent class from github
	dist = tf.reduce_sum(tf.square(t), 1)
	dist = tf.reshape(dist, [-1, 1])
	sq_dists = tf.add(tf.subtract(dist, tf.multiply(tf.cast(2.0, dtype = TEST_FLOAT), tf.matmul(t, tf.transpose(t)))),tf.transpose(dist))
	Kt = tf.multiply(tf.cast(variance,dtype = TEST_FLOAT), tf.exp(
				tf.negative(tf.math.divide(tf.abs(sq_dists), tf.multiply(tf.cast(2.0, dtype = TEST_FLOAT), tf.square(tf.cast(lengthscale,dtype=TEST_FLOAT))))))) + tf.constant(1e-5 * np.identity(T),dtype=TEST_FLOAT)
	return Kt

def calc_ELL_real(A,W,N_rf,S,N,T,D,variance,lengthscale,t,sigma2_g,sigma2_n):
	# calculates ELL term
	# this code taken from latent class from github
	Kt = kernel(t,variance,lengthscale)
	I = tf.constant(np.identity(N), dtype=TEST_FLOAT)
	
	B = tf.multiply(A, W)
	IB = tf.subtract(I, B)
	L = tf.matrix_inverse(IB)
	Eg = tf.matmul(tf.matmul(L, tf.matmul(B, tf.transpose(B, [0, 2, 1]))),
				tf.transpose(L, [0, 2, 1]))
	Sn = tf.add(tf.multiply(sigma2_g, Eg), tf.multiply(sigma2_n, I))
	Kf = tf.matrix_inverse(tf.matmul(tf.transpose(IB, [0, 2, 1]), IB))
	lt, Qt = tf.self_adjoint_eig(Kt)
	lt = tf.cast(lt, TEST_FLOAT)
	Qt = tf.cast(Qt, TEST_FLOAT)
	ln, Qn = tf.self_adjoint_eig(Sn)
	Ln_inv = tf.matrix_diag(tf.sqrt(tf.math.divide(tf.constant(1., dtype=TEST_FLOAT), ln)))
	Ln_inv_Qn = tf.matmul(Ln_inv, tf.transpose(Qn, [0, 2, 1]))
	Ktilda_f = tf.matmul(tf.matmul(Ln_inv_Qn, Kf), tf.transpose(Ln_inv_Qn, [0, 2, 1]))
	ltilda_f, Qtilda_f = tf.self_adjoint_eig(Ktilda_f)

	Lt_Lf = tf.add(tf.matmul(tf.transpose(tf.expand_dims(tf.tile(tf.expand_dims(lt, -1), [1, S]), -3)),tf.expand_dims(ltilda_f, -2)),tf.constant(1.0, dtype=TEST_FLOAT))

	logdet = tf.multiply(tf.cast(T, TEST_FLOAT), tf.reduce_sum(tf.log(ln))) + \
				tf.reduce_sum(tf.log(Lt_Lf))

	Ytilda = tf.matmul(tf.matmul(tf.tile(tf.expand_dims(Y, -3), [S, 1, 1]), Qn), Ln_inv)

	Qt_expanded = tf.tile(tf.expand_dims(Qt, -3), [S, 1, 1])
	Ytf = tf.multiply(tf.math.divide(tf.constant(1.0, dtype=TEST_FLOAT), Lt_Lf),
						tf.matmul(tf.transpose(Qt_expanded, [0, 2, 1]), tf.matmul(Ytilda, Qtilda_f)))
	ySy = tf.reduce_sum(tf.matrix_diag_part(
		tf.matmul(tf.matmul(tf.matmul(tf.transpose(Ytilda, [0, 2, 1]), Qt_expanded), Ytf),tf.transpose(Qtilda_f, [0, 2, 1]))))
	_half = tf.constant(0.5, dtype=TEST_FLOAT)
	res = tf.multiply(-_half, tf.math.divide(logdet, S)) + tf.multiply(-_half, tf.math.divide(ySy, S))  - (N * T) / tf.constant(2.0, TEST_FLOAT) * tf.log(tf.constant(2.0, dtype=TEST_FLOAT) * np.pi)
	return res

	
if __name__ == '__main__':
# run the code by 
# python3 test_kernel_approximation.py -nrf 10000 -s 200
# parameters:
# nrf - number of random features
# s - number of MC samples

	sess = tf.InteractiveSession()
	# parse important variables
	N_rf,S = parse_args()
	# the rest of variables assign the same way as in code in Latnet
	N,T,D,variance,lengthscale,t,p,posterior_lambda_,sigma2_n,sigma2_g = initalize_variables()
	# pick random Y values
	Y = tf.random_normal(((T,N)), dtype=TEST_FLOAT)
	
	# get all the random variables in the model
	Omega = get_omega(S,N_rf,D,lengthscale)
	Gamma = get_gamma(S,N,N_rf)
	W = get_w(S,N,p)
	A = get_a(S,N,p,posterior_lambda_)
	# calculate ELL with random feature expansion
	ELL_approx = calc_ELL_approx(Omega,Gamma,W,A,N_rf,S,N,T,D,variance,t,Y,sigma2_n,sigma2_g)
	print("\napproximated ELL: {}".format(ELL_approx.eval()))
	# calculate ELL based on code from Latnet
	ELL_real = calc_ELL_real(A,W,N_rf,S,N,T,D,variance,lengthscale,t,sigma2_g,sigma2_n)
	print("\nreal ELL: {}".format(ELL_real.eval()))
	
