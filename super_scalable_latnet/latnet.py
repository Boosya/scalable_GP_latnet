__author__ = 'AD'

import tensorflow as tf

import numpy as np
from tensorflow.python.framework.errors import OpError


class Latnet:
	FLOAT = tf.float64

	@staticmethod
	def replace_nan_with_zero(w):
		"""
		Replaces NANs with zeros.

		Args:
			w: Input tensor

		Returns:
			replaces NAN elements in input argument `w' with zeros.
		"""
		return tf.where(tf.is_nan(w), tf.ones_like(w) * 0.0, w)

	@staticmethod
	def contains_nan(w):
		"""
		Given argument `w', the method returns whether there are any NANs in `w'.
		"""
		for w_ in w:
			if tf.reduce_all(tf.is_nan(w_)) is None:
				return tf.reduce_all(tf.is_nan(w_))
		return tf.reduce_all(tf.is_nan(w_))

	@staticmethod
	def get_opzimier(objective, trainables, learning_rate, max_global_norm=1.0):
		"""
		Calculates the Tensorflow operation for optimizing `objective' function using AdamOptimizer.
		Note that NANs in the gradients will be replaced by zeros.

		Args:
			objective: double. Objective function to be optimized.
			trainables: List of variables which will be optimized.
			learning_rate: Learning rate of AdamOptimizer
			max_global_norm: Used for gradient clipping.
		"""
		grads = tf.gradients(objective, trainables)
		grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
		grad_var_pairs = zip([Latnet.replace_nan_with_zero(g)
		                     for g in grads], trainables)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		return optimizer.apply_gradients(grad_var_pairs), grads, Latnet.contains_nan(grads)

	@staticmethod
	def run_model(D, N_rf, t, Y, init_sigma2_n, init_lengthscle, init_variance, init_p, lambda_prior, lambda_postetior, var_lr, hyp_lr, n_samples, approximate_kernel, seed=None, fixed_kernel=False):
		"""
		This method calculates Tensorflow quantities that will be used for training LATNET.

		Args:
			t: array of size N x 1
			Y: array of size N x T
			init_sigma2_g: double. Initial variance of the connection noise.
			init_sigma2_n: double. Initial variance of the observation noise.
			init_lengthscle: double. Initial length scale of the RBF kernel.
			init_variance: double. Initial variance of the RBF kenel.
			lambda_prior: double. lambda of the Concrete distribution used for prior over `A_ij' matrix.
			lambda_postetior: double. lambda of the Concrete distribution used for posterior over `A_ij' matrix.
			var_lr: double. Learning rate for updating variational parameters.
			hyp_lr: Learning rate for updating hyper parameters.
			n_samples: int. Number of Monte-Carlo samples used.
			seed: int. Seed used for random number generators.
			fix_kernel: Boolean. Whether to optimize kenrel parameters during the optimization of hyper-parameters.

		Returns:
			var_opt: Tensorflow operation for optimizing variational parameters for one step.
			hyp_opt: Tensorflow operation for optimizing hyper-parameters for one step.
			elbo: Tensor with shape () containing ELBO, i.e., evidence lower-bound.
			kl_W: Tensor with shape () containing KL-divergence between prior over W and posterior W. It is
				calculated analytically.
			kl_A: Tensor with shape () containing KL-divergence between prior over A and posterior A.
				It is calculated using sampling.
			ell: Tensor with shape () containing approximated expected log-likelihood term.
			log_sigma2_n: Tensor of shape () containing the logarithm of sigma2_n, i.e., variance of observation noise.
			log_sigma2_g: Tensor of shape () containing the logarithm of sigma2_g, i.e., variance of connection noise.
			mu: Tensor of shape N x N containing variational parameter mu which is mean of W.
			log_sigma2: Tensor of shape N x N containing logarithm of sigma_2 which is variance of W.
			log_alpha: Tensor of shape N x N containing the logarithm of parameter alpha for Concrete distribution over A_ij.
			log_lengthscale: Tensor of shape () containing the logarithm of length scale of kernel.
			log_variance: Tensor of shape () containing the logarithm of variance of kernel.
			var_nans: Tensor of shape () representing whether there is any NAN in the gradients of variational parameters.
			hyp_nans: Tensor of shape () representing whether there is any NAN in the gradients of hyper parameters.
		"""

		# get variables that will be optimized.
		log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, log_lengthscale, log_variance = \
			Latnet.get_vairables(D, N_rf, t, Y, init_sigma2_n,
			                     init_lengthscle, init_variance, init_p)

		# get KL terms of ELL terms.
		kl_W, kl_G, kl_O, kl_A, ell = \
			Latnet.get_elbo(D, N_rf, t, Y, tf.exp(log_sigma2_n), mu, tf.exp(log_sigma2), mu_gamma, tf.exp(log_sigma2_gamma), mu_omega, tf.exp(log_sigma2_omega), tf.exp(log_alpha), tf.exp(log_lengthscale), tf.exp(
			    log_variance), tf.constant(lambda_prior, dtype=Latnet.FLOAT), tf.constant(lambda_postetior, dtype=Latnet.FLOAT), init_p, n_samples, approximate_kernel=approximate_kernel, seed=seed)

		# calculating ELBO
		# elbo = ell
		elbo = tf.negative(kl_W) + tf.negative(kl_G) + \
		                   tf.negative(kl_O) + tf.negative(kl_A) + ell
		# get the operation for optimizing variational parameters
		var_opt, _, var_nans = Latnet.get_opzimier(tf.negative(elbo), [
		                                           mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha], var_lr)

		if fixed_kernel:
			# get the operation for optimizing hyper-parameters except kernel parameters.
			hyp_opt, _, hyp_nans = Latnet.get_opzimier(
			    tf.negative(elbo), [log_sigma2_n], hyp_lr)
		else:
			# get the operation for optimizing hyper-parameters (kernel parameters and noise parameters)
			hyp_opt, _, hyp_nans = Latnet.get_opzimier(tf.negative(
			    elbo), [log_sigma2_n, log_lengthscale, log_variance], hyp_lr)

		return var_opt, hyp_opt, elbo, kl_W, kl_A, kl_G, kl_O, ell, log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, log_lengthscale, log_variance, var_nans, hyp_nans

	@staticmethod
	def logp_logistic(X, alpha, lambda_):
		"""
		Logarithm of Concrete distribution with parameter `alpha' and hyper-parameter `lambda_' at points `X', i.e.,
			Concrete(X;alpha, lambda_)

		Args:
			X: Tensor of shape S x N x N. Locations at which the distribution will be calculated.
			alpha:  Tensor of shape N x N. To be written.
			lambda_: double. Tensor of shape ().

		Returns:
			: A tensor of shape S x N x N. Element ijk is:
				log lambda_  - lambda_ * X_ijk + log alpha_jk - 2 log (1 + exp (-lambda_ * X_ijk + log alpha_jk))

		"""

		mu = tf.log(alpha)
		return tf.subtract(tf.add(tf.subtract(tf.log(lambda_), tf.multiply(lambda_, X)), mu),
					  tf.multiply(tf.constant(2.0, dtype=Latnet.FLOAT),
								  tf.log(tf.add(tf.constant(1.0, dtype=Latnet.FLOAT),
												tf.exp(tf.add(tf.negative(tf.multiply(lambda_, X)), mu))))))

	@staticmethod
	def get_KL_logistic(X, posterior_alpha, prior_lambda_, posterior_lambda_, prior_alpha):
		"""
		Calculates KL divergence between two Concrete distributions using samples from posterior Concrete distribution.

		KL(Concrete(alpha, posterior_lambda_) || Concrete(prior_alpha, prior_lambda))

		Args:
			X: Tensor of shape S x N x N. These are samples from posterior Concrete distribution.
			posterior_alpha: Tensor of shape N x N. alpha for posterior distributions.
			prior_lambda_: Tensor of shape (). prior_lambda_ of prior distribution.
			posterior_lambda_: Tensor of shape (). posterior_lambda_ for posterior distribution.
			prior_alpha: Tensor of shape N x N. alpha for prior distributions.

		Returns:
			: Tensor of shape () representing KL divergence between the two concrete distributions.

		"""
		logdiff = Latnet.logp_logistic(X, posterior_alpha, posterior_lambda_) - Latnet.logp_logistic(X, prior_alpha,
																									 prior_lambda_)
		logdiff = tf.matrix_set_diag(logdiff, tf.zeros((tf.shape(logdiff)[0], tf.shape(logdiff)[1]),
													   dtype=Latnet.FLOAT))  # set diagonal part to zero
		return tf.reduce_sum(tf.reduce_mean(logdiff, [0]))

	@staticmethod
	def get_KL_normal(posterior_mu, posterior_sigma2, prior_mu, prior_sigma2):
		"""
		Calculates KL divergence between two Mormal distributions, i.e.,
			KL(Normal(mu, sigma2), Normal(prior_mu, prior_sigma2))

		Args:
			posterior_mu: Tensor of shape N x N.
			posterior_sigma2: Tensor of shape N x N.
			prior_mu: Tensor of shape N x N.
			prior_sigma2: Tensor of shape N x N.

		Returns:
			Tensor of shape (), which contains the KL divergence.
		"""

		_half = tf.constant(0.5, dtype=Latnet.FLOAT)
		kl = tf.add(tf.math.divide(tf.add(tf.square(tf.subtract(posterior_mu, prior_mu)), posterior_sigma2), tf.multiply(tf.constant(
		    2., dtype=Latnet.FLOAT), prior_sigma2))					, -_half + _half * tf.log(prior_sigma2) - _half * tf.log(posterior_sigma2))
		kl = tf.matrix_set_diag(tf.expand_dims(
		    kl, -3), tf.zeros((1, tf.shape(kl)[0]), dtype=Latnet.FLOAT))
		return tf.reduce_sum(kl[0])

	def get_DKL_normal(mu, sigma2, prior_mu, prior_sigma2):
		"""
		Calculates KL divergence between two Mormal distributions, i.e.,
			KL(Normal(mu, sigma2), Normal(prior_mu, prior_sigma2))

		Args:
			posterior_mu: Tensor of shape N x N.
			posterior_sigma2: Tensor of shape N x N.
			prior_mu: Tensor of shape N x N.
			prior_sigma2: Tensor of shape N x N.

		Returns:
			Tensor of shape (), which contains the KL divergence.
		"""

		_half = tf.constant(0.5, dtype=Latnet.FLOAT)
		_munus_one = tf.constant(-1, dtype=Latnet.FLOAT)
		kl = _half*tf.add(tf.add(tf.math.log(tf.math.divide(prior_sigma2, sigma2))
			+ _munus_one, tf.math.divide(sigma2, prior_sigma2)),
			tf.math.divide(tf.square(tf.subtract(mu, prior_mu)), prior_sigma2))
		return tf.reduce_sum(kl)

	@staticmethod
	def get_priors(D, N_rf, N, p, lengthscale):
		"""
		Return parameters for prior distributions over W and A,
			W ~ Normal(prior_mu, prior_sigma2),
			A ~ Concrete(prior_alpha)

		Args:
			N: int. Number of nodes
			p: float. Sparciy of matrix A
		Returns:
			prior_mu: Tensor of size N x N.
			prior_sigma2:f Tensor of size N x N.
			prior_alpha: Tensor of size N x N.
		"""

		N_by_N = (N, N)
		N_by_2Nrf = (N, 2*N_rf)
		Nrf_by_D = (N_rf, D)

		prior_mu = tf.zeros(N_by_N, dtype=Latnet.FLOAT)
		prior_sigma2 = tf.multiply(
		    tf.cast(1. / (N * p), Latnet.FLOAT), tf.ones(N_by_N, dtype=Latnet.FLOAT))

		prior_mu_gamma = tf.zeros(N_by_2Nrf, dtype=Latnet.FLOAT)
		prior_sigma2_gamma = tf.ones(N_by_2Nrf, dtype=Latnet.FLOAT)

		prior_mu_omega = tf.zeros(Nrf_by_D, dtype=Latnet.FLOAT)
		prior_sigma2_omega = tf.ones(
		    Nrf_by_D, dtype=Latnet.FLOAT)/lengthscale/lengthscale

		prior_alpha = tf.multiply(
		    tf.cast(p / (1. - p), Latnet.FLOAT), tf.ones(N_by_N, dtype=Latnet.FLOAT))
		return prior_mu, prior_sigma2, prior_mu_gamma, prior_sigma2_gamma, prior_mu_omega, prior_sigma2_omega, prior_alpha

	@staticmethod
	def get_vairables(D, N_rf, t, Y,  init_sigma2_n, init_lengthscle, init_variance, init_p):
		"""
		Get tensor variables for the parameters that will be optimized (variational and hyper-parameters).
			W ~ Normal(mu, sigma2)
			A ~ Concrete(exp(log_alpha))

		Args:
			t: Tensor of shape T x 1
			Y: Tensor of shape T x N
			init_sigma2_g: init value of variance of connection noise.
			init_sigma2_n: init value of variance of observation nose.
			init_lengthscle: init value of kernel length-scale.
			init_variance: init value of kernel variance.

		Returns:
			log_sigma2_n: Tensor variable of shape ().
			log_sigma2_g: Tensor variable of shape ().
			mu: Tensor variable of shape N x N.
			log_sigma2: Tensor variable of shape N x N, which contains logarithm of sigma2.
			log_alpha: Tensor variable of shape N x N, which contains logarithm of alpha.
			log_lengthscale: Tensor variable of shape (), which contains logarithm of length-scale.
			log_variance: Tensor variable of shape (), which contains logarithm of variance.
		"""

		N = Y.shape[1]
		T = Y.shape[0]

		prior_mu, prior_sigma2, prior_mu_gamma, prior_sigma2_gamma, prior_mu_omega, prior_sigma2_omega, prior_alpha = Latnet.get_priors(
		    D, N_rf, N, init_p, init_lengthscle)

		mu = tf.Variable(prior_mu, dtype=Latnet.FLOAT)
		log_sigma2 = tf.Variable(tf.log(prior_sigma2), dtype=Latnet.FLOAT)

		mu_gamma = tf.Variable(prior_mu_gamma, dtype=Latnet.FLOAT)
		log_sigma2_gamma = tf.Variable(
		    tf.log(prior_sigma2_gamma), dtype=Latnet.FLOAT)

		mu_omega = tf.Variable(prior_mu_omega, dtype=Latnet.FLOAT)
		log_sigma2_omega = tf.Variable(
		    tf.log(prior_sigma2_omega), dtype=Latnet.FLOAT)

		log_alpha = tf.Variable(tf.log(prior_alpha), dtype=Latnet.FLOAT)

		log_sigma2_n = tf.Variable(
		    tf.log(tf.constant(init_sigma2_n, dtype=Latnet.FLOAT)), dtype=Latnet.FLOAT)

		log_lengthscale = tf.Variable(
		    tf.log(tf.constant(init_lengthscle, dtype=Latnet.FLOAT)), dtype=Latnet.FLOAT)
		log_variance = tf.Variable(
		    tf.log(tf.constant(init_variance, dtype=Latnet.FLOAT)), dtype=Latnet.FLOAT)

		return log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, log_lengthscale, log_variance

	@staticmethod
	def get_elbo(D, N_rf, t, Y, sigma2_n, mu, sigma2, mu_gamma, sigma2_gamma, mu_omega, sigma2_omega, alpha, lengthscale, variance,
				 prior_lambda_, posterior_lambda_, init_p, n_samples, approximate_kernel, seed=None):
		"""
		Calculates evidence lower bound (ELBO) for a set of posterior parameters using re-parametrization trick.

		posterior:    W ~ Normal(mu, sigma2), A ~ Normal(alpha)

		Args:
			t: Tensor with shape N x 1, which includes observation times.
			Y: Tensor wit shape T x N, which includes observation from N nodes at T time points.
			sigma2_n: Tensor with shape (). Observation noise variance.
			sigma2_g: Tensor with shape (). Connection noise variance.
			mu: Tensor with shape N x N.
			sigma2: Tensor with shape N x N.
			alpha: Tensor with shape N x N.
			lengthscale: double. Tensor with shape ().
			variance: double. Tensor with shape ().
			prior_lambda_: double. Tensor with shape ().
			posterior_lambda_: double. Tensor with shape ().
			n_samples: int. Tensor with shape (). Number of Monte-Carlo samples.
			seed: int. Seed used for random number generators.

		Returns:
			KL_normal: Tensor of shape () which contains KL-divergence between W_prior and W_posterior.
			KL_logistic:  Tensor of shape () which contains approximated KL-divergence between A_prior and A_posterior.
			ell: Tensor of shape () which contains approximated log-likelihood using Monte-Carlo samples.
		"""

		# number of nodes
		N = Y.shape[1]

		# number of observation per node.
		T = Y.shape[0]

		# number of Monte-Caro samples
		S = n_samples
		eps = tf.constant(1e-20, dtype=Latnet.FLOAT)
		Y = tf.constant(Y, dtype=Latnet.FLOAT)

		S_by_N_by_N = ((S, N, N))
		S_by_N_by_2Nrf = ((S, N, 2*N_rf))
		S_by_Nrf_by_D = ((S, N_rf, D))

		# sampling for W
		if seed is not None:
			z_W = tf.random_normal(S_by_N_by_N, dtype=Latnet.FLOAT, seed=seed)
		else:
			z_W = tf.random_normal(S_by_N_by_N, dtype=Latnet.FLOAT)
		W = tf.multiply(z_W, tf.sqrt(sigma2)) + mu
		W = tf.matrix_set_diag(W, tf.zeros((S, N), dtype=Latnet.FLOAT))

		# Gamma
		if seed is not None:
			z_G = tf.random_normal(S_by_N_by_2Nrf, dtype=Latnet.FLOAT, seed=seed)
		else:
			z_G = tf.random_normal(S_by_N_by_2Nrf, dtype=Latnet.FLOAT)
		Gamma = tf.multiply(z_G, tf.sqrt(sigma2_gamma)) + mu_gamma

		# Omega
		if seed is not None:
			z_O = tf.random_normal(S_by_Nrf_by_D, dtype=Latnet.FLOAT, seed=seed)
		else:
			z_O = tf.random_normal(S_by_Nrf_by_D, dtype=Latnet.FLOAT)
		Omega = tf.multiply(z_O, tf.sqrt(sigma2_omega)) + mu_omega

		# sampling for A
		if seed is not None:
			u = tf.random_uniform(S_by_N_by_N, minval=0, maxval=1,
			                      dtype=Latnet.FLOAT, seed=seed)
		else:
			u = tf.random_uniform(S_by_N_by_N, minval=0, maxval=1, dtype=Latnet.FLOAT)
		_A = tf.math.divide(
			tf.add(tf.log(alpha), tf.subtract(tf.log(u + eps),
			       tf.log(tf.constant(1.0, dtype=Latnet.FLOAT) - u + eps))),
			posterior_lambda_)
		A = tf.sigmoid(_A)
		A = tf.matrix_set_diag(A, tf.zeros((S, N), dtype=Latnet.FLOAT))

		prior_mu, prior_sigma2, prior_mu_gamma, prior_sigma2_gamma, prior_mu_omega, prior_sigma2_omega, prior_alpha = Latnet.get_priors(
		    D, N_rf, N, init_p, lengthscale)
		return Latnet.get_KL_normal(mu, sigma2, prior_mu, prior_sigma2), \
					Latnet.get_DKL_normal(mu_gamma, sigma2_gamma, prior_mu_gamma, prior_sigma2_gamma), \
					Latnet.get_DKL_normal(mu_omega, sigma2_omega, prior_mu_omega, prior_sigma2_omega), \
					Latnet.get_KL_logistic(_A, alpha, prior_lambda_, posterior_lambda_, prior_alpha), \
					Latnet.batch_ll_new(t, Y, sigma2_n, A, W, Gamma,
					                    Omega, variance, N, D, N_rf, T, S)

	@staticmethod
	def batch_ll_new(t, Y, sigma2_n, A, W, Gamma, Omega, variance, N, D, N_rf,  T, S):
		Omega_temp = tf.reshape(Omega, [S*N_rf, D])
		Fi_under = tf.reshape(tf.matmul(Omega_temp, tf.transpose(t)), [S, N_rf, T])
		Fi = tf.sqrt(tf.math.divide(variance, N_rf)) * \
		             tf.concat([tf.cos(Fi_under), tf.sin(Fi_under)], axis=1)
		Z = tf.matmul(Gamma, Fi)
		I = tf.constant(np.identity(N), dtype=Latnet.FLOAT)
		B = tf.multiply(A, W)
		v_current = tf.reshape(Z, [S, N, T])
		real_Y = tf.expand_dims(tf.transpose(Y), 0)
		exp_Y = tf.zeros((S, N, T), dtype=Latnet.FLOAT)
		exp_Y = v_current+exp_Y  # v
		v_current = tf.matmul(B, v_current)
		exp_Y = tf.add(v_current, exp_Y)  # v + Bv
		v_current = tf.matmul(B, v_current)
		exp_Y = tf.add(v_current, exp_Y)  # v + Bv +B2v

		real_Y = tf.expand_dims(tf.transpose(Y), 0)
		norm = tf.norm(exp_Y - real_Y, ord=2, axis=1)
		norm_sum_by_T = tf.reduce_sum(norm, axis=1)
		norm_sum_by_T_avg_by_S = tf.reduce_mean(norm_sum_by_T)
		_half = tf.constant(0.5, dtype=Latnet.FLOAT)
		_two = tf.constant(2.0, dtype=Latnet.FLOAT)
		_pi = np.pi
		_N = tf.constant(N, dtype=Latnet.FLOAT)
		res = -_half * N*T*tf.log(_two*_pi*sigma2_n) - \
		                          _half/sigma2_n*norm_sum_by_T_avg_by_S
		return res

	@staticmethod
	def converged(elbos):
		return False
		if len(elbos) < 5:
			return False
		last_5_elbos = elbos[-5:]
		last_4_percent_changes = [abs(last_5_elbos[i] - last_5_elbos[i+1]) / abs(
            last_5_elbos[i+1]) for i in range(len(last_5_elbos)-1)]
		if max(last_4_percent_changes) < 0.01:
			return True
		return False

	@staticmethod
	def optimize(s, D, N_rf, t, Y, targets, total_iters, local_iters, logger,
                 init_sigma2_n, init_lengthscle, init_variance, init_p,
                 lambda_prior, lambda_postetior, var_lr, hyp_lr, n_samples, approximate_kernel=True,
                 log_every=10, callback=None, seed=None, fix_kernel=False
                 ):
		logger.debug('\n\nParameters')
		logger.debug('init_sigma2_n {}'.format(init_sigma2_n))
		logger.debug('init_lengthscle {}'.format(init_lengthscle))
		logger.debug('init_variance {}'.format(init_variance))
		logger.debug('init_p {}'.format(init_p))
		logger.debug('lambda_prior {}'.format(lambda_prior))
		logger.debug('lambda_postetior {}'.format(lambda_postetior))
		logger.debug('var_lr {}'.format(var_lr))
		logger.debug('hyp_lr {}'.format(hyp_lr))
		logger.debug('n_samples {}\n\n'.format(n_samples))

		with tf.Session() as sess:
			var_opt, hyp_opt, elbo, kl_W, kl_A, kl_G, kl_O, ell, \
                log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, \
                log_lengthscale, log_variance, var_nans, hyp_nans = \
                Latnet.run_model(D, N_rf, tf.cast(t, Latnet.FLOAT), Y, init_sigma2_n, init_lengthscle, init_variance, init_p,
                                 lambda_prior, lambda_postetior, var_lr, hyp_lr, n_samples, approximate_kernel,  fixed_kernel=fix_kernel, seed=seed)

			init_op = tf.initializers.global_variables()
			sess.run(init_op)

            # current global iteration over optimization steps.
			_iter = 0

			logger.debug("prior lambda={:.3f}; posterior lambda={:.3f}; variat. learning rate={:.3f}; hyper learning rate={:.3f}".format(
                lambda_prior, lambda_postetior, var_lr, hyp_lr))

			while total_iters is None or _iter < total_iters:

				logger.debug("\nSUBJECT %d: ITERATION %d STARTED\n" %
                             (s, _iter))
                # optimizing variational parameters
				if 'var' in targets:
					elbos = []
					logger.debug("optimizing variational parameters")
					for i in range(0, local_iters['var']):
						try:
							output = sess.run(
                                [elbo, kl_W, kl_A, kl_G, kl_O, ell, var_opt, var_nans])
							if i % log_every == 0:
								elbos.append(output[0])
								logger.debug('\tlocal {:d} iter elbo: {:.0f} (KL W={:.0f}, KL A={:.0f}, KL G={:.0f},KL O={:.0f}, ell={:.0f}), {:d} nan in grads (err= {:d}). '.format(
                                    i, output[0], output[1], output[2], output[3], output[4], output[5], output[7], output[7] != 0))
								if Latnet.converged(elbos):
									break
						except OpError as e:
							logger.error(e.message)

                # optimizing hyper parameters
				if 'hyp' in targets:
					elbos = []
					logger.debug("optimizing hyper parameters")
					for i in range(0, local_iters['hyp']):
						try:
							output = sess.run(
                                [elbo, kl_W, kl_A, kl_G, kl_O, ell, hyp_opt, hyp_nans])
							if i % log_every == 0:
								elbos.append(output[0])
								logger.debug('\tlocal {:d} iter elbo: {:.0f} (KL W={:.0f}, KL A={:.0f}, KL G={:.0f},KL O={:.0f}, ell={:.0f}), {:d} nan in grads (err= {:d}). '.format(
                                    i, output[0], output[1], output[2], output[3], output[4], output[5], output[7], output[7] != 0))
								if Latnet.converged(elbos):
									break
						except OpError as e:
							logger.error(e.message)
				if callback is not None:
					sigma2_n_,  mu_, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale_, variance_ = \
                        sess.run((tf.exp(log_sigma2_n), mu, tf.exp(log_sigma2), mu_gamma, tf.exp(log_sigma2_gamma), mu_omega, tf.exp(
                            log_sigma2_omega), tf.exp(log_alpha), tf.exp(log_lengthscale), tf.exp(log_variance)))
					callback(alpha_, mu_, sigma2_, mu_gamma, sigma2_gamma_,
                             mu_omega, sigma2_omega_, sigma2_n_, lengthscale_, variance_)
				_iter += 1

			elbo_, sigma2_n_,  mu_, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale, variance = \
                sess.run((elbo, tf.exp(log_sigma2_n), mu, tf.exp(log_sigma2), mu_gamma, tf.exp(log_sigma2_gamma), mu_omega, tf.exp(
                    log_sigma2_omega), tf.exp(log_alpha), tf.exp(log_lengthscale), tf.exp(log_variance)))

		return elbo_, sigma2_n_, mu_, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale, variance
