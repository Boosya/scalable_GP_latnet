import csv
import random

__author__ = 'EG'

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def get_model_settings(flags):
    """
    Parses model settings

    :param flags: dict containing model settings
    :return:
        n_mc - number of samples for Monte Carlo sampling
        n_rf - number of random features for GP random feature expansion
        n_iterations - number of global iterations,
            each of iteration consisting of variational learning steps  and hyperparameter learning steps
        n_var_steps - number of training steps during variational learning
        n_hyp_steps - number of training steps during hyperparameter learning
        display_step - log elbo and its components every display_step steps
        lr - learning rate for variational learning
        hlr - hyperparameter learning rate
        inv_calculation - which method of finding (I-B)^-1 to use
        n_approx_terms - how many approximation terms to use in inverse approximation
    """
    n_mc = flags.get_flag('n_mc')
    n_rf = flags.get_flag('n_rff')
    n_iterations = flags.get_flag('n_iterations')
    n_var_steps = flags.get_flag('var_steps')
    n_hyp_steps = flags.get_flag('hyp_steps')
    display_step = flags.get_flag('display_step')
    lr = flags.get_flag('var_learning_rate')
    hlr = flags.get_flag('hyp_learning_rate')
    inv_calculation = flags.get_flag('inv_calculation')
    n_approx_terms = flags.get_flag('n_approx_terms')
    kl_g_weight = flags.get_flag('kl_g_weight')
    return n_mc, n_rf, n_iterations, n_var_steps, n_hyp_steps, display_step, lr, hlr, inv_calculation, n_approx_terms, kl_g_weight


def get_matrices(n_mc, n_nodes, n_rf, dtype, o_single_normal, mu_g, log_sigma2_g, mu_o, log_sigma2_o, mu_w,
                 log_sigma2_w, log_alpha, posterior_lambda_):
    """
    Samples from posterior of matrices used on ELBO calculation
    Gamma ~ N(mu_g, log_sigma2_g) - matrix of weights for random feature expansion
    Omega ~ N(mu_o, log_sigma2_o) - spectral frequencies matrix for random feature expansion
    W ~ N(mu_w, log_sigma2_w) - matrix showing the strength of connection between nodes
    A ~ concrete(alpha, posterior_lambda_) - matrix showing probability of connection between nodes

    :param n_mc: number of samples for mc sampling
    :param n_nodes: number of nodes in network
    :param n_rf: number of random features in random feature expansion
    :param dtype: type of data for Scalable latnet - float64
    :param o_single_normal: matrix with single normal realizations sampled for Omega in the beginning of the program -
        tensor of shape (n_mc, n_rf, dim)
    :param mu_g: variational parameter - mean of Gamma - tensor of shape (n_nodes, 2n_rf)
    :param log_sigma2_g: variational parameter - variance of Gamma - tensor of shape (n_nodes, 2n_rf)
    :param mu_o: variational parameter - mean of Omega - tensor of shape (n_rf, dim)
    :param log_sigma2_o: variational parameter - variance of Omega - tensor of shape (n_rf, dim)
    :param mu_w: variational parameter - mean of W - tensor of shape (n_nodes, n_nodes)
    :param log_sigma2_w: variational parameter - variance of W - tensor of shape (n_nodes, n_nodes)
    :param log_alpha: log of parameter alpha for Concrete distribution over A_ij - tensor of shape (n_nodes, n_nodes)
    :param posterior_lambda_: lambda of the Concrete distr. used for posterior over `A_ij' matrix - tensor of shape ()
    :return: matrices Gamma (g), Omega (o), B (b) - element wise A*B, _A
    """
    # Gamma
    z_g = tf.random.normal((n_mc, n_nodes, 2 * n_rf), dtype=dtype)
    g = tf.multiply(z_g, tf.sqrt(tf.exp(log_sigma2_g))) + mu_g

    # Omega
    o = tf.multiply(o_single_normal, tf.sqrt(tf.exp(log_sigma2_o))) + mu_o

    # W
    z_w = tf.random.normal((n_mc, n_nodes, n_nodes), dtype=dtype)
    w = tf.multiply(z_w, tf.sqrt(tf.exp(log_sigma2_w))) + mu_w
    w = tf.linalg.set_diag(w, tf.zeros((n_mc, n_nodes), dtype=dtype))

    # A
    u = tf.random.uniform((n_mc, n_nodes, n_nodes), minval=0, maxval=1, dtype=dtype)
    eps = tf.constant(1e-20, dtype=dtype)
    _a = tf.math.divide(tf.add(log_alpha, tf.subtract(tf.math.log(u + eps), tf.math.log(1 - u + eps))),
                        posterior_lambda_)
    a = tf.sigmoid(_a)
    a = tf.linalg.set_diag(a, tf.zeros((n_mc, n_nodes), dtype=dtype))

    # B
    b = tf.multiply(a, w)
    return g, o, b, _a


def calculate_ell(n_mc, n_rf, n_nodes, dim, dtype, g, o, b, log_sigma2_n, log_variance, log_lengthscale, real_data,
                  inv_calculation, n_approx_terms):
    """
    Calculates approximated expected log-likelihood term, expected signals and approximated kernel values

    :param n_mc: number of samples for mc sampling
    :param n_rf: number of random features in random feature expansion
    :param n_nodes: number of nodes in network
    :param dim: dimensionality of the data
    :param dtype: type of data for Scalable latnet - float64
    :param g: Gamma matrix  - tensor of shape (n_mc, n_nodes, 2n_rf)
    :param o: Omega (spectral frequency) matrix - tensor of shape (n_mc, n_rf, dim)
    :param b: matrix of strength of connections between nodes - tensor of shape (n_mc, n_nodes, n_nodes)
    :param log_sigma2_n: observation noise of the signals - tensor with shape ()
    :param log_variance: variance of kernel - tensor with shape ()
    :param log_lengthscale: lengthscale of kernel - tensor with shape ()
    :param real_data: signals observed - tensor with shape (n_nodes, n_signals)
    :param inv_calculation: flag showing method for calculating inverse of (I-B)^-1 to use ['approx','matrix_inverse']
    :param n_approx_terms: number of terms in approx calculations of inverse
    :return:
        ell - approximated expected log-likelihood term - tensor of shape ()
        exp_y - expected signals based on model - tensor of shape (n_nodes, n_signals)
        real_data - signals observed - tensor with shape (n_nodes, n_signals)
        Kt - kernel values at observation times t - tensor of shape (n_signals, n_signals)
        Kt_appr - approximated kernel values at observation times t - tensor of shape (n_signals, n_signals)
        ell_1 - first part of ell term - tensor of shape ()
        ell_2 - second part of ell term - tensor of shape ()
    """
    n_signals = real_data.shape[0]
    assert (n_nodes == real_data.shape[1])
    # t - normalized design matrix - tensor with shape (n_signals, n_dim)
    t = get_t(n_signals, dtype)
    # z - approximation of GP - tensor of shape (n_mc, n_nodes, n_signals)
    z = get_z(n_signals, n_mc, n_rf, dim, g, o, log_variance, t)

    # exp_y - expected signals based on model - tensor of shape (n_mc, n_nodes, n_signals)
    exp_y = get_exp_y(inv_calculation, n_approx_terms, z, b, n_mc, n_nodes, n_signals, dtype)

    d = exp_y - tf.transpose(real_data)
    norm = tf.pow(d, 2)
    norm_sum_by_t = tf.reduce_sum(norm, axis=[1, 2])
    norm_sum_by_t_avg_by_s = tf.reduce_mean(norm_sum_by_t)

    _two_pi = tf.constant(6.28, dtype=dtype)
    _half = tf.constant(0.5, dtype=dtype)
    first_part_ell = - _half * n_nodes * tf.cast(n_signals, dtype=dtype) * tf.math.log(
        tf.multiply(_two_pi, tf.exp(log_sigma2_n)))

    second_part_ell = - _half * tf.divide(norm_sum_by_t_avg_by_s, tf.exp(log_sigma2_n))
    ell = first_part_ell + second_part_ell

    return ell, tf.reduce_mean(exp_y, axis=0), real_data, first_part_ell, second_part_ell


def get_t(n_signals, dtype):
    """
    Get design matrix of size n_signals

    :param n_signals:  number of signal observations per node
    :param dtype: type of data for Scalable latnet - float64
    :return: t - normalized design matrix - tensor with shape (n_signals, n_dim)
    """
    t = tf.cast(tf.expand_dims(tf.range(n_signals), 1), dtype)
    mean = tf.math.reduce_mean(t)
    std = tf.math.reduce_std(t)
    t = (t - mean) / std
    return t


def get_z(n_signals, n_mc, n_rf, dim, g, o, log_variance, t):
    """
    Get approximation of GP using random feature expansion in form
    Z = Gamma * Fi, Fi = sqrt(variance / n_rf)[cos(Omega*t), sin(Omega*t)]

    :param n_signals: number of signal observations per node
    :param n_mc: number of samples for mc sampling
    :param n_rf: number of random features in random feature expansion
    :param dim: dimensionality of the data
    :param g: Gamma matrix  - tensor of shape (n_mc, n_nodes, 2n_rf)
    :param o: Omega (spectral frequency) matrix - tensor of shape (n_mc, n_rf, dim)
    :param log_variance: - kernel variance - tensor of shape ()
    :param t: design matrix containing normalized time points
    :return: z - approximation of GP - tensor of shape (n_mc, n_nodes, n_signals)
    """
    o_temp = tf.reshape(o, [n_mc * n_rf, dim])
    fi_under = tf.reshape(tf.matmul(o_temp, tf.transpose(t)), [n_mc, n_rf, n_signals])
    # get fi - random feature projections
    fi = tf.sqrt(tf.math.divide(tf.exp(log_variance), n_rf)) * tf.concat([tf.cos(fi_under), tf.sin(fi_under)], axis=1)
    # multiply projections on weights
    z = tf.matmul(g, fi)
    n_nodes = g.shape[1]
    assert (n_mc == z.shape[0])
    assert (n_nodes == z.shape[1])
    assert (n_signals == z.shape[2])
    return z


def get_exp_y(inv_calculation, n_approx_terms, z, b, n_mc, n_nodes, n_signals, dtype):
    """
    Get expected signal as follows: exp_y = (I-B)^-1*Z

    :param inv_calculation: flag showing method for calculating inverse of (I-B)^-1 to use ['approx','matrix_inverse']
    :param n_approx_terms: number of terms in approx calculations of inverse
    :param z: approximation of GP - tensor of shape (n_signals, n_nodes)
    :param b: matrix of strength of connections between nodes - tensor of shape (n_mc, n_nodes, n_nodes)
    :param n_mc: number of samples for mc sampling
    :param n_nodes: number of nodes in network
    :param n_signals: number of signal observations per node
    :param dtype: type of data for Scalable latnet - float64
    :return: exp_y - expected signals based on model - tensor of shape (n_nodes, n_signals)
    """
    exp_y = tf.zeros([n_mc, n_nodes, n_signals], dtype=dtype)
    if inv_calculation == "approx":
        v_current = tf.reshape(z, [n_mc, n_nodes, n_signals])
        exp_y = v_current
        for _ in range(n_approx_terms):
            v_current = tf.matmul(b, v_current)
            exp_y = tf.add(v_current, exp_y)
    elif inv_calculation == "matrix_inverse":
        identity = tf.constant(np.identity(n_nodes), dtype=dtype)
        identity_minus_b = tf.subtract(identity, b)
        identity_minus_b_inverse = tf.matrix_inverse(identity_minus_b)
        exp_y = tf.matmul(identity_minus_b_inverse, z)
    assert (exp_y.shape[0] == n_mc)
    assert (exp_y.shape[1] == n_nodes)
    assert (exp_y.shape[2] == n_signals)
    return exp_y


def get_kl_normal(posterior_mu, posterior_sigma2, prior_mu, prior_sigma2, dtype):
    kl = tf.add(tf.math.divide(tf.add(tf.square(tf.subtract(posterior_mu, prior_mu)), posterior_sigma2),
                               tf.multiply(tf.cast(2., dtype=dtype), prior_sigma2)),
                -0.5 + 0.5 * tf.math.log(prior_sigma2) - 0.5 * tf.math.log(posterior_sigma2))
    kl = tf.linalg.set_diag(tf.expand_dims(kl, -3), tf.zeros((1, tf.shape(kl)[0]), dtype=dtype))
    return tf.reduce_sum(kl[0])


def get_kl_logistic(x, posterior_alpha, prior_lambda_, posterior_lambda_, prior_alpha, dtype):
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
    logdiff = logp_logistic(x, posterior_alpha, posterior_lambda_, dtype) - logp_logistic(x, prior_alpha, prior_lambda_,
                                                                                          dtype)
    logdiff = tf.matrix_set_diag(logdiff, tf.zeros((tf.shape(logdiff)[0], tf.shape(logdiff)[1]),
                                                   dtype=dtype))  # set diagonal part to zero
    return tf.reduce_sum(tf.reduce_mean(logdiff, [0]))


def logp_logistic(X, alpha, lambda_, dtype):
    """
    Logarithm of Concrete distribution with parameter `alpha' and hyper-parameter `lambda_' at points `X', i.e.,
        Concrete(X;alpha, lambda_)

    Args:
        X: Tensor of shape S x N x N. Locations at which the distribution will be calculated.
        alpha:  Tensor of shape N x N. To be written.
        lambda_: double. Tensor of shape ().

    Returns:
        :param X:
        :param alpha:
        :param lambda_:
        :param dtype:
        : A tensor of shape S x N x N. Element ijk is:
            log lambda_  - lambda_ * X_ijk + log alpha_jk - 2 log (1 + exp (-lambda_ * X_ijk + log alpha_jk))

    """

    mu = tf.log(alpha)
    return tf.subtract(tf.add(tf.subtract(tf.log(lambda_), tf.multiply(lambda_, X)), mu),
                       tf.multiply(tf.constant(2.0, dtype=dtype), tf.log(tf.add(tf.constant(1.0, dtype=dtype), tf.exp(
                           tf.add(tf.negative(tf.multiply(lambda_, X)), mu))))))


def replace_nan_with_zero(w):
    return tf.compat.v1.where(tf.math.is_nan(w), tf.ones_like(w) * 0.0, w)


def contains_nan(w):
    for w_ in w:
        if tf.reduce_all(input_tensor=tf.math.is_nan(w_)) is None:
            return tf.reduce_all(input_tensor=tf.math.is_nan(w_))
    return tf.reduce_all(input_tensor=tf.math.is_nan(w_))


def get_optimizer(objective, trainables, learning_rate, max_global_norm=1.0):
    grads = tf.gradients(ys=objective, xs=trainables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    for gradient, variable in optimizer.compute_gradients(objective):
        print(gradient,variable.name)
        tf.summary.histogram("gradients/" + variable.name, gradient)
        tf.summary.histogram("variables/" + variable.name, variable)

    grad_var_pairs = zip([replace_nan_with_zero(g) for g in grads], trainables)

    return optimizer.apply_gradients(grad_var_pairs), grads, contains_nan(grads)


def get_dkl_normal(mu, sigma2, prior_mu, prior_sigma2):
    kl = 0.5 * tf.add(
        tf.add(tf.math.log(tf.math.divide(prior_sigma2, sigma2)) - 1, tf.math.divide(sigma2, prior_sigma2)),
        tf.math.divide(tf.square(tf.subtract(mu, prior_mu)), prior_sigma2))
    return tf.reduce_sum(kl)


def get_bool_array_of_upper_and_lower_triangular(array):
    tri_upper_no_diag = np.triu(array, k=1)
    tri_lower_no_diag = np.tril(array, k=-1)
    final_matrix = np.logical_or(tri_upper_no_diag, tri_lower_no_diag)
    final_matrix_without_diag = final_matrix[~np.eye(final_matrix.shape[0], dtype=bool)].reshape(final_matrix.shape[0],
                                                                                                 -1)
    result = []
    for row in final_matrix_without_diag:
        result.extend(row)
    return result


def get_array_of_upper_and_lower_triangular(array):
    tri_upper_no_diag = np.triu(array, k=1)
    tri_lower_no_diag = np.tril(array, k=-1)
    final_matrix = tri_upper_no_diag + tri_lower_no_diag
    final_matrix_without_diag = final_matrix[~np.eye(final_matrix.shape[0], dtype=bool)].reshape(final_matrix.shape[0],
                                                                                                 -1)
    result = []
    for row in final_matrix_without_diag:
        result.extend(row)
    return result


class ScalableLatnet:

    def __init__(self, flags, subject, dim, train_data, fold, test_data, true_conn, logger):
        self.FLOAT = tf.float64
        self.logger = logger
        self.subject = subject
        self.dim = dim
        (self.n_signals, self.n_nodes) = train_data.shape
        self.train_data = tf.constant(train_data)
        self.fold = fold
        self.test_data = tf.constant(test_data)
        self.true_conn = tf.reshape(tf.constant(true_conn, dtype=self.FLOAT), [self.n_nodes, self.n_nodes])

        # Set random seed for tensorflow and numpy operations
        tf.compat.v1.set_random_seed(flags.get_flag('seed'))
        np.random.seed(flags.get_flag('seed'))
        random.seed(flags.get_flag('seed'))

        self.tensorboard = flags.get_flag('tensorboard')

        # Parse model parameters and specifications
        self.n_mc, self.n_rf, self.n_iter, self.n_var_steps, self.n_hyp_steps, self.display_step, self.lr, self.hlr, self.inv_calculation, self.n_approx_terms, self.kl_g_weight = get_model_settings(
            flags)

        # Parse hyperparameters and init values
        self.init_sigma2_n, self.init_variance, self.init_lengthscale, self.init_p, self.prior_lambda_, self.posterior_lambda_ = self.get_hyperparameters(
            flags)

        ######################
        # Building the graph

        # Initialize prior over variables
        self.pr_mu_g, self.pr_sigma2_g, self.pr_mu_o, self.pr_mu_w, self.pr_sigma2_w, self.prior_alpha = self.initialize_priors()

        # Initialize variables
        self.mu_g, self.log_sigma2_g, self.mu_o, self.o_single_normal, self.log_sigma2_n, self.log_variance, self.mu_w, self.log_sigma2_w, self.log_alpha = self.initialize_variables()

        # Initialize priors and variables for things connected to omega
        self.log_lengthscale, self.pr_log_sigma2_o, self.log_sigma2_o = self.initialize_omega()

        # Sample from posterior to get kl and ell terms
        self.g, self.o, self.b, self._a = get_matrices(self.n_mc, self.n_nodes, self.n_rf, self.FLOAT,
                                                       self.o_single_normal, self.mu_g, self.log_sigma2_g, self.mu_o,
                                                       self.log_sigma2_o, self.mu_w, self.log_sigma2_w, self.log_alpha,
                                                       self.posterior_lambda_)

        # Calculate ell term, predicted signals
        self.ell, self.pred_signals, self.real_signals, self.ell_1, self.ell_2 = calculate_ell(n_mc=self.n_mc,
                                                                                               n_rf=self.n_rf,
                                                                                               n_nodes=self.n_nodes,
                                                                                               dim=self.dim,
                                                                                               dtype=self.FLOAT,
                                                                                               g=self.g, o=self.o,
                                                                                               b=self.b,
                                                                                               log_sigma2_n=self.log_sigma2_n,
                                                                                               log_variance=self.log_variance,
                                                                                               log_lengthscale=self.log_lengthscale,
                                                                                               real_data=self.train_data,
                                                                                               inv_calculation=self.inv_calculation,
                                                                                               n_approx_terms=self.n_approx_terms)

        # Calculate test signal prediction
        _, self.test_pred_signals, self.test_real_signals, _, _ = calculate_ell(n_mc=self.n_mc, n_rf=self.n_rf,
                                                                                n_nodes=self.n_nodes, dim=self.dim,
                                                                                dtype=self.FLOAT, g=self.g, o=self.o,
                                                                                b=self.b,
                                                                                log_sigma2_n=self.log_sigma2_n,
                                                                                log_variance=self.log_variance,
                                                                                log_lengthscale=self.log_lengthscale,
                                                                                real_data=self.test_data,
                                                                                inv_calculation=self.inv_calculation,
                                                                                n_approx_terms=self.n_approx_terms)

        # Get KL terms
        self.kl_o, self.kl_w, self.kl_g, self.kl_a = self.get_kl()

        # Calculate ELBO
        self.elbo = self.ell - self.kl_o - self.kl_w - self.kl_a - self.kl_g

        # Get the operation for optimizing variational parameters
        self.var_opt, _, self.var_nans = get_optimizer(tf.negative(self.elbo),
                                                       [self.mu_g, self.log_sigma2_g, self.mu_o, self.log_sigma2_o,
                                                        self.mu_w, self.log_sigma2_w, self.log_alpha], self.lr)
        # Get the operation for optimizing hyper parameters
        self.hyp_opt, _, self.hyp_nans = get_optimizer(tf.negative(self.elbo),
                                                       [self.log_sigma2_n, self.log_variance, self.log_lengthscale],
                                                       self.hlr)

        ######################
        # Initializing session

        # Initialize variables
        self.init_op = tf.compat.v1.initializers.global_variables()
        # config = tf.ConfigProto()
        # config.inter_op_parallelism_threads = 1
        # config.intra_op_parallelism_threads = 1
        self.sess = tf.compat.v1.Session()
        self.sess.run(self.init_op)
        self.summaries_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter("graphs", self.sess.graph)

    def optimize(self, result_filenames):
        ######################
        # Learning

        _iter = 0  # current global iteration over optimization steps
        self.global_step_id = 0
        while self.n_iter is None or _iter < self.n_iter:
            self.logger.debug("ITERATION {iter:d}".format(iter=_iter))
            self.run_step(_iter, self.n_var_steps, self.var_opt)
            self.run_step(_iter, self.n_hyp_steps, self.hyp_opt)
            _iter += 1
        self.run_variables()

        self.test_pred_signals_, self.test_real_signals_, self.true_conn_ = self.sess.run(
            [self.test_pred_signals, self.test_real_signals, self.true_conn])
        self.auc_ = self.calculate_auc()

        self.logger.debug("Resulting AUC {auc:.2f}".format(auc=self.auc_))
        self.write_results(result_filenames)
        return self.mu_w_, self.sigma2_w_, self.alpha_, self.mu_g_, self.sigma2_g_, self.mu_o_, self.sigma2_o_, self.sigma2_n_, self.variance_, self.lengthscale_

    def run_step(self, gl_i, n_steps, opt):
        for i in range(0, n_steps):
            self.run_optimization(opt)
            if i % self.display_step == 0:
                self.log_optimization(gl_i, i)

    def run_optimization(self, opt):
        self.elbo_, self.ell_, self.ell_1_, self.ell_2_, self.kl_g_, self.kl_o_, self.kl_w_, self.kl_a_, _, summary = self.sess.run(
            [self.elbo, self.ell, self.ell_1, self.ell_2, self.kl_g, self.kl_o, self.kl_w, self.kl_a, opt,
             self.summaries_op])
        self.summary_writer.add_summary(summary, self.global_step_id)
        self.global_step_id += 1

    def run_variables(self):
        self.elbo_, self.pred_signals_, self.real_signals_, self.mu_w_, self.sigma2_w_, self.alpha_, self.mu_g_, self.sigma2_g_, self.mu_o_, self.sigma2_o_, self.sigma2_n_, self.variance_, self.lengthscale_ = self.sess.run(
            (self.elbo, self.pred_signals, self.real_signals, self.mu_w, tf.exp(self.log_sigma2_w),
             tf.exp(self.log_alpha), self.mu_g, tf.exp(self.log_sigma2_g), self.mu_o, tf.exp(self.log_sigma2_o),
             tf.exp(self.log_sigma2_n), tf.exp(self.log_variance), tf.exp(self.log_lengthscale)))

    def log_optimization(self, gl_i, i):
        self.logger.debug("{gl_i:d} local {i:d} iter: elbo={elbo_:.0f} (ell {ell_:.0f} ({ell_1_:.0f},{ell_2_:.0f}), "
                          "kl_g {kl_g_:.1f}, kl_o {kl_o_:.1f}, kl_w {kl_w_:.1f}, kl_a {kl_a_:.1f})".format(gl_i=gl_i,
                                                                                                           i=i,
                                                                                                           elbo_=self.elbo_,
                                                                                                           ell_=-self.ell_,
                                                                                                           ell_1_=self.ell_1_,
                                                                                                           ell_2_=self.ell_2_,
                                                                                                           kl_g_=self.kl_g_,
                                                                                                           kl_o_=self.kl_o_,
                                                                                                           kl_w_=self.kl_w_,
                                                                                                           kl_a_=self.kl_a_))

    def get_hyperparameters(self, flags):
        init_sigma2_n = flags.get_flag('init_sigma2_n')
        init_variance = flags.get_flag('init_variance')
        init_lengthscale = tf.constant(flags.get_flag('init_lengthscale'), dtype=self.FLOAT)
        init_p = tf.constant(flags.get_flag('init_p'), dtype=self.FLOAT)
        posterior_lambda_ = tf.constant(flags.get_flag('posterior_lambda_'), dtype=self.FLOAT)
        prior_lambda_ = tf.constant(flags.get_flag('prior_lambda_'), dtype=self.FLOAT)
        return init_sigma2_n, init_variance, init_lengthscale, init_p, prior_lambda_, posterior_lambda_

    def initialize_priors(self):
        prior_mu_g = tf.zeros((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_sigma2_g = tf.ones((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_mu_o = tf.zeros((self.n_rf, self.dim), dtype=self.FLOAT)
        prior_mu_w = tf.zeros((self.n_nodes, self.n_nodes), dtype=self.FLOAT)
        prior_sigma2_w = 2 / self.n_nodes * tf.ones((self.n_nodes, self.n_nodes), dtype=self.FLOAT)
        prior_alpha = tf.multiply(tf.cast(self.init_p / (1. - self.init_p), self.FLOAT),
                                  tf.ones((self.n_nodes, self.n_nodes), dtype=self.FLOAT))
        return prior_mu_g, prior_sigma2_g, prior_mu_o, prior_mu_w, prior_sigma2_w, prior_alpha

    def initialize_variables(self):
        mu_g = tf.Variable(self.pr_mu_g, dtype=self.FLOAT, name="mu_g")
        log_sigma2_g = tf.Variable(tf.math.log(self.pr_sigma2_g), dtype=self.FLOAT, name="log_sigma2_g")
        mu_o = tf.Variable(self.pr_mu_o, dtype=self.FLOAT, name="mu_o")
        o_single_normal = np.random.normal(loc=0, scale=1, size=(self.n_mc, self.n_rf, self.dim))
        log_sigma_2_n = tf.Variable(tf.math.log(tf.cast(self.init_sigma2_n, dtype=self.FLOAT)), dtype=self.FLOAT,
                                    name="log_sigma_2_n")
        log_variance = tf.Variable(tf.math.log(tf.cast(self.init_variance, dtype=self.FLOAT)), dtype=self.FLOAT,
                                   name="log_variance")
        mu_w = tf.Variable(self.pr_mu_w, dtype=self.FLOAT, name="mu_w")
        log_sigma2_w = tf.Variable(tf.math.log(self.pr_sigma2_w), dtype=self.FLOAT, name="log_sigma2_w")
        log_alpha = tf.Variable(tf.math.log(self.prior_alpha), dtype=self.FLOAT, name="log_alpha")
        return mu_g, log_sigma2_g, mu_o, o_single_normal, log_sigma_2_n, log_variance, mu_w, log_sigma2_w, log_alpha

    def initialize_omega(self):
        log_lengthscale = tf.Variable(tf.math.log(self.init_lengthscale), dtype=self.FLOAT, name="log_lengthscale")
        pr_log_sigma2_o = -2 * log_lengthscale
        log_sigma2_o = tf.Variable(-2*tf.math.log(self.init_lengthscale), dtype=self.FLOAT, name="log_sigma_o")
        return log_lengthscale, pr_log_sigma2_o, log_sigma2_o

    def get_kl(self):
        kl_o = get_dkl_normal(self.mu_o, tf.exp(self.log_sigma2_o), self.pr_mu_o, tf.exp(self.pr_log_sigma2_o))
        kl_w = get_kl_normal(self.mu_w, tf.exp(self.log_sigma2_w), self.pr_mu_w, self.pr_sigma2_w, self.FLOAT)
        kl_g = get_dkl_normal(self.mu_g, tf.exp(self.log_sigma2_g), self.pr_mu_g, self.pr_sigma2_g)
        kl_a = get_kl_logistic(self._a, tf.exp(self.log_alpha), self.prior_lambda_, self.posterior_lambda_,
                               self.prior_alpha, self.FLOAT)
        return kl_o, kl_w, kl_g, kl_a

    def calculate_auc(self):
        self.real_conn_ = get_bool_array_of_upper_and_lower_triangular(self.true_conn_)
        self.p_ = self.alpha_ / (1.0 + self.alpha_)
        self.pred_conn_ = get_array_of_upper_and_lower_triangular(self.p_)
        self.auc_ = roc_auc_score(self.real_conn_, self.pred_conn_)
        return self.auc_

    def write_results(self, result_filenames):
        rand_node = random.randint(1, self.n_nodes - 1)
        train_real_row = self.real_signals_[:, rand_node]
        train_pred_row = self.pred_signals_[rand_node, :]
        with open(result_filenames + 'train_signal_prediction.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.subject, self.fold, rand_node, self.n_mc, self.n_rf])
            writer.writerow(train_real_row)
            writer.writerow(train_pred_row)
            writer.writerow([])
            file.close()

        with open(result_filenames + 'test_signal_prediction_all.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.subject, self.fold, self.n_mc, self.n_rf])
            writer.writerows(self.test_real_signals_)
            writer.writerows(self.test_pred_signals_)
            writer.writerows([])
            file.close()
        test_real_row = self.test_real_signals_[:, rand_node]
        test_pred_row = self.test_pred_signals_[rand_node, :]
        with open(result_filenames + 'test_signal_prediction.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.subject, self.fold, rand_node, self.n_mc, self.n_rf])
            writer.writerow(test_real_row)
            writer.writerow(test_pred_row)
            writer.writerows([])
            file.close()
        with open(result_filenames + 'result.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.subject, self.fold, rand_node, self.n_mc, self.n_rf, self.auc_])
            writer.writerows([])
            file.close()
