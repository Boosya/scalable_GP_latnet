import csv
import random

# tensorboard --logdir graphs --host=127.0.0.1


__author__ = 'EG'

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def get_model_settings(flags):
    '''
     Parses model settings such as
     n_mc - number of samples for Monte Carlo sampling
     n_rf - number of random features for GP random feature expansion
     n_iterations - number of global iterations, 
        each of iteration consisting of variational learning steps  and hyperparameter learning steps
    n_var_steps - number of training steps during variational learning
    n_hyp_steps - number of training steps during hyperparameter learning
    display_step - log elbo and its components every display_step steps
    lr - learning rate for variational learning
    hlr - hyperparameter learning rate
    '''
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


def replace_nan_with_zero(w):
    return tf.compat.v1.where(tf.math.is_nan(w), tf.ones_like(w) * 0.0, w)


def contains_nan(w):
    for w_ in w:
        if tf.reduce_all(input_tensor=tf.math.is_nan(w_)) is None:
            return tf.reduce_all(input_tensor=tf.math.is_nan(w_))
    return tf.reduce_all(input_tensor=tf.math.is_nan(w_))


def get_optimizer(objective, trainables, learning_rate, tensorboard, max_global_norm=1.0):
    grads = tf.gradients(ys=objective, xs=trainables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
    grad_var_pairs = zip([replace_nan_with_zero(g) for g in grads], trainables)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    if tensorboard:
        grads_ = optimizer.compute_gradients(objective)
        for gradient, variable in grads_:
            tf.summary.histogram("gradients/" + variable.name, gradient)
    return optimizer.apply_gradients(grad_var_pairs), grads, contains_nan(grads)


def get_dkl_normal(mu, sigma2, prior_mu, prior_sigma2):
    kl = 0.5 * tf.add(
        tf.add(tf.math.log(tf.math.divide(prior_sigma2, sigma2)) - 1, tf.math.divide(sigma2, prior_sigma2)),
        tf.math.divide(tf.square(tf.subtract(mu, prior_mu)), prior_sigma2))
    return tf.reduce_sum(kl)


def get_kl_normal(posterior_mu, posterior_sigma2, prior_mu, prior_sigma2, dtype):
    kl = tf.add(tf.math.divide(tf.add(tf.square(tf.subtract(posterior_mu, prior_mu)), posterior_sigma2),
                               tf.multiply(tf.cast(2., dtype=dtype), prior_sigma2)),
                -0.5 + 0.5 * tf.math.log(prior_sigma2) - 0.5 * tf.math.log(posterior_sigma2))
    kl = tf.linalg.set_diag(tf.expand_dims(kl, -3), tf.zeros((1, tf.shape(kl)[0]), dtype=dtype))
    return tf.reduce_sum(kl[0])


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


def get_kl_logistic(X, posterior_alpha, prior_lambda_, posterior_lambda_, prior_alpha, dtype):
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
    logdiff = logp_logistic(X, posterior_alpha, posterior_lambda_, dtype) - logp_logistic(X, prior_alpha, prior_lambda_,
                                                                                          dtype)
    logdiff = tf.matrix_set_diag(logdiff, tf.zeros((tf.shape(logdiff)[0], tf.shape(logdiff)[1]),
                                                   dtype=dtype))  # set diagonal part to zero
    return tf.reduce_sum(tf.reduce_mean(logdiff, [0]))


def get_matrices(n_mc, n_nodes, n_rf, dtype, o_single_normal, mu_g, log_sigma2_g, mu_o, log_sigma2_o, mu_w,
                 log_sigma2_w, log_alpha, posterior_lambda_):
    # Gamma
    z_g = tf.random.normal((n_mc, n_nodes, 2 * n_rf), dtype=dtype)
    g = tf.multiply(z_g, tf.sqrt(tf.exp(log_sigma2_g))) + mu_g

    # Omega
    o = tf.multiply(o_single_normal, tf.sqrt(tf.exp(log_sigma2_o))) + mu_o

    # sampling for W
    z_w = tf.random.normal((n_mc, n_nodes, n_nodes), dtype=dtype)
    w = tf.multiply(z_w, tf.sqrt(tf.exp(log_sigma2_w))) + mu_w
    w = tf.linalg.set_diag(w, tf.zeros((n_mc, n_nodes), dtype=dtype))

    # sampling for A
    u = tf.random.uniform((n_mc, n_nodes, n_nodes), minval=0, maxval=1, dtype=dtype)
    eps = tf.constant(1e-20, dtype=dtype)
    _a = tf.math.divide(tf.add(log_alpha, tf.subtract(tf.math.log(u + eps), tf.math.log(1 - u + eps))),
                        posterior_lambda_)
    a = tf.sigmoid(_a)
    a = tf.linalg.set_diag(a, tf.zeros((n_mc, n_nodes), dtype=dtype))

    b = tf.multiply(a, w)
    return g, o, b, _a


def calculate_ell(n_mc, n_rf, n_nodes, dim, dtype, g, o, b, log_sigma2_n, log_variance, log_lengthscale, real_data,
                  inv_calculation, n_approx_terms, tensorboard=None):
    n_signals = real_data.shape[0]
    t = get_t(n_signals, dtype)
    z = get_z(n_signals, n_mc, n_rf, dim, g, o, log_variance, t, n_nodes)

    Kt = kernel(t, tf.exp(log_variance), tf.exp(log_lengthscale), n_signals, dtype)
    Kt_appr = tf.reduce_mean(tf.matmul(tf.transpose(z, perm=[0, 2, 1]), z), axis=0)

    exp_y = get_exp_y(inv_calculation, n_approx_terms, z, b, n_mc, n_nodes, n_signals, dtype)
    exp_y = tf.reduce_mean(exp_y, axis=0)
    exp_y = tf.transpose(exp_y)
    assert (exp_y.shape[0] == n_signals)
    assert (exp_y.shape[1] == n_nodes)

    ell, first_ell_part, second_ell_part, ell_norm = calculate_ell_(exp_y, real_data, dtype, n_nodes, n_signals,
                                                                    log_sigma2_n, tensorboard)
    return ell, exp_y, real_data, Kt, Kt_appr, first_ell_part, second_ell_part, ell_norm


def get_t(n_signals, dtype):
    # generate design matrix with shape (n_signals, n_dim)
    t = tf.cast(tf.expand_dims(tf.range(n_signals), 1), dtype)
    mean = tf.math.reduce_mean(t)
    std = tf.math.reduce_std(t)
    t = (t - mean) / std
    return t


def get_z(n_signals, n_mc, n_rf, dim, g, o, log_variance, t, n_nodes):
    o_temp = tf.reshape(o, [n_mc * n_rf, dim])
    fi_under = tf.reshape(tf.matmul(o_temp, tf.transpose(t)), [n_mc, n_rf, n_signals])
    fi = tf.sqrt(tf.math.divide(tf.exp(log_variance), n_rf)) * tf.concat([tf.cos(fi_under), tf.sin(fi_under)], axis=1)
    z = tf.matmul(g, fi)
    return z


def kernel(X, variance, lengthscale, T, dtype):
    """
        RBF - Radial basis kernel (aka Squared Exponential Kernel)
        Args:
            X: Tensor of shape T x 1 containing the location of data-points.
            variance: double. Tensor of shape (), determining the variance of kernel.
            lengthscale: double. Tensor of shape (), determining the lengthscale of the kernel.
            T: int. Number of data-points.
        Returns:
            : Tensor of shape T x T, in which element ij is: variance * exp(-(x_i _ x_j)^2/(2 * lengthscale ** 2))
            Note that a latent noise is added to the kernel for numerical stability.
        """
    dist = tf.reduce_sum(tf.square(X), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(tf.cast(2., dtype), tf.matmul(X, tf.transpose(X)))),
                      tf.transpose(dist))
    return tf.multiply(variance, tf.exp(
        tf.negative(tf.div(tf.abs(sq_dists), tf.multiply(tf.cast(2.0, dtype), tf.square(lengthscale)))))) + tf.constant(
        1e-5 * np.identity(T), dtype=dtype)


def get_exp_y(inv_calculation, n_approx_terms, z, b, n_mc, n_nodes, n_signals, dtype):
    exp_y = tf.zeros([n_mc, n_nodes, n_signals])
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
    return exp_y


def calculate_ell_(exp_y, real_y, dtype, n_nodes, n_signals, log_sigma2_n, tensorboard=None):
    ell_norm = get_norm(exp_y, real_y)
    _two_pi = tf.constant(6.28, dtype=dtype)
    _half = tf.constant(0.5, dtype=dtype)
    first_ell_part = - _half * n_nodes * tf.cast(n_signals, dtype=dtype) * tf.math.log(
        tf.multiply(_two_pi, tf.exp(log_sigma2_n)))
    second_ell_part = - _half * tf.divide(ell_norm, tf.exp(log_sigma2_n))
    ell = first_ell_part + second_ell_part
    return ell, first_ell_part, second_ell_part, ell_norm


def get_norm(exp_y, real_y):
    return tf.norm(exp_y - real_y, ord=2)


class ScalableLatnet:

    def __init__(self, flags, dim, train_data, test_data, true_conn, logger, subject, fold):
        self.subject = subject
        self.fold = fold
        self.FLOAT = tf.float64
        self.logger = logger
        self.dim = dim
        (self.n_signals, self.n_nodes) = train_data.shape
        self.train_data = tf.constant(train_data, dtype=self.FLOAT)
        self.test_data = tf.constant(test_data, dtype=self.FLOAT)
        self.true_conn = tf.reshape(tf.constant(true_conn, dtype=self.FLOAT), [self.n_nodes, self.n_nodes])

        # Set random seed for tensorflow and numpy operations
        tf.compat.v1.set_random_seed(flags.get_flag('seed'))
        np.random.seed(flags.get_flag('seed'))
        random.seed(flags.get_flag('seed'))

        self.tensorboard = flags.get_flag('tensorboard')

        self.n_mc, self.n_rf, self.n_iter, self.n_var_steps, self.n_hyp_steps, self.display_step, self.lr, self.hlr, self.inv_calculation, self.n_approx_terms, self.kl_g_weight = get_model_settings(
            flags)

        self.init_sigma2_n, self.init_variance, self.init_lengthscale, self.init_p, self.prior_lambda_, self.posterior_lambda_ = self.get_hyperparameters(
            flags)

        self.pr_mu_g, self.pr_sigma2_g, self.pr_mu_o, self.pr_mu_w, self.pr_sigma2_w, self.prior_alpha = self.initialize_priors()
        self.mu_g, self.log_sigma2_g, self.mu_o, self.o_single_normal, self.log_sigma2_n, self.log_variance, self.mu_w, self.log_sigma2_w, self.log_alpha = self.initialize_variables()
        self.log_lengthscale, self.pr_log_sigma2_o, self.log_sigma2_o = self.initialize_omega()

        # sample from posterior to get kl and ell terms
        self.g, self.o, self.b, self._a = get_matrices(self.n_mc, self.n_nodes, self.n_rf, self.FLOAT,
                                                       self.o_single_normal, self.mu_g, self.log_sigma2_g, self.mu_o,
                                                       self.log_sigma2_o, self.mu_w, self.log_sigma2_w, self.log_alpha,
                                                       self.posterior_lambda_)
        self.ell, self.pred_signals, self.real_signals, self.train_Kt, self.train_Kt_appr, self.first_ell_part, self.second_ell_part, self.ell_norm, = calculate_ell(
            n_mc=self.n_mc, n_rf=self.n_rf, n_nodes=self.n_nodes, dim=self.dim, dtype=self.FLOAT, g=self.g, o=self.o,
            b=self.b, log_sigma2_n=self.log_sigma2_n, log_variance=self.log_variance,
            log_lengthscale=self.log_lengthscale, real_data=self.train_data, inv_calculation=self.inv_calculation,
            n_approx_terms=self.n_approx_terms, tensorboard=self.tensorboard)
        self.mse_Kt_appr = tf.reduce_mean(tf.pow(self.train_Kt - self.train_Kt_appr, 2))
        _, self.test_pred_signals, self.test_real_signals, _, _, _, _, _ = calculate_ell(n_mc=self.n_mc, n_rf=self.n_rf,
                                                                                         n_nodes=self.n_nodes,
                                                                                         dim=self.dim, dtype=self.FLOAT,
                                                                                         g=self.g, o=self.o, b=self.b,
                                                                                         log_sigma2_n=self.log_sigma2_n,
                                                                                         log_variance=self.log_variance,
                                                                                         log_lengthscale=self.log_lengthscale,
                                                                                         real_data=self.test_data,
                                                                                         inv_calculation=self.inv_calculation,
                                                                                         n_approx_terms=self.n_approx_terms,
                                                                                         tensorboard=self.tensorboard)
        self.test_mse = tf.reduce_mean(tf.math.pow(self.test_real_signals - tf.transpose(self.test_pred_signals), 2))

        self.kl_o, self.kl_w, self.kl_g, self.kl_a = self.get_kl()
        # calculating ELBO
        # self.elbo = self.ell - self.kl_o - self.kl_w - self.kl_a - self.kl_g*self.kl_g_weight
        self.elbo = self.ell - self.kl_o - self.kl_w

        # get the operation for optimizing variational parameters
        self.var_opt, _, self.var_nans = get_optimizer(tf.negative(self.elbo),
                                                       [self.mu_g, self.log_sigma2_g, self.mu_o, self.log_sigma2_o,
                                                        self.mu_w, self.log_sigma2_w, self.log_alpha], self.lr,
                                                       self.tensorboard)
        # get the operation for optimizing hyper parameters
        self.hyp_opt, _, self.hyp_nans = get_optimizer(tf.negative(self.elbo),
                                                       [self.log_sigma2_n, self.log_variance, self.log_lengthscale],
                                                       self.hlr, self.tensorboard)

        # initialize variables
        self.init_op = tf.compat.v1.initializers.global_variables()
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 3
        config.inter_op_parallelism_threads = 3
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(self.init_op)
        if self.tensorboard:
            self.summaries_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter("graphs", self.sess.graph)

    def optimize(self, result_filenames):
        # current global iteration over optimization steps.
        _iter = 0
        self.global_step_id = 0
        while self.n_iter is None or _iter < self.n_iter:
            self.logger.debug("ITERATION {iter:d}".format(iter=_iter))
            self.run_step(_iter, self.n_var_steps, self.var_opt)
            self.run_step(_iter, self.n_hyp_steps, self.hyp_opt)
            _iter += 1
        self.run_variables()

        self.test_pred_signals_, self.test_real_signals_, self.test_mse_, self.true_conn_ = self.sess.run(
            [self.test_pred_signals, self.test_real_signals, self.test_mse, self.true_conn])
        self.auc_ = self.calculate_auc()

        self.logger.debug("Resulting MSE {mse:.2f}".format(mse=self.test_mse_))
        self.logger.debug("Resulting AUC {auc:.2f}".format(auc=self.auc_))
        self.write_results(result_filenames)
        return self.mu_w_, self.sigma2_w_, self.alpha_, self.mu_g_, self.sigma2_g_

    def run_step(self, gl_i, n_steps, opt):
        for i in range(0, n_steps):
            self.run_optimization(opt)
            if i % self.display_step == 0:
                self.log_optimization(gl_i, i)

    def run_optimization(self, opt):
        self.elbo_, self.ell_, self.first_ell_part_, self.second_ell_part_, self.ell_norm_, self.kl_g_, self.kl_o_, self.kl_w_, self.kl_a_, _, self.mse_Kt_appr_ = self.sess.run(
            [self.elbo, self.ell, self.first_ell_part, self.second_ell_part, self.ell_norm, self.kl_g, self.kl_o,
             self.kl_w, self.kl_a, opt, self.mse_Kt_appr])
        assert (self.ell_ < 0)
        if self.tensorboard:
            summary = self.sess.run([self.summaries_op])
            self.summary_writer.add_summary(summary, self.global_step_id)
        self.global_step_id += 1

    def run_variables(self):
        self.elbo_, self.pred_signals_, self.real_signals_, self.mu_w_, self.sigma2_w_, self.alpha_, self.mu_g_, self.sigma2_g_ = self.sess.run(
            (self.elbo, self.pred_signals, self.real_signals, self.mu_w, tf.exp(self.log_sigma2_w),
             tf.exp(self.log_alpha), self.mu_g, tf.exp(self.log_sigma2_g)))

    def log_optimization(self, gl_i, i):
        self.logger.debug(
            "{gl_i:d} local {i:d} iter: elbo={elbo_:.0f} (ell {ell_:.0f} (first {first:.0f}, second {second:.0f}, "
            "norm {norm:.0f}), kl_g {kl_g_:.1f}, kl_o {kl_o_:.1f}, kl_w {kl_w_:.1f}, kl_a {kl_a_:.1f}), Kt appr error "
            "{kt_err:.1f}".format(gl_i=gl_i, i=i, elbo_=self.elbo_, ell_=-self.ell_, kl_g_=self.kl_g_, kl_o_=self.kl_o_,
                                  kl_w_=self.kl_w_, kl_a_=self.kl_a_, kt_err=self.mse_Kt_appr_,
                                  first=self.first_ell_part_, second=self.second_ell_part_, norm=self.ell_norm_))

    def get_hyperparameters(self, flags):
        init_sigma2_n = flags.get_flag('init_sigma2_n')
        init_variance = flags.get_flag('init_variance')
        init_lengthscale = flags.get_flag('init_lengthscale')
        init_p = tf.constant(flags.get_flag('init_p'), dtype=self.FLOAT)
        posterior_lambda_ = tf.constant(flags.get_flag('posterior_lambda_'), dtype=self.FLOAT)
        prior_lambda_ = tf.constant(flags.get_flag('prior_lambda_'), dtype=self.FLOAT)
        return init_sigma2_n, init_variance, init_lengthscale, init_p, prior_lambda_, posterior_lambda_

    def initialize_priors(self):
        prior_mu_g = tf.zeros((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_sigma2_g = tf.ones((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_mu_o = tf.zeros((self.n_rf, self.dim), dtype=self.FLOAT)
        prior_mu_w = tf.zeros((self.n_nodes, self.n_nodes), dtype=self.FLOAT)
        prior_sigma2_w = tf.ones((self.n_nodes, self.n_nodes), dtype=self.FLOAT)
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

    def initialize_omega_old(self):
        init_lengthscale_matrix = tf.constant(self.init_lengthscale, dtype=self.FLOAT) * tf.ones((self.n_rf, self.dim),
                                                                                                 dtype=self.FLOAT)
        log_lengthscale = tf.Variable(2 * tf.math.log(init_lengthscale_matrix), dtype=self.FLOAT,
                                      name="log_lengthscale")
        pr_log_sigma2_o = -log_lengthscale
        log_sigma2_o = tf.Variable(pr_log_sigma2_o, dtype=self.FLOAT, name="log_sigma2_o")
        return log_lengthscale, pr_log_sigma2_o, log_sigma2_o

    def initialize_omega(self):
        log_lengthscale = tf.Variable(tf.math.log(tf.constant(self.init_lengthscale, dtype=self.FLOAT)))
        pr_log_sigma2_o = -2 * log_lengthscale * tf.ones((self.n_rf, self.dim), dtype=self.FLOAT)
        log_sigma2_o = tf.Variable(pr_log_sigma2_o, dtype=self.FLOAT, name="log_sigma2_o")
        return log_lengthscale, pr_log_sigma2_o, log_sigma2_o

    def get_kl(self):
        kl_o = get_dkl_normal(self.mu_o, tf.exp(self.log_sigma2_o), self.pr_mu_o, tf.exp(self.pr_log_sigma2_o))
        kl_w = get_kl_normal(self.mu_w, tf.exp(self.log_sigma2_w), self.pr_mu_w, self.pr_sigma2_w, self.FLOAT)
        kl_g = get_dkl_normal(self.mu_g, tf.exp(self.log_sigma2_g), self.pr_mu_g, self.pr_sigma2_g)
        kl_a = get_kl_logistic(self._a, tf.exp(self.log_alpha), self.prior_lambda_, self.posterior_lambda_,
                               self.prior_alpha, self.FLOAT)
        return kl_o, kl_w, kl_g, kl_a

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
            writer.writerow([self.subject, self.fold, rand_node, self.n_mc, self.n_rf, self.test_mse_, self.auc_])
            writer.writerows([])
            file.close()

    def calculate_auc(self):
        self.real_conn_ = self.get_bool_array_of_upper_and_lower_triangular(self.true_conn_)
        self.p_ = self.alpha_ / (1.0 + self.alpha_)
        self.pred_conn_ = self.get_array_of_upper_and_lower_triangular(self.p_)
        self.auc_ = roc_auc_score(self.real_conn_, self.pred_conn_)
        return self.auc_

    def get_bool_array_of_upper_and_lower_triangular(self, array):
        tri_upper_no_diag = np.triu(array, k=1)
        tri_lower_no_diag = np.tril(array, k=-1)
        final_matrix = np.logical_or(tri_upper_no_diag, tri_lower_no_diag)
        final_matrix_without_diag = final_matrix[~np.eye(final_matrix.shape[0], dtype=bool)].reshape(
            final_matrix.shape[0], -1)
        result = []
        for row in final_matrix_without_diag:
            result.extend(row)
        return result

    def get_array_of_upper_and_lower_triangular(self, array):
        tri_upper_no_diag = np.triu(array, k=1)
        tri_lower_no_diag = np.tril(array, k=-1)
        final_matrix = tri_upper_no_diag + tri_lower_no_diag
        final_matrix_without_diag = final_matrix[~np.eye(final_matrix.shape[0], dtype=bool)].reshape(
            final_matrix.shape[0], -1)
        result = []
        for row in final_matrix_without_diag:
            result.extend(row)
        return result
