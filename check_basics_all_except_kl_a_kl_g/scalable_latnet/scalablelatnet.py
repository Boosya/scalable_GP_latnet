import csv
import random

__author__ = 'EG'

import numpy as np
import tensorflow as tf


def get_model_settings(flags):
    n_mc = flags.get_flag('n_mc')
    n_rf = flags.get_flag('n_rff')
    n_iterations = flags.get_flag('n_iterations')
    n_var_steps = flags.get_flag('var_steps')
    n_hyp_steps = flags.get_flag('hyp_steps')
    display_step = flags.get_flag('display_step')
    lr = flags.get_flag('var_learning_rate')
    hlr = flags.get_flag('hyp_learning_rate')
    return n_mc, n_rf, n_iterations, n_var_steps, n_hyp_steps, display_step, lr, hlr


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
    grad_var_pairs = zip([replace_nan_with_zero(g) for g in grads], trainables)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
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


class ScalableLatnet:

    def __init__(self, flags, subject, dim, train_data, validation_data, test_data, true_conn, logger):
        self.FLOAT = tf.float64
        self.logger = logger
        self.subject = subject
        self.dim = dim
        (self.n_signals, self.n_nodes) = train_data.shape
        self.train_data = tf.constant(train_data, dtype=self.FLOAT)
        self.validation_data = tf.constant(validation_data, dtype=self.FLOAT)
        self.test_data = tf.constant(test_data, dtype=self.FLOAT)
        self.true_conn = tf.reshape(tf.constant(true_conn, dtype=self.FLOAT), [self.n_nodes, self.n_nodes])

        # Set random seed for tensorflow and numpy operations
        tf.compat.v1.set_random_seed(flags.get_flag('seed'))
        np.random.seed(flags.get_flag('seed'))
        random.seed(flags.get_flag('seed'))

        self.n_mc, self.n_rf, self.n_iter, self.n_var_steps, self.n_hyp_steps, self.display_step, self.lr, self.hlr = get_model_settings(
            flags)

        self.init_sigma2_n, self.init_variance, self.init_lengthscale, self.init_p, self.prior_lambda_, self.posterior_lambda_ = self.get_hyperparameters(
            flags)

        self.pr_mu_g, self.pr_sigma2_g, self.pr_mu_o, self.pr_mu_w, self.pr_sigma2_w, self.prior_alpha = self.initialize_priors()
        self.mu_g, self.log_sigma2_g, self.mu_o, self.o_single_normal, self.log_sigma2_n, self.log_variance, self.mu_w, self.log_sigma2_w, self.log_alpha = self.initialize_variables()
        self.log_lengthscale, self.pr_log_sigma2_o, self.log_sigma2_o = self.initialize_omega()

        # sample from posterior to get kl and ell terms
        self.g, self.o, self.b, self._a = self.get_matrices(self.mu_g, self.log_sigma2_g, self.mu_o, self.log_sigma2_o,
                                                            self.mu_w, self.log_sigma2_w, self.log_alpha,
                                                            self.posterior_lambda_)
        self.ell, self.pred_signals, self.real_signals = self.calculate_ell(self.g, self.o, self.b, self.log_sigma2_n,
                                                                            self.log_variance, self.train_data)

        self.kl_o, self.kl_w, self.kl_g = self.get_kl()
        # calculating ELBO
        self.elbo = self.ell - self.kl_o - self.kl_w

        # get the operation for optimizing variational parameters
        self.var_opt, _, self.var_nans = get_optimizer(tf.negative(self.elbo),
                                                       [self.mu_g, self.log_sigma2_g, self.mu_o, self.log_sigma2_o,
                                                        self.mu_w, self.log_sigma2_w, self.log_alpha], self.lr)
        # get the operation for optimizing hyper parameters
        self.hyp_opt, _, self.hyp_nans = get_optimizer(tf.negative(self.elbo),
                                                       [self.log_sigma2_n, self.log_variance, self.log_lengthscale],
                                                       self.hlr)

        # initialize variables
        self.init_op = tf.compat.v1.initializers.global_variables()
        self.sess = tf.compat.v1.Session()
        self.sess.run(self.init_op)

    def optimize(self):
        # current global iteration over optimization steps.
        _iter = 0
        while self.n_iter is None or _iter < self.n_iter:
            self.logger.debug("\nSUBJECT %d: ITERATION %d STARTED\n" % (self.subject, _iter))
            self.run_step(self.n_var_steps, self.var_opt)
            self.run_step(self.n_hyp_steps, self.hyp_opt)
            _iter += 1

        self.run_variables()

        row_n = random.randint(1, self.n_nodes - 1)
        real_row = self.real_signals_[:, row_n]
        pred_row = self.pred_signals_[row_n, :]
        with open('signal_prediction.csv', 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.subject, row_n])
            writer.writerow(real_row)
            writer.writerow(pred_row)
            writer.writerow([])
            file.close()
        return self.mu_w_, self.sigma2_w_, self.alpa_, self.mu_g_, self.sigma2_g_

    def run_step(self, n_steps, opt):
        for i in range(0, n_steps):
            self.run_optimization(opt)
            if i % self.display_step == 0:
                self.log_optimization(i)

    def run_optimization(self, opt):
        self.elbo_, self.ell_, self.kl_g_, self.kl_o_, self.kl_w_, _ = self.sess.run(
            [self.elbo, self.ell, self.kl_g, self.kl_o, self.kl_w, opt])

    def run_variables(self):
        self.elbo_, self.pred_signals_, self.real_signals_, self.mu_w_, self.sigma2_w_, self.alpa_, self.mu_g_, self.sigma2_g_ = self.sess.run((
            self.elbo, self.pred_signals, self.real_signals, self.mu_w, tf.exp(self.log_sigma2_w),
            tf.exp(self.log_alpha), self.mu_g, tf.exp(self.log_sigma2_g)))

    def log_optimization(self, i):
        self.logger.debug(
            " local {i:d} iter: elbo={elbo_:.0f} (ell {ell_:.0f}, kl_g {kl_g_:.1f}, kl_o {kl_o_:.1f}, kl_w {kl_w_:.1f})".format(
                i=i, elbo_=self.elbo_, ell_=self.ell_, kl_g_=self.kl_g_, kl_o_=self.kl_o_, kl_w_=self.kl_w_))

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
        mu_g = tf.Variable(self.pr_mu_g, dtype=self.FLOAT)
        log_sigma2_g = tf.Variable(tf.math.log(self.pr_sigma2_g), dtype=self.FLOAT)
        mu_o = tf.Variable(self.pr_mu_o, dtype=self.FLOAT)

        o_single_normal = np.random.normal(loc=0, scale=1, size=(self.n_mc, self.n_rf, self.dim))
        log_sigma_2_n = tf.Variable(tf.math.log(tf.cast(self.init_sigma2_n, dtype=self.FLOAT)), dtype=self.FLOAT)
        log_variance = tf.Variable(tf.math.log(tf.cast(self.init_variance, dtype=self.FLOAT)), dtype=self.FLOAT)
        mu_w = tf.Variable(self.pr_mu_w, dtype=self.FLOAT)
        log_sigma2_w = tf.Variable(tf.math.log(self.pr_sigma2_w), dtype=self.FLOAT)
        log_alpha = tf.Variable(tf.math.log(self.prior_alpha), dtype=self.FLOAT)
        return mu_g, log_sigma2_g, mu_o, o_single_normal, log_sigma_2_n, log_variance, mu_w, log_sigma2_w, log_alpha

    def initialize_omega(self):
        init_lengthscale_matrix = tf.constant(self.init_lengthscale, dtype=self.FLOAT) * tf.ones((self.n_rf, self.dim),
                                                                                                 dtype=self.FLOAT)
        log_lengthscale = tf.Variable(2 * tf.math.log(init_lengthscale_matrix), dtype=self.FLOAT)
        pr_log_sigma2_o = -log_lengthscale
        log_sigma2_o = tf.Variable(pr_log_sigma2_o, dtype=self.FLOAT)
        return log_lengthscale, pr_log_sigma2_o, log_sigma2_o

    def get_matrices(self, mu_g, log_sigma2_g, mu_o, log_sigma2_o, mu_w, log_sigma2_w, log_alpha, posterior_lambda_):
        # Gamma
        z_g = tf.random.normal((self.n_mc, self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        g = tf.multiply(z_g, tf.sqrt(tf.exp(log_sigma2_g))) + mu_g

        # Omega
        o = tf.multiply(self.o_single_normal, tf.sqrt(tf.exp(log_sigma2_o))) + mu_o

        # sampling for W
        z_w = tf.random.normal((self.n_mc, self.n_nodes, self.n_nodes), dtype=self.FLOAT)
        w = tf.multiply(z_w, tf.sqrt(tf.exp(log_sigma2_w))) + mu_w
        w = tf.linalg.set_diag(w, tf.zeros((self.n_mc, self.n_nodes), dtype=self.FLOAT))

        # sampling for A
        u = tf.random.uniform((self.n_mc, self.n_nodes, self.n_nodes), minval=0, maxval=1, dtype=self.FLOAT)
        eps = tf.constant(1e-20, dtype=self.FLOAT)
        _a = tf.math.divide(tf.add(log_alpha, tf.subtract(tf.math.log(u + eps), tf.math.log(1 - u + eps))),
                            posterior_lambda_)
        a = tf.sigmoid(_a)
        a = tf.linalg.set_diag(a, tf.zeros((self.n_mc, self.n_nodes), dtype=self.FLOAT))

        b = tf.multiply(a, w)
        return g, o, b, _a

    def calculate_ell(self, g, o, b, log_sigma2_n, log_variance, real_data):
        n_signals = real_data.shape[0]
        # generate design matrix with shape (n_signals, n_dim)
        t = tf.cast(tf.expand_dims(tf.range(n_signals), 1), self.FLOAT)
        mean = tf.math.reduce_mean(t)
        std = tf.math.reduce_std(t)
        t = (t - mean) / std

        z = self.get_z(n_signals, g, o, log_variance, t)
        noise = np.random.normal(loc=0, scale=0.0001, size=(self.n_mc, self.n_nodes, self.n_signals))
        z = tf.add(z, tf.matmul(b, noise))

        v_current = tf.reshape(z, [self.n_mc, self.n_nodes, n_signals])
        exp_y = v_current
        for i in range(3):
            v_current = tf.matmul(b, v_current)
            exp_y = tf.add(v_current, exp_y)

        real_y = tf.expand_dims(tf.transpose(real_data), 0)
        norm = tf.norm(exp_y - real_y, ord=2, axis=1)
        norm_sum_by_t = tf.reduce_sum(norm, axis=1)
        norm_sum_by_t_avg_by_s = tf.reduce_mean(norm_sum_by_t)

        _two_pi = tf.constant(6.28, dtype=self.FLOAT)
        _half = tf.constant(0.5, dtype=self.FLOAT)
        first_part_ell = - _half * self.n_nodes * tf.cast(n_signals, dtype=self.FLOAT) * tf.math.log(
            tf.multiply(_two_pi, tf.exp(log_sigma2_n)))

        second_part_ell = - _half * tf.divide(norm_sum_by_t_avg_by_s, tf.exp(log_sigma2_n))
        ell = first_part_ell + second_part_ell

        return ell, tf.reduce_mean(exp_y, axis=0), real_data

    def get_z(self, n_signals, g, o, log_variance, t):
        o_temp = tf.reshape(o, [self.n_mc * self.n_rf, self.dim])
        fi_under = tf.reshape(tf.matmul(o_temp, tf.transpose(t)), [self.n_mc, self.n_rf, n_signals])
        fi = tf.sqrt(tf.math.divide(tf.exp(log_variance), self.n_rf)) * tf.concat([tf.cos(fi_under), tf.sin(fi_under)],
                                                                                  axis=1)
        return tf.matmul(g, fi)

    def get_kl(self):
        kl_o = get_dkl_normal(self.mu_o, tf.exp(self.log_sigma2_o), self.pr_mu_o, tf.exp(self.pr_log_sigma2_o))
        kl_w = get_kl_normal(self.mu_w, tf.exp(self.log_sigma2_w), self.pr_mu_w, self.pr_sigma2_w, self.FLOAT)
        kl_g = get_dkl_normal(self.mu_g, tf.exp(self.log_sigma2_g), self.pr_mu_g, self.pr_sigma2_g)
        return kl_o, kl_w, kl_g
