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
    display_step = flags.get_flag('display_step')
    lr = flags.get_flag('var_learning_rate')
    return n_mc, n_rf, n_iterations, n_var_steps, display_step, lr


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

        self.n_mc, self.n_rf, self.n_iter, self.n_steps, self.display_step, self.lr = get_model_settings(flags)

        self.pr_mu_g, self.pr_sigma2_g, self.pr_mu_o, self.pr_sigma2_o = self.initialize_priors()
        self.mu_g, self.log_sigma2_g, self.mu_o, self.log_sigma2_o, self.o_single_normal = self.initialize_variables()

        # sample from posterior to get kl and ell terms
        self.g, self.o = self.get_matrices(self.mu_g, self.log_sigma2_g, self.mu_o, self.log_sigma2_o)
        self.ell, self.pred_signals, self.real_signals = self.calculate_ell(self.g, self.o, self.train_data)

        # calculating ELBO
        self.elbo = self.ell

        # get the operation for optimizing variational parameters
        self.var_opt, _, self.var_nans = get_optimizer(tf.negative(self.elbo),
                                                       [self.mu_g, self.log_sigma2_g, self.mu_o, self.log_sigma2_o],
                                                       self.lr)

        # initialize variables
        self.init_op = tf.compat.v1.initializers.global_variables()
        self.sess = tf.compat.v1.Session()
        self.sess.run(self.init_op)

    def optimize(self):
        # current global iteration over optimization steps.
        _iter = 0
        while self.n_iter is None or _iter < self.n_iter:
            self.logger.debug("\nSUBJECT %d: ITERATION %d STARTED\n" % (self.subject, _iter))
            self.run_step(self.n_steps)
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

    def run_step(self, n_steps):
        for i in range(0, n_steps):
            self.run_optimization()
            if i % self.display_step == 0:
                self.log_optimization(i)

    def run_optimization(self):
        self.elbo_, _ = self.sess.run([self.elbo, self.var_opt])

    def run_variables(self):
        self.elbo_, self.mu_g_, self.sigma2_g_, self.mu_o_, self.sigma2_o_, self.pred_signals_, self.real_signals_ = self.sess.run(
            (self.elbo, self.mu_g, tf.exp(self.log_sigma2_g), self.mu_o, tf.exp(self.log_sigma2_o), self.pred_signals,
             self.real_signals))

    def log_optimization(self, i):
        self.logger.debug(
            " local {i:d} iter: elbo={elbo_:.0f}".format(i=i, elbo_=self.elbo_))

    def initialize_priors(self):
        prior_mu_g = tf.zeros((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_sigma2_g = tf.ones((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_mu_o = tf.zeros((self.n_rf, self.dim), dtype=self.FLOAT)
        prior_sigma2_o = tf.ones((self.n_rf, self.dim), dtype=self.FLOAT)
        return prior_mu_g, prior_sigma2_g, prior_mu_o, prior_sigma2_o

    def initialize_variables(self):
        mu_g = tf.Variable(self.pr_mu_g, dtype=self.FLOAT)
        log_sigma2_g = tf.Variable(tf.math.log(self.pr_sigma2_g), dtype=self.FLOAT)
        mu_o = tf.Variable(self.pr_mu_o, dtype=self.FLOAT)
        log_sigma2_o = tf.Variable(tf.math.log(self.pr_sigma2_o), dtype=self.FLOAT)
        o_single_normal = np.random.normal(loc=0, scale=1, size=(self.n_mc, self.n_rf, self.dim))
        return mu_g, log_sigma2_g, mu_o, log_sigma2_o, o_single_normal

    def get_matrices(self, mu_g, log_sigma2_g, mu_o, log_sigma2_o):
        # Gamma
        z_g = tf.random.normal((self.n_mc, self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        g = tf.multiply(z_g, tf.sqrt(tf.exp(log_sigma2_g))) + mu_g

        # Omega
        o = tf.multiply(self.o_single_normal, tf.sqrt(tf.exp(log_sigma2_o))) + mu_o
        return g, o

    def calculate_ell(self, g, o, real_data):
        n_signals = real_data.shape[0]
        # generate design matrix with shape (n_signals, n_dim)
        t = tf.cast(tf.expand_dims(tf.range(n_signals), 1), self.FLOAT)
        mean = tf.math.reduce_mean(t)
        std = tf.math.reduce_std(t)
        t = (t - mean) / std

        z = self.get_z(n_signals, g, o, t)
        exp_y = z
        real_y = tf.expand_dims(tf.transpose(real_data), 0)
        norm = tf.norm(exp_y - real_y, ord=2, axis=1)
        norm_sum_by_t = tf.reduce_sum(norm, axis=1)
        norm_sum_by_t_avg_by_s = tf.reduce_mean(norm_sum_by_t)

        return -norm_sum_by_t_avg_by_s, tf.reduce_mean(exp_y, axis=0), real_data

    def get_z(self, n_signals, g, o, t):
        o_temp = tf.reshape(o, [self.n_mc * self.n_rf, self.dim])
        fi_under = tf.reshape(tf.matmul(o_temp, tf.transpose(t)), [self.n_mc, self.n_rf, n_signals])
        fi = tf.concat([tf.cos(fi_under), tf.sin(fi_under)], axis=1)
        return tf.matmul(g, fi)
