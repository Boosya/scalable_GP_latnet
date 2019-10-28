__author__ = 'EG'

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors import OpError
from sklearn.metrics import roc_auc_score

class ScalableLatnet:

    def __init__(self, flags, subject, dim, train_data, validation_data, test_data, true_conn, logger, session=None):
        self.FLOAT = tf.float64
        self.logger = logger
        self.subject = subject
        self.dim = dim
        (self.n_signals, self.n_nodes) = train_data.shape
        self.train_data = tf.constant(train_data, dtype=self.FLOAT)
        self.validation_data = tf.constant(validation_data, dtype=self.FLOAT)
        self.test_data = tf.constant(test_data, dtype=self.FLOAT)
        self.true_conn = tf.reshape(tf.constant(true_conn, dtype=self.FLOAT), [self.n_nodes, self.n_nodes])

        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1
        self.config = config

        # Set random seed for tensorflow and numpy operations
        tf.compat.v1.set_random_seed(flags.get_flag('seed'))
        np.random.seed(flags.get_flag('seed'))

        self.n_mc, self.n_rf, self.learn_omega, self.inv_calculation, self.n_approx_terms, self.n_iterations, self.n_var_steps, self.n_hyp_steps, self.n_all_steps, self.display_step, self.return_best_state, self.print_auc = self.get_model_settings(
            flags)
        if not self.print_auc:
            self.auc_ = 0

        self.init_p, self.init_lengthscale, self.init_sigma2_n, self.init_variance, self.posterior_lambda_, self.prior_lambda_ = self.get_hyperparameters(
            flags)

        self.one = tf.constant(1.0, dtype=self.FLOAT)
        self.half = tf.constant(0.5, dtype=self.FLOAT)
        self.eps = tf.constant(1e-20, dtype=self.FLOAT)

        self.prior_mu, self.prior_sigma2, self.prior_mu_gamma, self.prior_sigma2_gamma, self.prior_mu_omega, self.prior_alpha = self.initialize_priors()
        self.mu, self.log_sigma2, self.mu_gamma, self.log_sigma2_gamma, self.mu_omega, self.log_alpha, self.log_sigma2_n, self.log_variance = self.initialize_variables()
        self.log_lengthscale, self.prior_log_sigma2_omega, self.log_sigma2_omega, self.omega_single_normal, self.omega_prior_fixed = self.initialize_prior_and_variable_for_sigma2_omega()

        # sample from posterior to get kl and ell terms
        self.w, self.gamma, self.omega, self._a, self.a = self.get_matrices(self.mu, self.log_sigma2, self.mu_gamma,
                                                                            self.log_sigma2_gamma, self.mu_omega,
                                                                            self.log_sigma2_omega)
        self.kl_w, self.kl_gamma, self.kl_omega, self.kl_a = self.calculate_kl_terms()
        self.eig_check, self.first_part_ell, self.second_part_ell, self.ell, _, _ = self.calculate_ell(self.gamma,
            self.omega, self.log_variance, self.log_sigma2_n, self.train_data)

        # calculating ELBO
        self.elbo = tf.negative(self.kl_w) + tf.negative(self.kl_gamma) + tf.negative(self.kl_omega) + tf.negative(
            self.kl_a) + self.ell

        _, _, _, self.valid_ell, _, _ = self.calculate_ell(self.gamma, self.omega, self.log_variance, self.log_sigma2_n,
                                                           self.validation_data)

        _, _, _, self.test_ell, self.pred_signals_normalized, self.real_signals_normalilzed = self.calculate_ell(
            self.gamma, self.omega, self.log_variance, self.log_sigma2_n, self.test_data)

        # get the operation for optimizing variational parameters
        self.var_opt, _, self.var_nans = self.get_opzimier(tf.negative(self.elbo),
                                                           [self.mu, self.log_sigma2, self.mu_gamma,
                                                            self.log_sigma2_gamma, self.mu_omega, self.log_sigma2_omega,
                                                            self.log_alpha], flags.get_flag('var_learning_rate'))

        if flags.get_flag('learn_lengthscale') == 'yes':
            self.hyp_opt, _, self.hyp_nans = self.get_opzimier(tf.negative(self.elbo),
                                                               [self.log_sigma2_n, self.log_variance,
                                                                self.log_lengthscale],
                                                               flags.get_flag('hyp_learning_rate'))
            # get general optimization parameters
            self.all_opt, _, self.all_nans = self.get_opzimier(tf.negative(self.elbo),
                                                               [self.mu, self.log_sigma2, self.mu_gamma,
                                                                self.log_sigma2_gamma, self.mu_omega,
                                                                self.log_sigma2_omega, self.log_alpha,
                                                                self.log_sigma2_n, self.log_variance,
                                                                self.log_lengthscale],
                                                               flags.get_flag('all_learning_rate'))
        else:
            self.hyp_opt, _, self.hyp_nans = self.get_opzimier(tf.negative(self.elbo),
                                                               [self.log_sigma2_n, self.log_variance],
                                                               flags.get_flag('hyp_learning_rate'))
            # get general optimization parameters
            self.all_opt, _, self.all_nans = self.get_opzimier(tf.negative(self.elbo),
                                                               [self.mu, self.log_sigma2, self.mu_gamma,
                                                                self.log_sigma2_gamma, self.mu_omega,
                                                                self.log_sigma2_omega, self.log_alpha,
                                                                self.log_sigma2_n, self.log_variance],
                                                               flags.get_flag('all_learning_rate'))

        # initialize variables
        self.init_op = tf.compat.v1.initializers.global_variables()

        if session:
            self.sess = session
        else:
            self.sess = tf.compat.v1.Session(config=self.config)

        self.sess.run(self.init_op)

    def optimize(self):
        # current global iteration over optimization steps.
        _iter = 0
        best_elbo = None
        try:
            while self.n_iterations is None or _iter < self.n_iterations:
                self.logger.debug("\nSUBJECT %d: ITERATION %d STARTED\n" % (self.subject, _iter))
                # optimizing variational parameters
                if self.n_var_steps > 0:
                    self.logger.debug("optimizing variational parameters")
                    self.run_step(self.n_var_steps, self.var_opt, self.var_nans, best_elbo)

                # optimizing hyper parameters
                if self.n_hyp_steps > 0:
                    self.logger.debug("optimizing hyper parameters")
                    self.run_step(self.n_hyp_steps, self.hyp_opt, self.hyp_nans, best_elbo)

                # optimizing all the parameters at once
                if self.n_all_steps > 0:
                    self.logger.debug("optimizing all parameters")
                    self.run_step(self.n_all_steps, self.all_opt, self.all_nans, best_elbo)
                _iter += 1
        except OpError as e:
            self.logger.error(e.message)

        if not self.return_best_state:
            self.run_variables()
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        self.logger.debug(
            "real\n{real}\nexp\n{exp}".format(real=self.real_signals_normalilzed_, exp=self.pred_signals_normalized_))
        self.logger.debug("TEST ELL {:.0f}".format(self.test_ell_))
        return self.elbo_, self.sigma2_n_, self.mu_, self.sigma2_, self.mu_gamma_, self.sigma2_gamma_, self.mu_omega_, self.sigma2_omega_, self.alpha_, self.lengthscale_, self.log_variance_

    def run_step(self, n_steps, opt, nans, best_elbo):
        for i in range(0, n_steps):
            nans_ = self.run_optimization(opt, nans)
            if self.return_best_state and (not best_elbo or self.elbo_ > best_elbo):
                best_elbo = self.elbo_
                self.run_variables()
            if i % self.display_step == 0:
                self.log_optimization(i, nans_)

    def run_optimization(self, opt, nans):
        if self.print_auc:
            self.elbo_, self.kl_w_, self.kl_a_, self.kl_gamma_, self.kl_omega_, self.ell_, self.val_ell_, self.first_part_ell_, self.second_part_ell_, self.eig_check_, _, nans_, self.true_conn_,  self.alpha_ = self.sess.run(
                [self.elbo, self.kl_w, self.kl_a, self.kl_gamma, self.kl_omega, self.ell, self.valid_ell,
                 self.first_part_ell, self.second_part_ell, self.eig_check, opt, nans, self.true_conn, tf.exp(self.log_alpha)])
            self.real_conn_ = self.get_bool_array_of_upper_and_lower_triangular(self.true_conn_)
            self.p_ = self.alpha_ / (1.0 + self.alpha_)
            self.pred_conn_ = self.get_array_of_upper_and_lower_triangular(self.p_)
            self.auc_ = roc_auc_score(self.real_conn_, self.pred_conn_)
        else:
            self.elbo_, self.kl_w_, self.kl_a_, self.kl_gamma_, self.kl_omega_, self.ell_, self.val_ell_, self.first_part_ell_, self.second_part_ell_, self.eig_check_, _, nans_ = self.sess.run(
                [self.elbo, self.kl_w, self.kl_a, self.kl_gamma, self.kl_omega, self.ell, self.valid_ell,
                 self.first_part_ell, self.second_part_ell, self.eig_check, opt, nans])

        return nans_

    def run_variables(self):
        self.elbo_, self.test_ell_, self.sigma2_n_, self.mu_, self.sigma2_, self.mu_gamma_, self.sigma2_gamma_, self.mu_omega_, self.sigma2_omega_, self.alpha_, self.lengthscale_, self.log_variance_, self.pred_signals_normalized_, self.real_signals_normalilzed_ = self.sess.run(
            (self.elbo, self.test_ell, tf.exp(self.log_sigma2_n), self.mu, tf.exp(self.log_sigma2), self.mu_gamma,
             tf.exp(self.log_sigma2_gamma), self.mu_omega, tf.exp(self.log_sigma2_omega), tf.exp(self.log_alpha),
             tf.exp(self.log_lengthscale), tf.exp(self.log_variance), self.pred_signals_normalized,
             self.real_signals_normalilzed))

    def log_optimization(self, i, nans_):


        self.logger.debug(" local {i:d} iter: "
                          "valid_ell: {ell_:.0f}, auc: {auc_:.2f} elbo: {elbo:.0f} "
                          "(KL W={kl_w:.2f}, KL A={kl_a:.2f}, "
                          "KL G={kl_gamma:.2f},KL O={kl_omega:.2f}, ell={ell:.0f}, "
                          "outer_ell={first_part_ell:.0f}, netw_ell={second_part_ell:.0f},"
                          "eig_check = {eig_check}), "
                          "{nans:d} nan in grads".format(ell_=self.val_ell_, auc_=self.auc_, i=i, elbo=self.elbo_, kl_w=self.kl_w_,
                                                         kl_a=self.kl_a_, kl_gamma=self.kl_gamma_,
                                                         kl_omega=self.kl_omega_, ell=self.ell_,
                                                         first_part_ell=self.first_part_ell_,
                                                         second_part_ell=self.second_part_ell_,
                                                         eig_check=self.eig_check_, nans=nans_))

    def get_model_settings(self, flags):
        n_mc = flags.get_flag('n_mc')
        n_rf = flags.get_flag('n_rff')
        learn_omega = flags.get_flag('learn_Omega')
        inv_calculation = flags.get_flag('inv_calculation')
        n_approx_terms = flags.get_flag('n_approx_terms')
        n_iterations = flags.get_flag('n_iterations')
        n_var_steps = flags.get_flag('var_steps')
        n_hyp_steps = flags.get_flag('hyp_steps')
        n_all_steps = flags.get_flag('all_steps')
        display_step = flags.get_flag('display_step')
        return_best_state = flags.get_flag("return_best_state")
        print_auc = flags.get_flag("print_auc")
        return n_mc, n_rf, learn_omega, inv_calculation, n_approx_terms, n_iterations, n_var_steps, n_hyp_steps, n_all_steps, display_step, return_best_state, print_auc

    def get_hyperparameters(self, flags):
        init_p = flags.get_flag('init_p')
        init_lengthscale = flags.get_flag('init_lengthscale')
        init_sigma2_n = flags.get_flag('init_sigma2_n')
        init_variance = flags.get_flag('init_variance')
        posterior_lambda_ = tf.constant(flags.get_flag('lambda_posterior'), dtype=self.FLOAT)
        prior_lambda_ = tf.constant(flags.get_flag('lambda_prior'), dtype=self.FLOAT)
        return init_p, init_lengthscale, init_sigma2_n, init_variance, posterior_lambda_, prior_lambda_

    def initialize_priors(self):
        prior_mu = tf.zeros((self.n_nodes, self.n_nodes), dtype=self.FLOAT)
        prior_sigma2 = tf.multiply(tf.cast(1. / (self.n_nodes * self.init_p), self.FLOAT),
                                   tf.ones((self.n_nodes, self.n_nodes), dtype=self.FLOAT))
        prior_mu_gamma = tf.zeros((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_sigma2_gamma = tf.ones((self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        prior_mu_omega = tf.zeros((self.n_rf, self.dim), dtype=self.FLOAT)
        prior_alpha = tf.multiply(tf.cast(self.init_p / (1. - self.init_p), self.FLOAT),
                                  tf.ones((self.n_nodes, self.n_nodes), dtype=self.FLOAT))
        return prior_mu, prior_sigma2, prior_mu_gamma, prior_sigma2_gamma, prior_mu_omega, prior_alpha

    def initialize_variables(self):
        mu = tf.Variable(self.prior_mu, dtype=self.FLOAT)
        log_sigma2 = tf.Variable(tf.math.log(self.prior_sigma2), dtype=self.FLOAT)
        mu_gamma = tf.Variable(self.prior_mu_gamma, dtype=self.FLOAT)
        log_sigma2_gamma = tf.Variable(tf.math.log(self.prior_sigma2_gamma), dtype=self.FLOAT)
        mu_omega = tf.Variable(self.prior_mu_omega, dtype=self.FLOAT)
        log_alpha = tf.Variable(tf.math.log(self.prior_alpha), dtype=self.FLOAT)
        log_sigma2_n = tf.Variable(tf.math.log(tf.constant(self.init_sigma2_n, dtype=self.FLOAT)), dtype=self.FLOAT)
        log_variance = tf.Variable(tf.math.log(tf.constant(self.init_variance, dtype=self.FLOAT)), dtype=self.FLOAT)
        return mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_alpha, log_sigma2_n, log_variance

    def initialize_prior_and_variable_for_sigma2_omega(self):
        log_lengthscale = tf.Variable(tf.math.log(tf.constant(self.init_lengthscale, dtype=self.FLOAT)),
                                      dtype=self.FLOAT)
        prior_log_sigma2_omega = -2 * log_lengthscale
        log_sigma2_omega = tf.Variable(prior_log_sigma2_omega, dtype=self.FLOAT)
        omega_single_normal = np.random.normal(loc=0, scale=1, size=(self.n_mc, self.n_rf, self.dim))
        omega_prior_fixed = omega_single_normal * tf.sqrt(tf.exp(prior_log_sigma2_omega))
        return log_lengthscale, prior_log_sigma2_omega, log_sigma2_omega, omega_single_normal, omega_prior_fixed

    def get_matrices(self, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega):
        # sampling for W
        z_w = tf.random.normal((self.n_mc, self.n_nodes, self.n_nodes), dtype=self.FLOAT)
        w = tf.multiply(z_w, tf.sqrt(tf.exp(log_sigma2))) + mu
        w = tf.linalg.set_diag(w, tf.zeros((self.n_mc, self.n_nodes), dtype=self.FLOAT))

        # Gamma
        z_gamma = tf.random.normal((self.n_mc, self.n_nodes, 2 * self.n_rf), dtype=self.FLOAT)
        gamma = tf.multiply(z_gamma, tf.sqrt(tf.exp(log_sigma2_gamma))) + mu_gamma

        # Omega
        if self.learn_omega == 'var-resampled':
            z_omega = tf.random.normal((self.n_mc, self.n_rf, self.dim), dtype=self.FLOAT)
            omega = tf.multiply(z_omega, tf.sqrt(tf.exp(log_sigma2_omega))) + mu_omega
        elif self.learn_omega == 'var-fixed':
            omega = tf.multiply(self.omega_single_normal, tf.sqrt(tf.exp(log_sigma2_omega))) + mu_omega
        elif self.learn_omega == 'prior-fixed':
            omega = self.omega_prior_fixed
        else:
            raise Exception

        # sampling for A
        u = tf.random.uniform((self.n_mc, self.n_nodes, self.n_nodes), minval=0, maxval=1, dtype=self.FLOAT)
        _a = tf.math.divide(
            tf.add(self.log_alpha, tf.subtract(tf.math.log(u + self.eps), tf.math.log(self.one - u + self.eps))),
            self.posterior_lambda_)
        a = tf.sigmoid(_a)
        a = tf.linalg.set_diag(a, tf.zeros((self.n_mc, self.n_nodes), dtype=self.FLOAT))
        return w, gamma, omega, _a, a

    def calculate_kl_terms(self):
        kl_w, _ = self.get_kl_normal(self.mu, tf.exp(self.log_sigma2), self.prior_mu, self.prior_sigma2)
        kl_gamma = self.get_dkl_normal(self.mu_gamma, tf.exp(self.log_sigma2_gamma), self.prior_mu_gamma,
                                       self.prior_sigma2_gamma)
        kl_omega = self.get_dkl_normal(self.mu_omega, tf.exp(self.log_sigma2_omega), self.prior_mu_omega,
                                       tf.exp(self.prior_log_sigma2_omega))
        kl_a = self.get_kl_logistic(self._a, tf.exp(self.log_alpha), self.prior_lambda_, self.posterior_lambda_,
                                    self.prior_alpha)
        return kl_w, kl_gamma, kl_omega, kl_a

    def get_kl_normal(self, posterior_mu, posterior_sigma2, prior_mu, prior_sigma2):
        kl = tf.add(tf.math.divide(tf.add(tf.square(tf.subtract(posterior_mu, prior_mu)), posterior_sigma2),
                                   tf.multiply(2 * self.one, prior_sigma2)),
                    -self.half + self.half * tf.math.log(prior_sigma2) - self.half * tf.math.log(posterior_sigma2))
        kl = tf.linalg.set_diag(tf.expand_dims(kl, -3), tf.zeros((1, tf.shape(kl)[0]), dtype=self.FLOAT))
        return tf.reduce_sum(kl[0]), kl

    def get_dkl_normal(self, mu, sigma2, prior_mu, prior_sigma2):
        kl = self.half * tf.add(
            tf.add(tf.math.log(tf.math.divide(prior_sigma2, sigma2)) - self.one, tf.math.divide(sigma2, prior_sigma2)),
            tf.math.divide(tf.square(tf.subtract(mu, prior_mu)), prior_sigma2))
        return tf.reduce_sum(kl)

    def get_kl_logistic(self, X, posterior_alpha, prior_lambda_, posterior_lambda_, prior_alpha):
        prior_mu = tf.math.log(prior_alpha)
        prior = tf.subtract(tf.add(tf.subtract(tf.math.log(prior_lambda_), tf.multiply(prior_lambda_, X)), prior_mu),
                            tf.multiply(2 * self.one, tf.math.log(tf.add(self.one, tf.exp(
                                tf.add(tf.negative(tf.multiply(prior_lambda_, X)), prior_mu))))))
        posterior_mu = tf.math.log(posterior_alpha)
        posterior = tf.subtract(
            tf.add(tf.subtract(tf.math.log(posterior_lambda_), tf.multiply(posterior_lambda_, X)), posterior_mu),
            tf.multiply(2 * self.one, tf.math.log(
                tf.add(self.one, tf.exp(tf.add(tf.negative(tf.multiply(posterior_lambda_, X)), posterior_mu))))))

        logdiff = posterior - prior
        # set diagonal part to zero
        logdiff = tf.linalg.set_diag(logdiff, tf.zeros((tf.shape(logdiff)[0], tf.shape(logdiff)[1]), dtype=self.FLOAT))
        return tf.reduce_sum(tf.reduce_mean(logdiff, [0]))

    def calculate_ell(self, gamma, omega, log_variance, log_sigma2_n, real_data):
        n_signals = real_data.shape[0]
        n_nodes = real_data.shape[1]
        t = tf.cast(tf.expand_dims(tf.range(n_signals), 1), self.FLOAT)
        mean = tf.math.reduce_mean(t)
        std = tf.math.reduce_std(t)
        t = (t - mean) / std
        z = self.get_z(n_signals, gamma, omega, t, log_variance)
        b, exp_y = self.get_exp_y(z, n_signals)
        eig_check = self.check_eigvalues(b)
        real_y = tf.expand_dims(tf.transpose(real_data), 0)
        norm = tf.norm(exp_y - real_y, ord=2, axis=1)
        norm_sum_by_t = tf.reduce_sum(norm, axis=1)
        norm_sum_by_t_avg_by_s = tf.reduce_mean(norm_sum_by_t)
        _two_pi = tf.constant(6.28, dtype=self.FLOAT)
        first_part_ell = - self.half * self.n_nodes * tf.cast(n_signals, dtype=self.FLOAT) * tf.math.log(
            tf.multiply(_two_pi, tf.exp(log_sigma2_n)))

        second_part_ell = - self.half * tf.divide(norm_sum_by_t_avg_by_s, tf.exp(log_sigma2_n))
        ell = first_part_ell + second_part_ell
        return eig_check, first_part_ell, second_part_ell, ell, self.normalize(tf.reduce_mean(exp_y, axis=0), n_signals,
                                                                               n_nodes), self.normalize(real_data,
                                                                                                        n_signals,
                                                                                                        n_nodes)

    def get_z(self, n_signals, gamma, omega, t, log_variance):
        omega_temp = tf.reshape(omega, [self.n_mc * self.n_rf, self.dim])
        fi_under = tf.reshape(tf.matmul(omega_temp, tf.transpose(t)), [self.n_mc, self.n_rf, n_signals])
        fi = tf.sqrt(tf.math.divide(tf.exp(log_variance), self.n_rf)) * tf.concat([tf.cos(fi_under), tf.sin(fi_under)],
                                                                                  axis=1)
        return tf.matmul(gamma, fi)

    def get_exp_y(self, z, n_signals):
        identity = tf.constant(np.identity(self.n_nodes), dtype=self.FLOAT)
        b = tf.multiply(self.a, self.w)
        # add_noise = False
        # if add_noise:
        #     S_N_T = (S,N,T)
        #     noise = np.random.normal(loc=0,scale=0.0001,size=S_N_T)
        #     Z = tf.add(Z,tf.matmul(B,noise))
        identity_minus_b = tf.subtract(identity, b)
        if self.inv_calculation == 'approx':
            v_current = tf.reshape(z, [self.n_mc, self.n_nodes, n_signals])
            exp_y = v_current
            for i in range(self.n_approx_terms):
                v_current = tf.matmul(b, v_current)
                exp_y = tf.add(v_current, exp_y)  # v + Bv
        elif self.inv_calculation == 'solver':
            exp_y = tf.linalg.solve(identity_minus_b, z)
        elif self.inv_calculation == 'cholesky':
            identity_minus_b_col = self.cholesky_decompose(identity_minus_b)
            exp_y = tf.linalg.cholesky_solve(identity_minus_b_col, z)
        else:
            identity_minus_b_inverse = tf.matrix_inverse(identity_minus_b)
            exp_y = tf.matmul(identity_minus_b_inverse, z)
        return b, exp_y

    def check_eigvalues(self, b):
        eig_values, eig_vectors = tf.linalg.eigh(b)
        max_eig_check = abs(tf.reduce_max(eig_values)) < 1
        min_eig_check = abs(tf.reduce_min(eig_values)) < 1
        return max_eig_check & min_eig_check

    def cholesky_decompose(self, x):
        # add small noise to eigenvalues
        eig_values, eig_vectors = tf.linalg.eigh(x)
        diag_eig_values = tf.linalg.diag(eig_values + self.eps)
        vect_diag = (tf.matmul(eig_vectors, diag_eig_values))
        vec_transp = (tf.transpose(eig_vectors, perm=[0, 2, 1]))
        return tf.cholesky(tf.matmul(vect_diag, vec_transp))

    def normalize(self, tensor, shape_0, shape_1):
        tensor_reshaped = tf.reshape(tensor, shape=[shape_0 * shape_1, 1])
        mean = tf.math.reduce_mean(tensor_reshaped)
        std = tf.math.reduce_std(tensor_reshaped)
        tensor_normalized_reshaped = (tensor_reshaped - mean) / std
        return tf.reshape(tensor_normalized_reshaped, shape=[shape_0, shape_1])

    def get_opzimier(self, objective, trainables, learning_rate, max_global_norm=1.0):
        """
        Calculates the Tensorflow operation for optimizing `objective' function using AdamOptimizer.
        Note that NANs in the gradients will be replaced by zeros.
        Args:
                objective: double. Objective function to be optimized.
                trainables: List of variables which will be optimized.
                learning_rate: Learning rate of AdamOptimizer
                max_global_norm: Used for gradient clipping.
        """
        grads = tf.gradients(ys=objective, xs=trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip([self.replace_nan_with_zero(g) for g in grads], trainables)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        return optimizer.apply_gradients(grad_var_pairs), grads, self.contains_nan(grads)

    def replace_nan_with_zero(self, w):
        return tf.compat.v1.where(tf.math.is_nan(w), tf.ones_like(w) * 0.0, w)

    def contains_nan(self, w):
        for w_ in w:
            if tf.reduce_all(input_tensor=tf.math.is_nan(w_)) is None:
                return tf.reduce_all(input_tensor=tf.math.is_nan(w_))
        return tf.reduce_all(input_tensor=tf.math.is_nan(w_))

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
        final_matrix = tri_upper_no_diag+tri_lower_no_diag
        final_matrix_without_diag = final_matrix[~np.eye(final_matrix.shape[0], dtype=bool)].reshape(
            final_matrix.shape[0], -1)
        result = []
        for row in final_matrix_without_diag:
            result.extend(row)
        return result
