__author__ = 'AD'

import tensorflow as tf

import numpy as np
from tensorflow.python.framework.errors import OpError


class ScalableLatnet:
    FLOAT = tf.float64

    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.compat.v1.name_scope('summaries'):
            mean = tf.reduce_mean(input_tensor=var)
            tf.compat.v1.summary.scalar('mean', mean)
            with tf.compat.v1.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
            tf.compat.v1.summary.scalar('stddev', stddev)
            tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
            tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
            tf.compat.v1.summary.histogram('histogram', var)
        tf.summary.tensor_summary('tensor', var)

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
        grad_var_pairs = zip([ScalableLatnet.replace_nan_with_zero(g)
                              for g in grads], trainables)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.apply_gradients(grad_var_pairs), grads, ScalableLatnet.contains_nan(grads)

    @staticmethod
    def run_model(flags, D, t, Y):

        # get variables that will be optimized.
        log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, log_lengthscale, log_variance = \
            ScalableLatnet.get_vairables(flags, D, Y)

        # get KL terms of ELL terms.
        kl_W, kl_G, kl_O, kl_A, ell, first, second, third, eig_check = \
            ScalableLatnet.get_elbo(flags, D, t, Y, tf.exp(log_sigma2_n), mu, tf.exp(log_sigma2), mu_gamma,
                                    tf.exp(log_sigma2_gamma), mu_omega, tf.exp(log_sigma2_omega),
                                    tf.exp(log_alpha), tf.exp(log_variance))

        # calculating ELBO
        elbo = tf.negative(kl_W) + tf.negative(kl_G) + \
               tf.negative(kl_O) + tf.negative(kl_A) + ell

        # get the operation for optimizing variational parameters
        var_opt, _, var_nans = ScalableLatnet.get_opzimier(tf.negative(elbo),
                                                           [mu, log_sigma2,
                                                            mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega,
                                                            log_alpha], flags.get_flag().var_learning_rate)

        hyp_opt, _, hyp_nans = ScalableLatnet.get_opzimier(tf.negative(
            elbo), [log_sigma2_n, log_variance], flags.get_flag().hyp_learning_rate)
        merged = tf.summary.merge_all()

        return merged, var_opt, hyp_opt, elbo, kl_W, kl_A, kl_G, kl_O, ell, first, second, third, eig_check, log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, log_lengthscale, log_variance, var_nans, hyp_nans

    @staticmethod
    def logp_logistic(X, alpha, lambda_):

        mu = tf.log(alpha)
        return tf.subtract(tf.add(tf.subtract(tf.log(lambda_), tf.multiply(lambda_, X)), mu),
                           tf.multiply(tf.constant(2.0, dtype=ScalableLatnet.FLOAT),
                                       tf.log(tf.add(tf.constant(1.0, dtype=ScalableLatnet.FLOAT),
                                                     tf.exp(tf.add(tf.negative(tf.multiply(lambda_, X)), mu))))))

    @staticmethod
    def get_KL_logistic(X, posterior_alpha, prior_lambda_, posterior_lambda_, prior_alpha):

        logdiff = ScalableLatnet.logp_logistic(X, posterior_alpha, posterior_lambda_) - ScalableLatnet.logp_logistic(X,
                                                                                                                     prior_alpha,
                                                                                                                     prior_lambda_)
        logdiff = tf.matrix_set_diag(logdiff, tf.zeros((tf.shape(logdiff)[0], tf.shape(logdiff)[1]),
                                                       dtype=ScalableLatnet.FLOAT))  # set diagonal part to zero
        return tf.reduce_sum(tf.reduce_mean(logdiff, [0]))

    @staticmethod
    def get_KL_normal(posterior_mu, posterior_sigma2, prior_mu, prior_sigma2):

        _half = tf.constant(0.5, dtype=ScalableLatnet.FLOAT)
        kl = tf.add(
            tf.math.divide(tf.add(tf.square(tf.subtract(posterior_mu, prior_mu)), posterior_sigma2), tf.multiply(
                tf.constant(2., dtype=ScalableLatnet.FLOAT), prior_sigma2)),
            -_half + _half * tf.log(prior_sigma2) - _half * tf.log(posterior_sigma2))
        kl = tf.matrix_set_diag(tf.expand_dims(
            kl, -3), tf.zeros((1, tf.shape(kl)[0]), dtype=ScalableLatnet.FLOAT))
        return tf.reduce_sum(kl[0])

    def get_DKL_normal(mu, sigma2, prior_mu, prior_sigma2):

        _half = tf.constant(0.5, dtype=ScalableLatnet.FLOAT)
        _munus_one = tf.constant(-1, dtype=ScalableLatnet.FLOAT)
        kl = _half * tf.add(tf.add(tf.math.log(tf.math.divide(prior_sigma2, sigma2))
                                   + _munus_one, tf.math.divide(sigma2, prior_sigma2)),
                            tf.math.divide(tf.square(tf.subtract(mu, prior_mu)), prior_sigma2))
        return tf.reduce_sum(kl)

    @staticmethod
    def get_priors(flags, D, N, T):

        N_by_N = (N, N)
        N_by_2Nrf = (N, 2 * flags.get_flag().n_rff)
        Nrf_by_D = (flags.get_flag().n_rff, D)

        p = flags.get_flag().init_p
        lengthscale = 1. / np.sqrt(T)

        prior_mu = tf.zeros(N_by_N, dtype=ScalableLatnet.FLOAT)
        prior_sigma2 = tf.multiply(
            tf.cast(1. / (N * p), ScalableLatnet.FLOAT), tf.ones(N_by_N, dtype=ScalableLatnet.FLOAT))

        prior_mu_gamma = tf.zeros(N_by_2Nrf, dtype=ScalableLatnet.FLOAT)
        prior_sigma2_gamma = tf.ones(N_by_2Nrf, dtype=ScalableLatnet.FLOAT)

        prior_mu_omega = tf.zeros(Nrf_by_D, dtype=ScalableLatnet.FLOAT)
        prior_sigma2_omega = tf.ones(
            Nrf_by_D, dtype=ScalableLatnet.FLOAT) / lengthscale / lengthscale

        prior_alpha = tf.multiply(
            tf.cast(p / (1. - p), ScalableLatnet.FLOAT), tf.ones(N_by_N, dtype=ScalableLatnet.FLOAT))
        return prior_mu, prior_sigma2, prior_mu_gamma, prior_sigma2_gamma, prior_mu_omega, prior_sigma2_omega, prior_alpha

    @staticmethod
    def get_vairables(flags, D, Y):

        N = Y.shape[1]
        T = Y.shape[0]

        prior_mu, prior_sigma2, prior_mu_gamma, prior_sigma2_gamma, prior_mu_omega, prior_sigma2_omega, prior_alpha = ScalableLatnet.get_priors(
            flags,
            D, N, T)

        mu = tf.Variable(prior_mu, dtype=ScalableLatnet.FLOAT)
        log_sigma2 = tf.Variable(tf.log(prior_sigma2), dtype=ScalableLatnet.FLOAT)

        mu_gamma = tf.Variable(prior_mu_gamma, dtype=ScalableLatnet.FLOAT)
        log_sigma2_gamma = tf.Variable(
            tf.log(prior_sigma2_gamma), dtype=ScalableLatnet.FLOAT)

        mu_omega = tf.Variable(prior_mu_omega, dtype=ScalableLatnet.FLOAT)
        log_sigma2_omega = tf.Variable(
            tf.log(prior_sigma2_omega), dtype=ScalableLatnet.FLOAT)

        log_alpha = tf.Variable(tf.log(prior_alpha), dtype=ScalableLatnet.FLOAT)

        log_sigma2_n = tf.Variable(
            tf.log(tf.constant(flags.get_flag().init_sigma2_n, dtype=ScalableLatnet.FLOAT)), dtype=ScalableLatnet.FLOAT)

        log_lengthscale = tf.Variable(
            tf.log(tf.constant(flags.get_flag().init_lengthscale, dtype=ScalableLatnet.FLOAT)),
            dtype=ScalableLatnet.FLOAT)
        log_variance = tf.Variable(
            tf.log(tf.constant(flags.get_flag().init_variance, dtype=ScalableLatnet.FLOAT)), dtype=ScalableLatnet.FLOAT)

        return log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, log_lengthscale, log_variance

    @staticmethod
    def get_elbo(flags, D, t, Y, sigma2_n, mu, sigma2, mu_gamma, sigma2_gamma, mu_omega, sigma2_omega, alpha,
                 variance):

        # number of nodes
        N = Y.shape[1]

        # number of observation per node.
        T = Y.shape[0]

        # number of Monte-Caro samples
        S = flags.get_flag().n_mc

        posterior_lambda_ = tf.constant(flags.get_flag().lambda_postetior, dtype=ScalableLatnet.FLOAT)
        prior_lambda_ = tf.constant(flags.get_flag().lambda_prior, dtype=ScalableLatnet.FLOAT)

        eps = tf.constant(1e-20, dtype=ScalableLatnet.FLOAT)
        Y = tf.constant(Y, dtype=ScalableLatnet.FLOAT)

        S_by_N_by_N = ((S, N, N))
        S_by_N_by_2Nrf = ((S, N, 2 * flags.get_flag().n_rff))
        S_by_Nrf_by_D = ((S, flags.get_flag().n_rff, D))
        # sampling for W
        z_W = tf.random_normal(S_by_N_by_N, dtype=ScalableLatnet.FLOAT)
        with tf.name_scope('W'):
            W = tf.multiply(z_W, tf.sqrt(sigma2)) + mu
            W = tf.matrix_set_diag(W, tf.zeros((S, N), dtype=ScalableLatnet.FLOAT))
            # Latnet.variable_summaries(W)

        # Gamma
        z_G = tf.random_normal(S_by_N_by_2Nrf, dtype=ScalableLatnet.FLOAT)
        Gamma = tf.multiply(z_G, tf.sqrt(sigma2_gamma)) + mu_gamma

        # Omega
        z_O = tf.random_normal(S_by_Nrf_by_D, dtype=ScalableLatnet.FLOAT)
        Omega = tf.multiply(z_O, tf.sqrt(sigma2_omega)) + mu_omega

        # sampling for A
        u = tf.random_uniform(S_by_N_by_N, minval=0, maxval=1, dtype=ScalableLatnet.FLOAT)
        _A = tf.math.divide(
            tf.add(
                tf.log(alpha),
                tf.subtract(
                    tf.log(u + eps),
                    tf.log(tf.constant(1.0, dtype=ScalableLatnet.FLOAT) - u + eps)
                )
            ),
            posterior_lambda_)
        A = tf.sigmoid(_A)
        A = tf.matrix_set_diag(A, tf.zeros((S, N), dtype=ScalableLatnet.FLOAT))

        prior_mu, prior_sigma2, prior_mu_gamma, prior_sigma2_gamma, prior_mu_omega, prior_sigma2_omega, prior_alpha = ScalableLatnet.get_priors(
            flags,
            D, N, T)
        kl_W = ScalableLatnet.get_KL_normal(mu, sigma2, prior_mu, prior_sigma2)
        kl_G = ScalableLatnet.get_DKL_normal(mu_gamma, sigma2_gamma, prior_mu_gamma, prior_sigma2_gamma)
        kl_O = ScalableLatnet.get_DKL_normal(mu_omega, sigma2_omega, prior_mu_omega, prior_sigma2_omega)
        kl_A = ScalableLatnet.get_KL_logistic(_A, alpha, prior_lambda_, posterior_lambda_, prior_alpha)
        ell, first, second, third, eig_check = ScalableLatnet.batch_ll_new(flags, t, Y, sigma2_n, A, W, Gamma, Omega,
                                                                           variance,
                                                                           N, D, flags.get_flag().n_rff, T, S)
        return kl_W, kl_G, kl_O, kl_A, ell, first, second, third, eig_check

    @staticmethod
    def batch_ll_new(flags, t, Y, sigma2_n, A, W, Gamma, Omega, variance, N, D, N_rf, T, S):

        Omega_temp = tf.reshape(Omega, [S * N_rf, D])
        Fi_under = tf.reshape(
            tf.matmul(Omega_temp, tf.transpose(t)), [S, N_rf, T])
        Fi = tf.sqrt(tf.math.divide(variance, N_rf)) * \
             tf.concat([tf.cos(Fi_under), tf.sin(Fi_under)], axis=1)
        Z = tf.matmul(Gamma, Fi)
        I = tf.constant(np.identity(N), dtype=ScalableLatnet.FLOAT)
        with tf.name_scope('B'):
            B = tf.multiply(A, W)
            # Latnet.variable_summaries(B)
        add_noise = False
        if add_noise:
            S_N_T = (S, N, T)
            noise = np.random.normal(loc=0, scale=0.0001, size=S_N_T)
            Z = tf.add(Z, tf.matmul(B, noise))
        IB = tf.subtract(I, B)

        with tf.name_scope('eig_check'):
            eig_values, eig_vectors = tf.linalg.eigh(B)
            max_eig_check = abs(tf.reduce_max(eig_values)) < 1
            min_eig_check = abs(tf.reduce_min(eig_values)) < 1
            eig_check = max_eig_check & min_eig_check
            # Latnet.variable_summaries(eig_check)

        approximate_inverse = flags.get_flag().inv_calculation
        if approximate_inverse == 'approx':
            v_current = tf.reshape(Z, [S, N, T])
            exp_Y = v_current
            n_approx = flags.get_flag().n_approx_terms
            for i in range(n_approx):
                v_current = tf.matmul(B, v_current)
                exp_Y = tf.add(v_current, exp_Y)  # v + Bv

        elif approximate_inverse == 'solver':
            with tf.name_scope('exp_Y'):
                exp_Y = tf.linalg.solve(IB, Z)
        elif approximate_inverse == 'cholesky':
            IB_chol = ScalableLatnet.cholesky_decompose(IB)
            with tf.name_scope('exp_Y'):
                exp_Y = tf.linalg.cholesky_solve(IB_chol, Z)
        else:
            IB_inverse = tf.matrix_inverse(IB)
            with tf.name_scope('exp_Y'):
                exp_Y = tf.matmul(IB_inverse, Z)
        real_Y = tf.expand_dims(tf.transpose(Y), 0)
        with tf.name_scope('norm'):
            norm = tf.norm(exp_Y - real_Y, ord=2, axis=1)
            ScalableLatnet.variable_summaries(norm)
        norm_sum_by_T = tf.reduce_sum(norm, axis=1)
        norm_sum_by_T_avg_by_S = tf.reduce_mean(norm_sum_by_T)
        third = norm_sum_by_T_avg_by_S
        _half = tf.constant(0.5, dtype=ScalableLatnet.FLOAT)
        _two = tf.constant(2.0, dtype=ScalableLatnet.FLOAT)
        _two_pi = tf.constant(6.28, dtype=ScalableLatnet.FLOAT)
        _N = tf.constant(N, dtype=ScalableLatnet.FLOAT)
        _T = tf.constant(T, dtype=ScalableLatnet.FLOAT)
        first = tf.multiply(
            _half, tf.multiply(_N, tf.multiply(_T, tf.log(tf.multiply(_two_pi, sigma2_n)))))
        second = _half * tf.divide(norm_sum_by_T_avg_by_S, sigma2_n)
        res = -tf.add(first, second)
        return res, first, second, third, eig_check

    def cholesky_decompose(M):
        # add small noise to eigenvalues
        eig_values, eig_vectors = tf.linalg.eigh(M)
        eps = 2
        diag_eig_values = tf.linalg.diag(eig_values + eps)
        vect_diag = (tf.matmul(eig_vectors, diag_eig_values))
        vec_transp = (tf.transpose(eig_vectors, perm=[0, 2, 1]))
        M = tf.matmul(vect_diag, vec_transp)
        M_chol = tf.cholesky(M)
        return M_chol

    @staticmethod
    def optimize(flags, s, D,  t, Y, logger, callback=None):

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        ## Set random seed for tensorflow and numpy operations
        tf.set_random_seed(flags.get_flag().seed)
        np.random.seed(flags.get_flag().seed)

        with tf.Session(config=config) as sess:
            merged, var_opt, hyp_opt, elbo, kl_W, kl_A, kl_G, kl_O, ell, first, second, third, eig_check, \
            log_sigma2_n, mu, log_sigma2, mu_gamma, log_sigma2_gamma, mu_omega, log_sigma2_omega, log_alpha, \
            log_lengthscale, log_variance, var_nans, hyp_nans = \
                ScalableLatnet.run_model(flags, D, tf.cast(t, ScalableLatnet.FLOAT), Y)

            train_writer = tf.summary.FileWriter('results/fmri/fmri_sim2_scalableGPL/train', sess.graph)
            init_op = tf.initializers.global_variables()
            sess.run(init_op)

            # current global iteration over optimization steps.
            _iter = 0

            id = 0
            best_elbo = None
            while flags.get_flag().n_iterations is None or _iter < flags.get_flag().n_iterations:

                logger.debug("\nSUBJECT %d: ITERATION %d STARTED\n" %
                             (s, _iter))
                # optimizing variational parameters
                if flags.get_flag().var_steps > 0:
                    elbos = []
                    logger.debug("optimizing variational parameters")
                    for i in range(0, flags.get_flag().var_steps):
                        try:
                            output = sess.run(
                                [merged, elbo, kl_W, kl_A, kl_G, kl_O, ell, first, second, third, eig_check, var_opt,
                                 var_nans])
                            id += 1
                            cur_elbo = output[1]
                            if i % flags.get_flag().display_step == 0:
                                elbos.append(cur_elbo)
                                logger.debug(
                                    '\tlocal {:d} iter elbo: {:.0f} (KL W={:.0f}, KL A={:.0f}, KL G={:.0f},KL O={:.0f}, ell={:.0f}, first={:.0f}, second={:.0f}, third ={:f}, eig_check = {}), {:d} nan in grads (err= {:d}). '.format(
                                        i, output[1], output[2], output[3], output[4], output[5], output[6], output[7],
                                        output[8], output[9], output[10], output[12], output[12] != 0))
                                train_writer.add_summary(output[0], id)
                        except OpError as e:
                            logger.error(e.message)

                # optimizing hyper parameters
                if flags.get_flag().hyp_steps > 0:
                    elbos = []
                    logger.debug("optimizing hyper parameters")
                    for i in range(0, flags.get_flag().hyp_steps):
                        try:
                            output = sess.run(
                                [merged, elbo, kl_W, kl_A, kl_G, kl_O, ell, first, second, third, eig_check, hyp_opt,
                                 hyp_nans])
                            id += 1
                            if i % flags.get_flag().display_step == 0:
                                elbos.append(output[1])
                                logger.debug(
                                    '\tlocal {:d} iter elbo: {:.0f} (KL W={:.0f}, KL A={:.0f}, KL G={:.0f},KL O={:.0f}, ell={:.0f}, first={:.0f}, second={:.0f}, third = {:f}, eig_check = {}), {:d} nan in grads (err= {:d}). '.format(
                                        i, output[1], output[2], output[3], output[4], output[5], output[6], output[7],
                                        output[8], output[9], output[10], output[12], output[12] != 0))
                                train_writer.add_summary(output[0], id)
                        except OpError as e:
                            logger.error(e.message)
                if callback is not None:
                    sigma2_n_, mu_, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale_, variance_ = \
                        sess.run((tf.exp(log_sigma2_n), mu, tf.exp(log_sigma2), mu_gamma, tf.exp(log_sigma2_gamma),
                                  mu_omega, tf.exp(
                            log_sigma2_omega), tf.exp(log_alpha), tf.exp(log_lengthscale), tf.exp(log_variance)))
                    callback(alpha_, mu_, sigma2_, mu_gamma, sigma2_gamma_,
                             mu_omega, sigma2_omega_, sigma2_n_, lengthscale_, variance_)
                _iter += 1

            elbo_, sigma2_n_, mu_, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale, variance = \
                sess.run((elbo, tf.exp(log_sigma2_n), mu, tf.exp(log_sigma2), mu_gamma, tf.exp(log_sigma2_gamma),
                          mu_omega, tf.exp(
                    log_sigma2_omega), tf.exp(log_alpha), tf.exp(log_lengthscale), tf.exp(log_variance)))

        return elbo_, sigma2_n_, mu_, sigma2_, mu_gamma, sigma2_gamma_, mu_omega, sigma2_omega_, alpha_, lengthscale, variance
