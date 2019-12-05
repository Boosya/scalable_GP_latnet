from numpy import sqrt


# tf_upgrade_v2 \
#   --infile scalable_GP_latnet/scalable_latnet/scalablelatnet.py \
#   --outfile scalable_GP_latnet/scalable_latnet/scalablelatnet_v2.py

class Flags():
    def __init__(self, sims, Ti, s):
        # learn_Omega - How to treat Omega - fixed (from the prior), optimized, or learned variationally
        #               [prior-fixed, var-fixed, var-resampled]

        # inv_calculation -  Way to approximate inverse of the matrix : [approx, solver, matrix_inverse]
        self.flags = {'sim': sims, 'Ti': Ti, 's': s, 'seed': 1, 'test_percent': 0.0,
                      'learn_Omega': 'var-fixed', 'learn_lengthscale': 'yes',
                      'inv_calculation': 'approx', 'n_approx_terms': 3, 'n_iterations': 7, 'var_steps': 2000,
                      'hyp_steps': 2000, 'display_step': 10, 'var_learning_rate': 0.01,
                      'hyp_learning_rate': 0.001, 'n_mc': 200, 'n_rff': 500,
                      'prior_lambda_': 1., 'posterior_lambda_': .15, 'init_sigma2_n': 0.31, 'init_variance': 0.5,
                      'init_lengthscale': 1. / sqrt(Ti), 'init_p': 0.5, 'tensorboard': False, 'kl_g_weight': 1}

    def get_flag(self, key):
        return self.flags.get(key)

    def set_flags(self, new_flags):
        return self.flags.update(new_flags)

    def del_all_flags(self):
        del self.flags

    def log_flags(self, logger):
        for flag in self.flags:
            logger.debug("\t {}: {}".format(flag, self.flags.get(flag)))
