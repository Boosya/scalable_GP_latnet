from numpy import sqrt


# tf_upgrade_v2 \
#   --infile scalable_GP_latnet/scalable_latnet/scalablelatnet.py \
#   --outfile scalable_GP_latnet/scalable_latnet/scalablelatnet_v2.py

class Flags():
    def __init__(self):
        # learn_Omega - How to treat Omega - fixed (from the prior), optimized, or learned variationally
        #               [prior-fixed, var-fixed, var-resampled]

        # inv_calculation -  Way to approximate inverse of the matrix : [approx, solver, cholesky, matrix_inverse]
        self.flags = {'seed': 1, 'validation_percent': 0.0, 'test_percent': 0.0,
                      'return_best_state': False, 'learn_Omega': 'var-fixed', 'learn_lengthscale': 'yes',
                      'inv_calculation': 'solver', 'n_approx_terms': 5, 'n_iterations': 5, 'var_steps': 500,
                      'hyp_steps': 10000, 'all_steps': 0, 'display_step': 10, 'var_learning_rate': 0.01,
                      'all_learning_rate': 0.001, 'hyp_learning_rate': 0.0001, 'n_mc': 50, 'n_rff': 500,
                      'prior_lambda_': 0.5, 'posterior_lambda_': 2. / 3., 'init_sigma2_n': 0.06, 'init_variance': 0.6,
                      'init_lengthscale': 2.0, 'init_p': 0.5, 'print_auc': False}

    def get_flag(self, key):
        return self.flags.get(key)

    def set_flags(self, new_flags):
        return self.flags.update(new_flags)

    def del_all_flags(self):
        del self.flags

    def log_flags(self, logger):
        for flag in self.flags:
            logger.debug("\t {}: {}".format(flag, self.flags.get(flag)))
