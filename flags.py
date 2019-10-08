from numpy import sqrt


# tf_upgrade_v2 \
#   --infile scalable_GP_latnet/scalable_latnet/scalablelatnet.py \
#   --outfile scalable_GP_latnet/scalable_latnet/scalablelatnet_v2.py

class Flags():
    def __init__(self,T,sims,Ti,s):
        self.flags = {}
        self.flags['sim'] = sims
        self.flags['Ti'] = Ti
        self.flags['s'] = s
        self.flags['seed'] = 1
        self.flags['return_best_state'] = True

        # How to treat Omega - fixed (from the prior), optimized, or learned variationally [prior-fixed, var-fixed, var-resampled]
        self.flags['learn_Omega'] = 'var-fixed'
        self.flags['learn_lengthscale'] = 'no'
        # Way to approximate inverse of the matrix :[approx,solver,cholesky,matrix_inverse]
        self.flags['inv_calculation'] = 'solver'
        self.flags['n_approx_terms'] = 5

        self.flags['n_iterations'] = 2
        self.flags['var_steps'] = 0
        self.flags['hyp_steps'] = 0
        self.flags['all_steps'] = 2000
        self.flags['display_step'] = 10
        self.flags['var_learning_rate'] = 0.01
        self.flags['all_learning_rate'] = 0.001
        self.flags['hyp_learning_rate'] = 0.01
        self.flags['n_mc'] = 10
        self.flags['n_rff'] = 500

        self.flags['lambda_prior'] = 1.
        self.flags['lambda_posterior'] = .15
        self.flags['init_sigma2_n'] = 0.31
        self.flags['init_variance'] = 0.5
        # self.flags['init_lengthscale'] = 1. / sqrt(T) #Change here 0.033119304807813776
        self.flags['init_lengthscale'] = 0.033119304807813776
        self.flags['init_p'] = 0.5

    def get_flag(self,key):
        return self.flags.get(key)

    def set_flags(self,new_flags):
        return self.flags.update(new_flags)

    def del_all_flags(self):
        del self.flags

    def log_flags(self,logger):
        for flag in self.flags:
            logger.debug("\t {}: {}".format(flag,self.flags.get(flag)))