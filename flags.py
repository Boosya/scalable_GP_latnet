import tensorflow as tf
from numpy import sqrt

class Flags():
	def __init__(self, T):
		flags = tf.app.flags
		# flags.DEFINE_integer('batch_size', 50, 'Batch size.  ')
		flags.DEFINE_string("sim", "sim2", "which dataset to work with")
		flags.DEFINE_integer("Ti", 100, "number of observations per node in dataset")
		flags.DEFINE_integer("s", 25, "data for which subject to use")
		flags.DEFINE_string('learn_Omega', 'prior_fixed', 'How to treat Omega - fixed (from the prior), optimized, or learned variationally')
		flags.DEFINE_integer("seed", 1, "Seed for random tf and np operations")

		flags.DEFINE_integer('n_iterations', 7, 'Number of iterations of variational and hyper parameters learning')
		flags.DEFINE_integer('var_steps', 2000, 'Number of optimizations of variational parameters')
		flags.DEFINE_integer('hyp_steps', 2000, 'Number of optimizations of hyper parameters')
		flags.DEFINE_integer('display_step', 10, 'Display progress every FLAGS.display_step iterations')
		flags.DEFINE_float('var_learning_rate', 0.01, 'Variational learning rate')
		flags.DEFINE_float('hyp_learning_rate', 0.001, 'Hyper parameters learning rate')

		flags.DEFINE_integer('n_mc', 200, 'Number of Monte Carlo samples used to compute stochastic gradients')
		flags.DEFINE_integer('n_rff', 500, 'Number of random features for kernel approximation using random feature expansion')

		flags.DEFINE_float('lambda_prior', 1., 'Prior for lambda for concrete distribution')
		flags.DEFINE_float('lambda_postetior', 0.15, 'Posterior for lambda for concrete distribution')
		flags.DEFINE_float('init_sigma2_n', 0.31, 'Prior over observation noise variance.')
		flags.DEFINE_float('init_sigma2_g', 0.0001, 'Prior for connection noise variance')
		flags.DEFINE_float('init_variance', 0.5, 'Prior for gp variance')
		flags.DEFINE_float('init_lengthscale', 1./sqrt(T), 'Prior for gp lengthscale')
		flags.DEFINE_float('init_p', 0.5, 'Prior over p')
		flags.DEFINE_string('inv_calculation', 'solver', 'Way to approximate inverse of the matrix :[approx,solver,cholesky,matrix_inverse]')
		flags.DEFINE_integer('n_approx_terms',5,'How many terms to use in Neuman approximation')
		self.FLAGS = flags.FLAGS

	def get_flag(self):
		return self.FLAGS

	def del_all_flags(self):
		flags_dict = self.FLAGS._flags()
		keys_list = [keys for keys in flags_dict]
		for keys in keys_list:
			self.FLAGS.__delattr__(keys)
	
	def log_flags(self, logger):
		for flag in self.FLAGS.flag_values_dict():
			logger.debug("\t {}: {}".format(flag, self.FLAGS.flag_values_dict().get(flag)))


# def create_or_rewrite_dir(dir_name):
# 	if not os.path.exists(dir_name):
# 		os.makedirs(dir_name)
# 	else:
# 		shutil.rmtree(dir_name)
# 		os.makedirs(dir_name)