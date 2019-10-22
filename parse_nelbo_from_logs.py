
import re
import csv

RESULTS='results/'
N_OBJECTS =50
elbo_pattern = re.compile('elbo: (-?\d+) ')
auc_pattern = re.compile('auc: (0\.\d\d) ')
methods = ['latnet','scalableGPL']

for method in methods:
	for sims in ['sim1','sim2','sim3']:
		"""sims: which simulation in the dataset """
		for Ti in [50, 100, 200]:
			"""Ti: number of observations"""
			result_output_folder = RESULTS + 'fmri/' + 'fmri_' + sims + '_' + method + '/' + str(Ti)
			for s in range(50):
				output_filename = 'nelbos/fmri_' + sims + '_' + method + '_' + str(Ti) + '_' + str(s)+ '.csv'
				with open(output_filename, 'w', newline='') as output_file:
					nelbos = []
					aucs = []
					writer = csv.writer(output_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
					log_filename = result_output_folder + '/subject_' + str(s) + '/' + 'run.log'
					for line in open(log_filename, 'r').readlines():
						elbo_match = elbo_pattern.search(line)
						if elbo_match:
							nelbo = -int(elbo_match.group(1))
							nelbos.append(nelbo)
						auc_match = auc_pattern.search(line)
						if auc_match:
							auc = int(auc_match.group(1))
							aucs.append(auc)
					writer.writerow(nelbos)
					writer.writerow(aucs)
					output_file.close()
				
