
import re
import csv

RESULTS='results/'
N_OBJECTS =50
kl_pattern = re.compile('kl_g (\d+\.\d),')

methods = ['scalableGPL']

for method in methods:
	for sims in ['sim1']:
		"""sims: which simulation in the dataset """
		for Ti in [50]:
			"""Ti: number of observations"""
			result_output_folder = RESULTS + 'fmri/' + 'fmri_' + sims + '_' + method + '/' + '/basic50'
			for s in range(1):
				output_filename = 'nelbos_fmri_' + sims + '_' + method + '_' + str(Ti) + '_' + str(s)+ '.csv'
				with open(output_filename, 'w', newline='') as output_file:
					nelbos = []
					aucs = []
					writer = csv.writer(output_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
					log_filename = result_output_folder + '/subject_' + str(s) + '/' + 'run.log'
					for line in open(log_filename, 'r').readlines():
						elbo_match = kl_pattern.search(line)
						if elbo_match:
							nelbo = -float(elbo_match.group(1))
							nelbos.append(nelbo)
					writer.writerow(nelbos)
					output_file.close()
				
