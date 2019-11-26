
import re
import csv

RESULTS='results/'
N_OBJECTS =50
elbo_pattern = re.compile('elbo=-(\d+) ')
ell_pattern = re.compile('ell (\d+),')
kl_g_pattern = re.compile('kl_g (\d+).')
kl_o_pattern = re.compile('kl_o (\d+).')
kl_w_pattern = re.compile('kl_w (\d+).')
kl_a_pattern = re.compile('kl_a (\d+).')

methods = ['scalableGPL']
" local 1830 iter: elbo=-1309 (ell 730, kl_g 54357.5, kl_o 0.9, kl_w 78.1, kl_a 499.9)"
for method in methods:
	for sims in ['sim2']:
		"""sims: which simulation in the dataset """
		for Ti in [100]:
			"""Ti: number of observations"""
			result_output_folder = RESULTS + 'fmri/' + 'fmri_' + sims + '_' + method + '/' + str(Ti)
			for s in range(1):
				output_filename = 'nelbos_fmri_' + sims + '_' + method + '_' + str(Ti) + '_' + str(s)+ '.csv'
				with open(output_filename, 'w', newline='') as output_file:
					nelbos = []
					ells = []
					kl_gs = []
					kl_os = []
					kl_ws = []
					kl_as = []
					aucs = []
					writer = csv.writer(output_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
					log_filename = result_output_folder + '/subject_' + str(s) + '/' + 'run.log'
					for line in open(log_filename, 'r').readlines():
						elbo_match = elbo_pattern.search(line)
						ell_match = ell_pattern.search(line)
						kl_g_match = kl_g_pattern.search(line)
						kl_o_match = kl_o_pattern.search(line)
						kl_w_match = kl_w_pattern.search(line)
						kl_a_match = kl_a_pattern.search(line)
						if elbo_match:
							nelbos.append(-int(elbo_match.group(1)))
							ells.append(-int(ell_match.group(1)))
							kl_gs.append(-int(kl_g_match.group(1)))
							kl_os.append(-int(kl_o_match.group(1)))
							kl_ws.append(-int(kl_w_match.group(1)))
							kl_as.append(-int(kl_a_match.group(1)))
					writer.writerow(nelbos)
					writer.writerow(ells)
					writer.writerow(kl_gs)
					writer.writerow(kl_os)
					writer.writerow(kl_ws)
					writer.writerow(kl_as)
					output_file.close()
