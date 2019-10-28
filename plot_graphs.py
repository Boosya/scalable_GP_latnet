from math import log

import matplotlib.pyplot as plt
import csv

folder = 'graphs/'

experiment = 'fmri'
sims = ['sim2']
methods = ['scalableGPL']

Tis = [100]

for sim in sims:
	for method in methods:
		for Ti in Tis:
			for subject in range(1):
				path_to_file = folder+ '_'.join((experiment,sim,method,str(Ti),str(subject)))+'.csv'
				with open(path_to_file) as nelbos_file:
					graphs = csv.reader(nelbos_file, delimiter=',')
					(nelbos, auc) = graphs
					iterations = [i*10 for i in range(len(nelbos))]
					nelbo_int = [log(int(i)) for i in nelbos]
					auc_float = [float(i) for i in auc]


					fig, ax = plt.subplots()
					ax.plot(iterations, nelbo_int)
					ax.plot(auc_float)
					total_n_iterations = len(nelbos)*10
					for i in range(1,total_n_iterations//4000):
						ax.axvline(x=4000*i,color = 'r')
					ax.set(xlabel='iteration', ylabel='nelbo',
						title='_'.join((experiment,sim,method,str(Ti),str(subject))))
					ax.grid()
					fig.savefig('_'.join((experiment,sim,method,str(Ti),str(subject)))+'.png')
					plt.close('all')