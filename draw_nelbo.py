import matplotlib.pyplot as plt
import csv

folder = 'nelbos/'

experiment = 'fmri'
sims = ['sim1','sim2','sim3']
methods = ['latnet','scalableGPL']

Tis = [50,100,200]

for sim in sims:
	for method in methods:
		for Ti in Tis:
			for subject in range(50):
				path_to_file = folder+ '_'.join((experiment,sim,method,str(Ti),str(subject)))+'.csv'
				with open(path_to_file) as nelbos_file:
					nelbos = csv.reader(nelbos_file, delimiter=',')
					for nelbo in nelbos:
						continue
					iterations = [i*10 for i in range(len(nelbo))]
					nelbo_int = [int(i) for i in nelbo]


					fig, ax = plt.subplots()
					ax.plot(iterations, nelbo_int)
					total_n_iterations = len(nelbo)*10
					for i in range(1,total_n_iterations//4000):
						ax.axvline(x=4000*i,color = 'r')
					ax.set(xlabel='iteration', ylabel='nelbo',
						title='_'.join((experiment,sim,method,str(Ti),str(subject))))
					ax.grid()
					fig.savefig('_'.join((experiment,sim,method,str(Ti),str(subject)))+'.png')
					plt.close('all')