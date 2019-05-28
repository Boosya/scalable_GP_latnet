import matplotlib.pyplot as plt
import csv

folder = 'nelbos/'

experiment = 'fmri'
sims = ['sim1']
methods = ['latnet']
Tis = [50]

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
					ax.axvline(x=4000,color = 'r')
					ax.axvline(x=4000*2, color = 'r')
					ax.axvline(x=4000*3, color = 'r')
					ax.axvline(x=4000*4, color = 'r')
					ax.axvline(x=4000*5, color = 'r')
					ax.axvline(x=4000*6, color = 'r')
					ax.set(xlabel='iteration', ylabel='nelbo',
						title='_'.join((experiment,sim,method,str(Ti),str(subject))))
					ax.grid()

					fig.savefig(folder+ '_'.join((experiment,sim,method,str(Ti),str(subject)))+'.png')
					# plt.show()