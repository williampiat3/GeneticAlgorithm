import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from functools import wraps
from operator import mul


class Visualizer():
	"""
	Class that you can inherit from
		This class is a bit special each function that you want to use for building a vizualization needs to be flagged
		using the @Toplot decorator

	"""

	def __init__(self,pickle_file,number_individuals,weights):
		"""
		Constructor
		Arguments:
			pickle_file (str): path of pickle file for the constrainted ga log_file 
			number_of_individuals (Integer): number of best individuals to consider
			weights (tuple): weighting the fitnesses
		"""
		with open(pickle_file,'rb') as pk_file:
			self.hparams=pk.load(pk_file)
			self.pops=pk.load(pk_file)
		self.weights = weights
		self.number_individuals = number_individuals
		self.aggregate = self.aggregate_indiv(self.pops,self.number_individuals)


	def aggregate_indiv(self,pops,number_individuals):
		aggregated_indivs=[]
		keys = pops[0][0]["phen"].keys()
		for pop in pops:
			#selecting the n best individuals
			indivs_selected = sorted(pop,key=lambda indiv: (sum(indiv["constraints"]),sum(map(mul,indiv["fits"],self.weights))),reverse=True)[:number_individuals]


			#averaging the constraints
			average_constraint = np.stack([np.array(indiv["constraints"]).astype(int) for indiv in indivs_selected],axis=0).mean(0)


			#averaging the fitnesses
			average_fitness = sum(map(mul,np.stack([indiv["fits"] for indiv in indivs_selected],axis=0).mean(0),self.weights))

			average_phen = {}
			for param in keys:
				average_phen[param] = np.mean([indiv["phen"][param] for indiv in indivs_selected])
			aggregated_indivs.append({"phen":average_phen,"constraints":average_constraint,"fits":average_fitness})
		return aggregated_indivs

	def __call__(self,**kwargs):
		for attr in dir(self):
			if "draw_" == attr[:5]:
				getattr(self,attr)(**kwargs)
		plt.show()

	def draw_constraints(self,titles,**kwargs):
		plt.figure()

		data = np.stack([indiv["constraints"] for indiv in self.aggregate],axis=1)
		sns.heatmap(data,vmin=0,vmax=1,cmap="coolwarm_r",yticklabels=titles)
		plt.yticks(rotation="horizontal")



if __name__ == "__main__":
	path_pickle= "/home/will/github/GeneticAlgorithm/Algorithms/results_constraints.pk"
	viz = Visualizer(path_pickle,80,(1,))
	arguments={
	"titles":["const1","const2"]


	}

	viz(**arguments)



