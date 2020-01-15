import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from functools import wraps
from operator import mul


class Visualizer():
	"""
	Class that you can inherit from to create your 
		This class is a bit special each function that you want to use for building a vizualization needs to be flagged
		starting the functions with "draw_"
		__call__(**kwargs) will forward all the kwargs to all these functions so don't name arguments the same if they are not meant to be the same

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
		"""
		Function to aggragete the best n individuals: n being defined by number_individuals
		Arguments:
			pops (a list of populations): in the format given by the constrainted algorithm
			number_individuals (Integer): number of best individuals to average
		"""
		aggregated_indivs=[]
		keys = pops[0][0]["phen"].keys()
		for pop in pops:
			#selecting the n best individuals
			indivs_selected = sorted(pop,key=lambda indiv: (sum(indiv["constraints"]),sum(map(mul,indiv["fits"],self.weights))),reverse=True)[:number_individuals]


			#averaging the constraints
			average_constraint = np.stack([np.array(indiv["constraints"]).astype(int) for indiv in indivs_selected],axis=0).mean(0)


			#averaging the fitnesses
			average_fitness = sum(map(mul,np.stack([indiv["fits"] for indiv in indivs_selected],axis=0).mean(0),self.weights))

			#averaging phenotype 
			average_phen = {}
			for param in keys:
				average_phen[param] = np.mean([indiv["phen"][param] for indiv in indivs_selected])
			aggregated_indivs.append({"phen":average_phen,"constraints":average_constraint,"fits":average_fitness})
		return aggregated_indivs

	def __call__(self,**kwargs):
		for attr in dir(self):
			if "draw_" == attr[:5]:
				plt.figure()
				getattr(self,attr)(**kwargs)
		plt.show()

	def draw_constraints(self,titles,**kwargs):
		"""
		Function to draw the constraint in a heatmap
		Arguments:
			titles (list): titles of the constraints
			**kwargs aguments used for other drawing function
		"""

		data = np.stack([indiv["constraints"] for indiv in self.aggregate],axis=1)
		sns.heatmap(data,vmin=0,vmax=1,cmap="coolwarm_r",yticklabels=titles)
		plt.yticks(rotation="horizontal")

	def draw_fitness(self,title_fitness,**kwargs):
		"""
		Function to draw the evolution of the fitness along the generations
		Arguments:
			title_fitness (str): title of the plot for the fitnesses
		"""
		#if there is only one fitness
		if isinstance(self.aggregate[0]["fits"],np.float64):
			values = [indiv["fits"] for indiv in self.aggregate]
		#if there are multiple fitnesses
		else:
			values = [sum(map(mul,indiv["fits"],self.weights)) for indiv in self.aggregate]
		plt.scatter(range(len(values)),values)
		plt.title(title_fitness)




if __name__ == "__main__":
	#Path of the
	path_pickle= "/home/will/github/GeneticAlgorithm/Algorithms/results_constraints.pk"
	viz = Visualizer(path_pickle,80,(1,))
	arguments={
	"titles":["const1","const2"],
	"title_fitness":"Average weighted fitness"


	}

	viz(**arguments)



