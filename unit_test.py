import unittest
from deap.benchmarks import griewank
from deap import tools
import numpy as np
from Algorithms import run_evolution
from Algorithms import continuous_ga
from Algorithms import run_hybrid_ga

"""
This file aims at building unit tests for the algorithms develloped in the Algorithm folder with benchmark function
The test class can be inherited from, allowing you to run the tests on your own problems
"""

def test_function(param):
	"""
	Griewank testing function (the minus is to convert the problem into a maximization)
	"""
	return  -1*griewank((param["x0"],param["x1"],param["x2"]))[0],

class TestGeneticAlgorithms(unittest.TestCase):
	def __init__(self,*args,**kwargs):
		super(TestGeneticAlgorithms,self).__init__(*args,**kwargs)
		#registering test function
		self.function = test_function
		#minima for comparing the distance to the minima
		self.minima = {"x0":0,"x1":0,"x2":0}
		#distance to consider we have found the objective
		self.epsilon = 1

	def is_a_success(self,indiv):
		"""
		Function to probe if the best solution was found
		"""
		phen = indiv["phen"]
		dist = sum([(phen[key] - self.minima[key])**2 for key in self.minima])
		return dist <= self.epsilon

	def test_discrete(self):
		"""
		Function to test the discrete approach
		"""

		#parameters
		hparams={
		"x0":np.linspace(-1,1,40),
		"x1":np.linspace(-1,1,40),
		"x2":np.linspace(-1,1,40)


		}
		#model without the tuple format
		model = lambda x : self.function(x)[0]
		#running GA
		pops = run_evolution(model,hparams,NGEN=200,mut_rate=0.2,tourn_size=3,cxpb=0.5, mutpb=0.3,nb_indiv=80,nb_threads=1,log_file=None,recover_file=None,gray_code=False)
		#taking the best
		best = sorted(pops[-1], key= lambda indiv: indiv["fits"],reverse=True)[0]
		#asserting if minima found
		self.assertTrue(self.is_a_success(best))

	def test_continuous(self):
		"""
		Function to test the continuous approach
		"""

		#parameters
		hparams={
		"x0":{"range": [-1,1],"scale":"linear"},
		"x1":{"range": [-1,1],"scale":"linear"},
		"x2":{"range": [-1,1],"scale":"linear"}


		}
		pops = continuous_ga(self.function,(1.,),hparams,NGEN=200,nb_indiv=80,nb_threads=1,cxpb=0.6, mutpb=0.3,mu=0,sigma=0.1, indpb=0.3,alpha=0.1,tournsize=3,stric_interval=False,log_file=None,recover_file=None)
		#taking the best
		best = sorted(pops[-1], key= lambda indiv: indiv["fits"],reverse=True)[0]
		#asserting if minima found
		self.assertTrue(self.is_a_success(best))

	def test_hybrid(self):
		"""
		Function to test hybrid approach
		"""
		#parameters
		hparams={
		"continuous":{
		"x0":{"range": [-1,1],"scale":"linear"},
		"x1":{"range": [-1,1],"scale":"linear"}


		},
		"discrete":{"x2":np.linspace(-1,1,40)}
		}

		pops = run_hybrid_ga(self.function,(1.,),hparams,NGEN=40,nb_indiv=80,nb_threads=1,cxpb=0.7, mutpb=0.4,discrete_cx=tools.cxTwoPoint ,discrete_cx_kwargs={},continuous_cx=tools.cxBlend,continuous_cx_kwargs={"alpha":0.1},discrete_mut=tools.mutFlipBit,discrete_mut_kwargs={"indpb":0.05},continuous_mut=tools.mutGaussian,continuous_mut_kwargs={"mu":0.,"sigma":0.1, "indpb":0.05},selection_op=tools.selTournament,selection_kwargs={"tournsize":3},stric_interval=False,log_file=None,recover_file=None,gray_code=True)
		#taking the best
		best = sorted(pops[-1], key= lambda indiv: indiv["fits"],reverse=True)[0]
		#asserting if minima found
		self.assertTrue(self.is_a_success(best))



if __name__ == "__main__":
	unittest.main()


