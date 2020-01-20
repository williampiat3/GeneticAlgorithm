import unittest
from utils import generate,evalOneMax,decorator_cross,decorator_mut,decorator_selection,recover_last_gen,evaluate
from dna_translator import DNA_creator
from deap import creator, tools, base

@decorator_cross
def dummy_cross(indiv1,indiv2):
	return indiv1,indiv2

@decorator_mut
def dummy_mut(indiv):
	return indiv,

@decorator_selection
def dummy_selection(indivs):
	return indivs


class TestUtils(unittest.TestCase):


	def __init__(self,*args,**kwargs):
		"""
		Constructor that creates the basic classes on which we will tes the operations
		"""
		super(TestUtils,self).__init__(*args,**kwargs)
		creator.create("FitnessMax", base.Fitness, weights=(1.,))
		#creating the individual with extra arguments (parents, mutated, id and age for logs)
		#these parameters are changed by the decorators of the operators to trace which operator was applied to whom
		creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)
		hparams = {"a":[1,0,8,9],"b":[2,5]}
		translator=DNA_creator(hparams,gray_code=False)

		self.toolbox = base.Toolbox()
		# generation operation: relies on the function generate_random of the translator
		self.toolbox.register("individual", generate,translator=translator,creator=creator)
		# a population is a repetition of individuals
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
		#function that returns the fitness from an individual
		self.toolbox.register("evaluate_one", evalOneMax,model=lambda x: 1,translator=translator)
		#function 2 for evaluation
		self.toolbox.register("evaluate_multi", evaluate,model=lambda x: (1,),weights=(1.,),translator=translator)
		#mate function decorated for logs of parents
		self.toolbox.register("mate", dummy_cross)
		#mutation function decorated for logging on which individual the mutation was done
		self.toolbox.register("mutate", dummy_mut)
		#selection operator decorated for changing logs
		self.toolbox.register("select", dummy_selection)

	def test_decorated_ops(self):
		#intializing pop
		population = self.toolbox.population(n=10)
		gen = 0
		#giving ids to the individuals
		for l,ind in enumerate(population):
			ind.age+=1
			if ind.mutated!=None or ind.parents!=None or gen ==0:
				ind.id=str(gen+1)+"."+str(l)
		#testing first attributes
		for ind in population:
			self.assertTrue(ind.parents is None)
			self.assertTrue(ind.mutated is None)
			self.assertTrue(ind.id is not None)
			self.assertTrue(ind.age == 1)
		#testing cross over
		population[0],population[1] = self.toolbox.mate(population[0],population[1])
		for ind in [population[0],population[1]]:
			#attribute parents has been changed by the decorator 
			self.assertTrue(ind.parents is not None)
		#testing mutation
		population[3], = self.toolbox.mutate(population[3])
		#attribute mutated has been changed by the decorator
		self.assertTrue(population[3].mutated is not None)
		#testing evaluation operators
		fits_one = self.toolbox.map(self.toolbox.evaluate_one,population)
		fits_multi = self.toolbox.map(self.toolbox.evaluate_multi,population)
		#simple check to see if we have similar objects
		self.assertTrue(all([fit1==fit2 for fit1,fit2 in zip(fits_one,fits_multi)]))

		#selction operator 
		population = self.toolbox.select(population)
		for ind in population:
			#checking if attributes were reset
			self.assertTrue(ind.parents is None)
			self.assertTrue(ind.mutated is None)
			self.assertTrue(ind.id is not None)
		#some new individuals were created so their age is equal to 0
		self.assertTrue(not all([(ind.age == 1) for ind in population]))


if __name__ == "__main__":
	unittest.main()




