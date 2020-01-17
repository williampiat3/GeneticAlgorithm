import unittest
import custom_fitnesses
# we can't test the fitnesses without wrapping them
from deap import creator



class TestFitnesses(unittest.TestCase):
	"""
	Test the different fitnesses 
	"""

	def __init_(self):
		super(TestUtils,self).__init__()

	def test_fitnesses(self):
		"""
		Function to test the comparison between fitnesses
		We are testing here the weighted fitness and the constrained fitness
		"""
		creator.create("FitnessWeighted",base=custom_fitnesses.FitnessReg,weights=(1.,0.5))
		creator.create("FitnessConstrained",base=custom_fitnesses.ConstrainedFitness, base_fit=creator.FitnessWeighted,weights=(1.,0.5))
		creator.create("IndividualWeighted", list, fitness=creator.FitnessWeighted)
		creator.create("IndividualConstrained", list, fitness=creator.FitnessConstrained)


		# testing weighted fitness
		indiv_reg_1 =  creator.IndividualWeighted([1,0])
		indiv_reg_2 = creator.IndividualWeighted([0,1])

		indiv_reg_1.fitness.values = (1,0)
		indiv_reg_2.fitness.values = (0,1)

		self.assertTrue(indiv_reg_1.fitness>indiv_reg_2.fitness)

		# testing constrained fitness
		indiv_cons_1 =  creator.IndividualConstrained([1,0])
		indiv_cons_2 = creator.IndividualConstrained([0,1])

		indiv_cons_1.fitness.values = (1,0)
		indiv_cons_2.fitness.values = (0,1)

		indiv_cons_1.fitness.constraints = [1,0]
		indiv_cons_2.fitness.constraints = [1,1]


		self.assertTrue(indiv_cons_1.fitness<indiv_cons_2.fitness)













if __name__ == "__main__":
	unittest.main()
