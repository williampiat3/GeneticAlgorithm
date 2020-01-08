from Utils.dna_translator import DNA_creator
from deap import creator, base, tools, algorithms
import itertools
import multiprocessing
import time
import pickle
import json
import random
from Utils.utils import generate,evalOneMax,decorator_cross,decorator_mut,decorator_selection,recover_last_gen



def creation_deap_classes(creator):
	"""
	Function creating basic fitness and individual classes
	"""
	creator.create("FitnessMax", base.Fitness, weights=(1.,))
	 
	#creating the individual with extra arguments (parents, mutated, id and age for logs)
	#these parameters are changed by the decorators of the operators to trace which operator was applied to whom
	creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)

def creation_tools(toolbox,model,translator,creator,mutpb,tourn_size,nb_threads,**kwargs):
	"""
	Function creating operators for the continous ga
	Warning the deap creator has to be instanciated with the class Individual created (call function creation deap classes first)
	Parameters:
		model (python callable) evaluation function 
		translator (Continuous_DNA instance): translator for interfacing genotype and phenotype
		creator (Deap creator): creator with classes Individual ans FitnessMax
		weights (tuple): weights of the metrics of the python callable
		alpha (Float): cross over parameter for allowing childs to step out of the parents' interval
		mutpb (float between 0 and 1): probability of mutating one gene
		tourn_size (Integer): size of the tournament  for selection
		nb_threads (Integer): number of process to run in parallel
		**kwargs: extra arguments for the model
	"""
	# generation operation: relies on the function generate_random of the translator
	toolbox.register("individual", generate,translator=translator,creator=creator)
	# a population is a repetition of individuals
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	#function that returns the fitness from an individual
	toolbox.register("evaluate", evalOneMax,model=model,translator=translator,**kwargs)
	#mate function decorated for logs of parents
	toolbox.register("mate", decorator_cross(tools.cxTwoPoint))
	#mutation function decorated for logging on which individual the mutation was done
	toolbox.register("mutate", decorator_mut(tools.mutFlipBit), indpb=mutpb)
	#selection operator decorated for changing logs
	toolbox.register("select", decorator_selection(tools.selTournament), tournsize=tourn_size)

	#Multiprocessing if the user specified multiple threads
	if nb_threads > 1:
		pool = multiprocessing.Pool(processes=nb_threads)
		toolbox.register("map",pool.map)


#Function that runs the evolutionnary algorithm (model is a function that takes a phenotype as an argument and a number of tests to make)

def run_evolution(model,hparams,NGEN=40,mut_rate=0.05,tourn_size=3,cxpb=0.5, mutpb=0.3,nb_indiv=80,nb_threads=1,log_file=None,recover_file=None,gray_code=False,**kwargs):
	"""
	Arguments:

		model (python callable): black box function to be optimized: python function that returns the fitness as first argument it must take one configuration of the hyperparameters 
		hparams (python dictionnary): parameter space defined in the following format:	
		 hparams={
						"character1":[1,2,3],
						"character2":["relu","sig"]
			}
		mut_rate (float between 0 and 1): the probability of mutating one gene
		tourn_size (int): the size of the tournament
		cxpb (float between 0 and 1): probability of crossing two individuals, otherwise the individuals are kept as such
		mutpb (float between 0 and 1): probability to mutate an individual
		nb_indiv (int): number of individuals per generation
		nb_thread (int): number of processes needed for	the GA to run
		log_file (str): name of the file for saving the generations
		recover_file (str): name of the log_file that you want to recover from
		gray_code (bool): binary encoding technique: if True gray code is used, binary encoding if False

		**kwargs is all the extra arguments that need to be passed to the model (a data reader for instance)
	Outputs:
		pops (list of populations(list of individuals)) this object is written in the pickle file but you can have it unpickled as the output of the algorithm


	"""
	if recover_file:
		with open(recover_file,'rb') as pickle_file:
			hparams=pickle.load(pickle_file)
			translator=DNA_creator(hparams,gray_code=gray_code)
			pops=pickle.load(pickle_file)

			#creating Individual and Fitness class
			creation_deap_classes(creator)

			#toolbox object from deap: allows to define functions in one line for operations on the individual
			toolbox = base.Toolbox()

			#registering the operators for running the algorithm
			creation_tools(toolbox,model,translator,creator,mutpb,tourn_size,nb_threads,**kwargs)

			population=recover_last_gen(pops,creator,translator)

			#Performing selection and cross and mutation to create the new generation

			population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=nb_indiv)
			population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
	else:
		translator=DNA_creator(hparams,gray_code=gray_code)

		#creating Individual and Fitness class
		creation_deap_classes(creator)

		#toolbox object from deap: allows to define functions in one line for operations on the individual
		toolbox = base.Toolbox()
		#registering the operators for running the algorithm
		creation_tools(toolbox,model,translator,creator,mutpb,tourn_size,nb_threads,**kwargs)
		#we are now creating the population 
		population = toolbox.population(n=nb_indiv)
		pops=[]
	

	init_integer=len(pops)

	#We are running the genetic algorithm
	for gen in range(init_integer,NGEN):
		print('Generation: '+str(gen+1))

		#creating ids if needed
		for l,ind in enumerate(population):
			ind.age+=1
			if ind.mutated!=None or ind.parents!=None or gen ==0:
				ind.id=str(gen+1)+"."+str(l)



		#Evaluation of the individuals
		fits = toolbox.map(toolbox.evaluate, population)
		#assigning fitness
		for fit, ind in zip(fits, population):
			ind.fitness.values = fit

		# Printing the best individual
		top1 = tools.selBest(population, k=1)
		print(translator.dna_to_phen(top1[0]))
		print(top1[0].fitness.values)


		#Registering the information in the pickle file
		pops.append([{"phen":translator.dna_to_phen(indiv),"fits":indiv.fitness.values,"age":indiv.age,"id":indiv.id,"parents":indiv.parents,"mutated":indiv.mutated} for indiv in population if translator.is_dna_viable(indiv)])




		#Selection of the individuals
		population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=len(population))

		population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
		
		if log_file:
			with open(log_file,'wb') as pickle_file:
				pickle.dump(hparams,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
				pickle.dump(pops,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
	return pops


def test_func(hparam):
	loss=0
	for key in hparam:
		loss+=hparam[key]
	return loss


if __name__ == '__main__':
	hparams = {
				"param1":[5,-3,-2,-1],
				"param2":[2,5,7]

		   }
		   
	pops = run_evolution(test_func,hparams,NGEN=30,nb_indiv=100,cxpb=0.6,log_file="results_test.pk")
