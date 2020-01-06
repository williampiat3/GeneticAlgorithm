from dna_translator import DNA_creator
from deap import creator, base, tools, algorithms
import itertools
import multiprocessing
import time
import pickle
import json
import random
from utils import generate,evalOneMax,decorator_cross,decorator_mut,decorator_selection,recover_last_gen

#Function that runs the evolutionnary algorithm (model is a function that takes a phenotype as an argument and a number of tests to make)

def run_evolution(model,hparams,NGEN=40,mut_rate=0.05,tourn_size=3,cxpb=0.5, mutpb=0.3,nb_indiv=80,nb_threads=1,log_file=None,recover_file=None,gray_code=False,**kwargs):
	"""Arguments:

	model is the black box function to be optimized: python function that returns the fitness as first argument it must take one configuration of the hyperparameters 
	hparams is the hyperparameter space defined as a python dictionnary:	
	 hparams={
					"character1":[1,2,3],
					"character2":["relu","sig"]
		}
	mut_rate is the probability of mutating one gene
	tourn_size is the size of the tournament
	cxpb is the probability of crossing two individuals, otherwise the individuals are kept as such
	mutpb is the probability to mutate an individual
	nb_indiv is the number of individuals per generation
	nb_thread is the number of processes needed for	 hparams={
					"character1":[1,2,3],
					"character2":["relu","sig"]
		}the GA to run
	log_file is the name of the file for saving the generations
	recover_file is the name of the log_file that you want to recover from
	gray_code is the binary encoding technique: if True gray code is used, binary encoding if False

	**kwargs is all the extra arguments that need to be passed to the model (a data reader for instance)
	The output of the file is a specific format: it is an array of generations, each generation an array of individuals, each individual is a python dictionnary containing the phenotype, the id,
	the id of the parents, the id of the mutated individual, the age of the individual


	"""
	if recover_file:
		with open(recover_file,'rb') as pickle_file:
			hparams=pickle.load(pickle_file)
			translator=DNA_creator(hparams,gray_code=gray_code)
			pops=pickle.load(pickle_file)
			
			creator.create("FitnessMax", base.Fitness, weights=(1.0,))
			#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
			creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)

			#toolbox object from deap: allows to define functions in one line for operations on the individual
			toolbox = base.Toolbox()
			toolbox.register("individual", generate,translator=translator,creator=creator)
			toolbox.register("population", tools.initRepeat, list, toolbox.individual)


			#deap basic funtion to cross a population 
			toolbox.register("evaluate", evalOneMax,model=model,translator=translator,**kwargs)
			toolbox.register("mate", decorator_cross(tools.cxTwoPoint))
			toolbox.register("mutate", decorator_mut(tools.mutFlipBit), indpb=mut_rate)
			toolbox.register("select", decorator_selection(tools.selTournament), tournsize=tourn_size)
			population=recover_last_gen(pops,creator,translator)

			#Performing selection and cross and mutation to create the new generation

			population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=nb_indiv)
			population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
	else:
		translator=DNA_creator(hparams)
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
		creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)

		#toolbox object from deap: allows to define functions in one line for operations on the individual
		toolbox = base.Toolbox()
		toolbox.register("individual", generate,translator=translator,creator=creator)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)


		#deap basic funtion to cross a population 
		toolbox.register("evaluate", evalOneMax,model=model,translator=translator,**kwargs)
		toolbox.register("mate", decorator_cross(tools.cxTwoPoint))
		toolbox.register("mutate", decorator_mut(tools.mutFlipBit), indpb=mut_rate)
		toolbox.register("select", decorator_selection(tools.selTournament), tournsize=tourn_size)
		#we are now creating the population 
		population = toolbox.population(n=nb_indiv)
		pops=[]
	

	init_integer=len(pops)
	# multiprocessing if the number of threads is above 1
	if nb_threads > 1:
		pool = multiprocessing.Pool(processes=nb_threads)
		toolbox.register("map",pool.map)
	#We are running the genetical algorithm
	for gen in range(init_integer,NGEN):
		print('Generation: '+str(gen+1))

		#creating ids if needed
		for l,ind in enumerate(population):
			ind.age+=1
			if ind.mutated!=None or ind.parents!=None or gen ==0:
				ind.id=str(gen+1)+"."+str(l)



		#Evaluation of the individuals
		fits = toolbox.map(toolbox.evaluate, population)
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
