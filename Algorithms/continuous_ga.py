from dna_translator import Continous_DNA
from deap import creator, base, tools, algorithms
import itertools
import multiprocessing
import time
import pickle
import json
import random
from utils import generate,decorator_cross,decorator_mut,decorator_selection,recover_last_gen,evaluate
from custom_fitnesses import FitnessReg

def creation_deap_classes(creator,weights):
	"""
	Function creating basic fitness and individual classes
	"""
	#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
	#the Fitness that we are using is not the lexicographic one, wa are summing the wvalues
	creator.create("FitnessMax", FitnessReg, weights=weights)
	 
	#creating the individual with extra arguments (parents, mutated, id and age for logs)
	#these parameters are changed by the decorators of the operators to trace which operator was applied to who
	creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)

def creation_tools(toolbox,model,translator,creator,weights,alpha,mu,sigma,indpb,tournsize,nb_threads,**kwargs):
	"""
	Function creating operators for the continous ga
	Warning the deap creator has to be instanciated with the class Individual created (call function creation deap classes first)
	Parameters:
		model (python callable) evaluation function 
		translator (Continuous_DNA instance): translator for interfacing genotype and phenotype
		creator (Deap creator): creator with classes Individual ans FitnessMax
		weights (tuple): weights of the metrics of the python callable
		alpha (Float): cross over parameter for allowing childs to step out of the parents' interval
		mu (Float): expectancy of the mutation
		sigma (Float): variance of the mutation
		indpb (float between 0 and 1): probability of mutating one gene
		tournsize (Integer): size of the tournament  for selection
		**kwargs: extra arguments for the model
	"""
	# generation operation: relies on the function generate_random of the translator
	toolbox.register("individual", generate,translator=translator,creator=creator)
	# a population is a repetition of individuals
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	#funtion that retruns the fitness from an individual
	toolbox.register("evaluate", evaluate,model=model,translator=translator,weights=weights,**kwargs)
	#mate function decorated for logs of parents
	toolbox.register("mate",decorator_cross( tools.cxBlend),alpha=alpha)
	#mutation function decorated for logging on which individual the mutation was done
	toolbox.register("mutate", decorator_mut(tools.mutGaussian), mu=mu,sigma=sigma, indpb=indpb)
	#selection operator decorated for changing logs
	toolbox.register("select",decorator_selection(tools.selTournament), tournsize=tournsize)

	#Multiprocessing if the user specified multiple threads
	if nb_threads > 1:
		pool = multiprocessing.Pool(processes=nb_threads)
		toolbox.register("map",pool.map)



def continuous_ga(model,weights,hparams,NGEN=40,nb_indiv=80,nb_threads=1,cxpb=0.7, mutpb=0.4,mu=0,sigma=0.3, indpb=0.05,alpha=0.1,tournsize=3,stric_interval=False,log_file=None,recover_file=None,**kwargs):
	"""
	Main function to run the genetic algorithm for continuous optimization

	Parameters:
		model (python callable) evaluation function 
		weights (tuple): weights of the metrics of the python callable
		hparams (python dict with specific structure): parameter space, the format is presented in the __main__ function
		NGEN (Integer): Number of generations/iterations
		nb_indiv (Integer):Number of individuals in the population
		nb_thread (Integer): number of processes that will run in parallel (allows a lot of speed up if the test function is costly)
		cxpb (Float between 0 and 1): probability of crossing two individuals
		mutpb (Float between 0 and 1): probability of mutating one individual
		mu (Float): expectancy of the mutation
		sigma (Float): variance of the mutation
		indpb (float between 0 and 1): probability of mutating one gene
		tournsize (Integer): size of the tournament for selection
		strict_interval (Bool): If True the solutions are constrained to the design space defined in hparams, if False the GA can evolve outside the bounds
		log_file (str): string of the log file to create to register all the executions and the relations between individuals
		recover_file (str): If you want to resume the computation from a log_file
		**kwargs: extra arguments forwarded to the model
	Outputs:
		pops (list of populations(list of individuals)) this object is written in the pickle file but you can have it unpickled as the output of the algorithm


	"""
	#if we are resuming the computation from a log_file
	if recover_file:
		with open(recover_file,'rb') as pickle_file:
			hparams=pickle.load(pickle_file)
			translator=Continous_DNA(hparams,stric_interval=stric_interval)
			pops=pickle.load(pickle_file)
			
			#creating Individual and Fitness class
			creation_deap_classes(creator,weights)

			#toolbox object from deap: allows to define functions in one line for operations on the individual
			toolbox = base.Toolbox()

			#registering the operators for running the algorithm
			creation_tools(toolbox,model,translator,creator,weights,alpha,mu,sigma,indpb,tournsize,nb_threads,**kwargs)

			#recovering the population from the log_file
			population=recover_last_gen(pops,creator,translator)

			#Performing selection and cross and mutation to create the new generation
			population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=nb_indiv)
			population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
	#otherwise initialize from
	else:
		translator=Continous_DNA(hparams,stric_interval=stric_interval)
		#creating Individual and Fitness class
		creation_deap_classes(creator,weights)

		#toolbox object from deap: allows to define functions in one line for operations on the individual
		toolbox = base.Toolbox()

		creation_tools(toolbox,model,translator,creator,weights,alpha,mu,sigma,indpb,tournsize,nb_threads,**kwargs)
		#we are now creating the population 
		population = toolbox.population(n=nb_indiv)
		
		pops=[]
	
	init_integer=len(pops)


	#Starting the main loop for the genetic algorithm
	for gen in range(init_integer,NGEN):
		print('Generation: '+str(gen+1))

		#creating ids if needed
		for l,ind in enumerate(population):
			ind.age+=1
			if  ind.mutated!=None or ind.parents!=None or gen ==0:
				ind.id=str(gen+1)+"."+str(l)

		#Evaluation of the individuals
		fits = toolbox.map(toolbox.evaluate, population)
		#assigning fitnesses
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

		#Crossing and mutating 
		population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
		if log_file:
			with open(log_file,'wb') as pickle_file:
				pickle.dump(hparams,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
				pickle.dump(pops,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
	#at the end we return all the executions even though they might have been registered in the pickle file
	return pops

#function to test the genetc optimization
def test_func(indiv):
	return -indiv["a"]**2 - indiv["b"]**2,



if __name__ == "__main__":
	#Parameter space, the structure for the continuous ga id the following one:
	hparams = {"a":{"range":[0,9],"scale":"linear"},
				"b":{"range":[0.1,100],"scale":"log"}}

	#running the evolution with default parameters on this simple prblem of a convex function
	pops = continuous_ga(test_func,(1.,),hparams,NGEN=40,nb_indiv=80,nb_threads=1,cxpb=0.7, mutpb=0.4,mu=0,sigma=0.3, indpb=0.05,alpha=0.1,tournsize=3,stric_interval=False,log_file=None,recover_file=None)
