from Utils.dna_translator import DNA_creator
from deap import creator, base, tools, algorithms
import itertools
import multiprocessing
import time
import pickle
import json
import random
from Utils.utils import generate,evalOneMaxMulti,decorator_cross,decorator_mut,decorator_selection,recover_last_gen,evaluate
from Utils.custom_fitnesses import FitnessReg



"""
Few changes need to be made here contrary to the previous programm
the model function need to return a tuple of all the fitnesses, they need to be positive
weights is the different weight that you want to give to the different fitnesses of the model


"""

def run_evolution_multicriteria(model,weights,hparams,NGEN=40,mut_rate=0.05,tourn_size=3,nb_indiv=80,nb_threads=1,cxpb=0.7, mutpb=0.4,log_file=None,recover_file=None,gray_code=False,**kwargs):
	if recover_file:
		with open(recover_file,'rb') as pickle_file:
			hparams=pickle.load(pickle_file)
			translator=DNA_creator(hparams,gray_code=gray_code)
			pops=pickle.load(pickle_file)
			
			creator.create("FitnessMax", FitnessReg, weights=weights)
			#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
			creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)
			#toolbox object from deap: allows to define functions in one line for operations on the individual
			toolbox = base.Toolbox()
			toolbox.register("individual", generate,translator=translator,creator=creator)
			toolbox.register("population", tools.initRepeat, list, toolbox.individual)
			#funtion that retruns the fitness from an individual
			#the output is a tuple to match the format of the weights given just above in fitnessMax definition


			#deap basic funtion to cross a population 
			toolbox.register("evaluate", evaluate,model=model,translator=translator,weights=weights,**kwargs)
			toolbox.register("mate", decorator_cross(tools.cxTwoPoint))
			toolbox.register("mutate", decorator_mut(tools.mutFlipBit), indpb=mut_rate)
			toolbox.register("select", decorator_selection(tools.selTournament), tournsize=tourn_size)
			population=recover_last_gen(pops,creator,translator)
			#nb_indiv=len(population)

			#Performing selection and cross and mutation to create the new generation

			population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=nb_indiv)
			population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
	else:
		translator=DNA_creator(hparams)
		creator.create("FitnessMax", FitnessReg, weights=weights)
		#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
		creator.create("Individual", list, fitness=creator.FitnessMax)
		#toolbox object from deap: allows to define functions in one line for operations on the individual
		toolbox = base.Toolbox()
		toolbox.register("individual", generate,translator=translator,creator=creator)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		#funtion that retruns the fitness from an individual
		#the output is a tuple to match the format of the weights given just above in fitnessMax definition


		#deap basic funtion to cross a population 
		toolbox.register("evaluate", evalOneMaxMulti,model=model,translator=translator,weights=weights,**kwargs)
		toolbox.register("mate", decorator_cross(tools.cxTwoPoint))
		toolbox.register("mutate", decorator_mut(tools.mutFlipBit), indpb=mut_rate)
		toolbox.register("select", decorator_selection(tools.selTournament), tournsize=tourn_size)
		#we are now creating the population 
		population = toolbox.population(n=nb_indiv)
		
		pops=[]
	
	init_integer=len(pops)

	if nb_threads > 1:
		pool = multiprocessing.Pool(processes=nb_threads)
		toolbox.register("map",pool.map)
	#We are running the genetical algorithm
	for gen in range(init_integer,NGEN):
		print('Generation: '+str(gen+1))

		#creating ids if needed
		for l,ind in enumerate(population):
			ind.age+=1
			if  ind.mutated!=None or ind.parents!=None or gen ==0:
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
	loss1=0
	loss2=0
	for key in ["param1","param2"]:
		loss1+=hparam[key]
	for key in ["param3"]:
		loss2-=hparam[key]
	return loss1,loss2


if __name__ == '__main__':
	hparams = {
				"param1":[5,-3,-2,-1],
				"param2":[2,5,7],
				"param3":[100,1000]

		   }
		   
	pops = run_evolution_multicriteria(test_func,(1.,0.001),hparams,NGEN=30,nb_indiv=10,cxpb=0.6,log_file="results_test.pk")


