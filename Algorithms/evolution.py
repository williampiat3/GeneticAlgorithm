from dna_translator import DNA_creator,HybridDNA
from deap import creator, base, tools, algorithms
import itertools
import multiprocessing
import time
import pickle
import json
import random
from utils import generate,evalOneMax,evalOneMaxMulti,decorator_cross,decorator_mut,decorator_selection,recover_last_gen,evaluate,hybrid_cx,hybrid_mut
from custom_fitnesses import FitnessReg




# Main function that will run the ga, same parameters, you can change here the 2 basic operators of the hybrid cross over and the hybrid mutation check deap tools on github for the kwargs of the operators
# Currently the default operators that are used for crossing are a for the discrete part a two point cross and for the continuous part a blend
#  and for mutation we have a bit flip for the discrete part and a gaussian noise for the continuous part
# One important extra feature here is the argument "stric_interval" that contraints the continuous variables in their interval if True. If False the ga can evolve beyond the range specified via cross over and mutation
def run_hybrid_ga(model,weights,hparams,NGEN=40,nb_indiv=80,nb_threads=1,cxpb=0.7, mutpb=0.4,discrete_cx=tools.cxTwoPoint ,discrete_cx_kwargs={},continuous_cx=tools.cxBlend,continuous_cx_kwargs={"alpha":0.1},discrete_mut=tools.mutFlipBit,discrete_mut_kwargs={"indpb":0.05},continuous_mut=tools.mutGaussian,continuous_mut_kwargs={"mu":0.,"sigma":0.1, "indpb":0.05},selection_op=tools.selTournament,selection_kwargs={"tournsize":3},stric_interval=False,log_file=None,recover_file=None,gray_code=True,**kwargs):
	if recover_file:
		with open(recover_file,'rb') as pickle_file:
			hparams=pickle.load(pickle_file)
			translator=HybridDNA(hparams,stric_interval=stric_interval,gray_code=gray_code)
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
			toolbox.register("mate",decorator_cross( hybrid_cx),discrete_cx=discrete_cx ,discrete_cx_kwargs=discrete_cx_kwargs,continuous_cx=continuous_cx,continuous_cx_kwargs=continuous_cx_kwargs)
			toolbox.register("mutate", decorator_mut(hybrid_mut), discrete_mut=discrete_mut ,discrete_mut_kwargs=discrete_mut_kwargs,continuous_mut=continuous_mut,continuous_mut_kwargs=continuous_mut_kwargs)
			toolbox.register("select",decorator_selection(selection_op), **selection_kwargs)
			population=recover_last_gen(pops,creator,translator)
			#nb_indiv=len(population)

			#Performing selection and cross and mutation to create the new generation

			population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=nb_indiv)
			population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
	else:
		translator=HybridDNA(hparams,stric_interval=stric_interval,gray_code=gray_code)
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
		toolbox.register("evaluate", evaluate,model=model,translator=translator,weights=weights,**kwargs)
		toolbox.register("mate", decorator_cross(hybrid_cx),discrete_cx=discrete_cx ,discrete_cx_kwargs=discrete_cx_kwargs,continuous_cx=continuous_cx,continuous_cx_kwargs=continuous_cx_kwargs)
		toolbox.register("mutate", decorator_mut(hybrid_mut), discrete_mut=discrete_mut ,discrete_mut_kwargs=discrete_mut_kwargs,continuous_mut=continuous_mut,continuous_mut_kwargs=continuous_mut_kwargs)
		toolbox.register("select", decorator_selection(selection_op), **selection_kwargs)
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


def cross_and_mut_pop(population, toolbox, cxpb,translator):
	# this is a rewritten piece of code from https://github.com/DEAP/deap/blob/master/deap/algorithms.py
	# the only difference is that the mutation rate is no longer a parameter of the algorithm but a parameter of the individual
	# It allows to have a dynamic adjustment of the mutation rate
	offspring = [toolbox.clone(ind) for ind in population]

	# Apply crossover and mutation on the offspring
	for i in range(1, len(offspring), 2):
		if random.random() < cxpb:
			offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
														  offspring[i])
			del offspring[i - 1].fitness.values, offspring[i].fitness.values

	for i in range(len(offspring)):
		if random.random() < translator.continuous_translator.dna_to_phen(offspring[i][1])["mutpb"]:
			offspring[i], = toolbox.mutate(offspring[i])
			del offspring[i].fitness.values

	return offspring


#Quite similar algorithm than the hybrid one however this one has an extra feature: the mutation rate is here a parameter of the individual
#Hence the format of the argument mutpb: it has to be defined as a continuous hyperparameter with a range and a scale
#If you set the argument strict_interval to the value False then the value can become lower than 0 and higher than 1 
def run_hybrid_ga_indiv_mut(model,weights,hparams,NGEN=40,nb_indiv=80,nb_threads=1,cxpb=0.7, mutpb={"range":[0.,0.4],"scale":"linear"},discrete_cx=tools.cxTwoPoint ,discrete_cx_kwargs={},continuous_cx=tools.cxBlend,continuous_cx_kwargs={"alpha":0.1},discrete_mut=tools.mutFlipBit,discrete_mut_kwargs={"indpb":0.05},continuous_mut=tools.mutGaussian,continuous_mut_kwargs={"mu":0.,"sigma":0.1, "indpb":0.05},selection_op=tools.selTournament,selection_kwargs={"tournsize":3},stric_interval=False,log_file=None,recover_file=None,gray_code=True,**kwargs):
	if recover_file:
		with open(recover_file,'rb') as pickle_file:
			hparams=pickle.load(pickle_file)
			#adding the mutation rate as a hyperparameter of the individuals
			hparams["continuous"]["mutpb"]=mutpb
			#creating the translator
			translator=HybridDNA(hparams,stric_interval=stric_interval,gray_code=gray_code)
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
			toolbox.register("mate",decorator_cross( hybrid_cx),discrete_cx=discrete_cx ,discrete_cx_kwargs=discrete_cx_kwargs,continuous_cx=continuous_cx,continuous_cx_kwargs=continuous_cx_kwargs)
			toolbox.register("mutate", decorator_mut(hybrid_mut), discrete_mut=discrete_mut ,discrete_mut_kwargs=discrete_mut_kwargs,continuous_mut=continuous_mut,continuous_mut_kwargs=continuous_mut_kwargs)
			toolbox.register("select",decorator_selection(selection_op), **selection_kwargs)
			population=recover_last_gen(pops,creator,translator)
			#nb_indiv=len(population)

			#Performing selection and cross and mutation to create the new generation

			population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=nb_indiv)
			population = cross_and_mut_pop(population, toolbox, cxpb=cxpb,translator=translator)
	else:
		#adding the mutation rate as a hyperparameter of the individuals
		hparams["continuous"]["mutpb"]=mutpb
		#creating the translator
		translator=HybridDNA(hparams,stric_interval=stric_interval,gray_code=gray_code)
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
		toolbox.register("evaluate", evaluate,model=model,translator=translator,weights=weights,**kwargs)
		toolbox.register("mate", decorator_cross(hybrid_cx),discrete_cx=discrete_cx ,discrete_cx_kwargs=discrete_cx_kwargs,continuous_cx=continuous_cx,continuous_cx_kwargs=continuous_cx_kwargs)
		toolbox.register("mutate", decorator_mut(hybrid_mut), discrete_mut=discrete_mut ,discrete_mut_kwargs=discrete_mut_kwargs,continuous_mut=continuous_mut,continuous_mut_kwargs=continuous_mut_kwargs)
		toolbox.register("select", decorator_selection(selection_op), **selection_kwargs)
		#we are now creating the population 
		population = toolbox.population(n=nb_indiv)
		
		pops=[]
	
	init_integer=len(pops)
	print(translator.discrete_translator.hparams)

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

		population = cross_and_mut_pop(population, toolbox, cxpb=cxpb,translator=translator)
		if log_file:
			with open(log_file,'wb') as pickle_file:
				pickle.dump(hparams,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
				pickle.dump(pops,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
	return pops

def test_func(hparam):
	loss=0
	for key in hparam:
		loss+=hparam[key]
	return loss,


if __name__ == '__main__':
	hparams = {"discrete":
						{"param1":[5,-3,-2,-1],
						 "param2":[2,5,7]
						},
		   "continuous":
						{"param3":{"range":[0,9],"scale":"linear"},
						 "param4":{"range":[0.1,100],"scale":"log"}

		   }
		   }
	pops = run_hybrid_ga(test_func,(1.,),hparams,NGEN=30,nb_indiv=100,cxpb=0.6,log_file="results_test.pk", stric_interval=True)
