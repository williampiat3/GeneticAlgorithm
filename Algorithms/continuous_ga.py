import operator
import random

import numpy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import math
import multiprocessing

#Here is a continous GA for real values

def mapping_scales(normed_param,range_param,scale="linear"):
	if scale=="linear":
		output=range_param[0]+normed_param*(range_param[1]-range_param[0])
	if scale == "log":
		log_scale=[math.log(range_param[0]),math.log(range_param[1])]
		output=math.exp(log_scale[0]+normed_param*(log_scale[1]-log_scale[0]))
	return output


#function that optimises a process using genetic algorithm
# the format of hparams is the following one:
# hparams={
# "parameter1":{"range":[1,9],
# 			  "scale":"linear"},
# "parameter2":{"range":[3,20],
# 			  "scale":"log"},

# }
# where range is the range of space in which you want the particle to evolve
# and scale is a string telling if you want the particle to evolve in a linear scale or a log scale on the range
def generate(size, pmin, pmax):
		part = creator.Individual(random.uniform(pmin, pmax) for _ in range(size)) 
		return part

	#function that translates the information of the particle in a metaparameter dictionnary
def translation(part,hparams):
		hparam={}
		i=0
		for key in hparams.keys():
			hparam[key]=mapping_scales(part[i],hparams[key]["range"],hparams[key]["scale"])
			i+=1
		return hparam
	#function that evaluates the model and returns the accuracy
def evaluation(part,model,hparams,**kwargs):
		hparam=translation(part,hparams)
		accuracy=model(hparam,**kwargs)
		return accuracy,


def run_ga(hparams, model,nb_individus=100,nb_generations=1000,nb_threads=1,mu=0,sigma=0.3, indpb=0.05,cxpb=0.5, mutpb=0.1,**kwargs):
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)
	keys_hparams=list(hparams.keys())
	#function that creates one particle
	
	#registering the function in the deap toolbox
	toolbox = base.Toolbox()
	toolbox.register("individual", generate, size=len(keys_hparams), pmin=0., pmax=1.)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("select", tools.selTournament, tournsize=3)
	toolbox.register("evaluate",evaluation,model=model,hparams=hparams,**kwargs)
	toolbox.register("mate", tools.cxBlend,alpha=0.1)
	toolbox.register("mutate", tools.mutGaussian,mu=mu,sigma=sigma, indpb=indpb)
	if nb_threads>1:
		pool = multiprocessing.Pool(processes=nb_threads)
		toolbox.register("map",pool.map)


	pop = toolbox.population(n=nb_individus)
	stats = tools.Statistics(lambda ind: ind.fitness.values)


	logbook = tools.Logbook()
	logbook.header = ["gen","best","params"]

		
	for gen in range(nb_generations):
		print('Generation: '+str(gen+1))
		offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
		fits = toolbox.map(toolbox.evaluate, offspring)
		for fit, ind in zip(fits, offspring):
			ind.fitness.values = fit
		pop = toolbox.select(offspring, k=len(pop))
		top1 = tools.selBest(pop, k=1)
		logbook.record(gen=gen,best=top1[0].fitness.values[0],params=translation(top1[0],hparams))
		print(logbook.stream)

	return True