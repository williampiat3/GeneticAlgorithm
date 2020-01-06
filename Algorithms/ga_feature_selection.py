#/******************************************************************************
# * Original Author: Team UQ/ML                                                *
# *                                                                            *
# * Creation Date  : May 2019                                                  *
# * Version        : 1.0                                                       *
# *----------------------------------------------------------------------------*
# * Description:                                                               *
# * This programm use a genetical approach for feature selection               *
# *----------------------------------------------------------------------------*
# * Comment: Fully handcrafted                                                 *
# *                                                                            *
# ******************************************************************************/
import pickle
import random
import numpy as np
from deap import creator, base, tools, algorithms
from GA.evolution import decorator_cross,decorator_mut,decorator_selection

#function to generate a random individual
def generate_individual(creator,min_index=0,max_index=12,length=3):
	indiv=creator.Individual(np.random.choice(list(range(min_index,max_index)),size=length,replace=False).tolist())
	indiv.parents=None
	indiv.mutated=None
	indiv.id=None
	indiv.age=0
	return indiv

#function to cross too individuals
# the function here takes all the genes selected in the two individuals and samples from it
# it is done twice so that we generate 2 new individuals out of the 2 previous
def cross_over(indiv1,indiv2):
	# the cross over of two individuals here is composed 
	smallest = min(len(indiv1),len(indiv2))
	largest = max(len(indiv1),len(indiv2))
	return tuple([creator.Individual(np.random.choice(list(set(indiv1+indiv2)),size=random.randint(smallest,largest),replace=False)) for _ in range(2)])

#function to mutate an individual
#this function is adding randomly (with probability add_mutpb) an index that is not present in the initial genotype
#it removes randomly (with probability rm_mutpb) a gene or multiple genes in the individual
def mutation(indiv,min_index=0,max_index=12,rm_mutpb=0.3,add_mutpb=0.3):
	#we add randomly a column in the genotype

	try:
		while random.random()<add_mutpb:
			#Create the list of the available genes
			list_available_gene = [i for i in range(min_index,max_index) if i not in indiv]
			#sample from it
			indiv.append(np.random.choice(list_available_gene,replace=False))
		#we remove randomly genes
		while random.random()<rm_mutpb:
			indiv.pop(random.randint(0,len(indiv)-1))
		return indiv,
	except ValueError:
		return indiv,
#function to make sure that the individual will be smaller than a max length
#it is removing random index until the individual has reached the desired size
def trimmering(indiv,length_max=15):
	while len(indiv)>length_max:
		indiv.pop(random.randint(0,len(indiv)-1))
	return indiv,

	pop=generate_random_population(number_of_individuals, size_genotype)
#function for evaluating the individual using kwargs
def evaluate(indiv,model,**kwargs):
	return model(indiv,**kwargs),

# test function for assesing the ga
def model(indiv):
	return sum(indiv)


def recover_last_gen(pops,creator):

	last_gen = pops[-1]
	last_pop = []
	#for each last entry in the log file we create an Individual with the right parameters
	for ind in last_gen:
		indiv = creator.Individual(ind["phen"])
		indiv.id = ind["id"]
		indiv.parents = ind["parents"]
		indiv.mutated = ind["mutated"]
		indiv.age = ind["age"]
		indiv.fitness.values = ind["fits"]
		last_pop.append(indiv)
	return last_pop



#the function that will find the best features
def run_genetic_selection(model,nb_indiv,nb_gen,min_index,max_index,rm_mutpb=0.5,add_mutpb=0.5,length_max=20,tourn_size=3,length_init=5,cxpb=0.5,mutpb=0.2,log_file=None,recover_file=None,**kwargs):
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
	creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)

	#toolbox object from deap: allows to define functions in one line for operations on the individual
	toolbox = base.Toolbox()
	toolbox.register("individual", generate_individual,creator=creator,min_index=min_index,max_index=max_index,length=length_init)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)


	#deap basic funtion to cross a population 
	toolbox.register("evaluate", evaluate,model=model,**kwargs)
	#We decorate the operators so as to create the visualization the parameters given here are the parameters of the operators
	# they can be changed in the parameters of the function
	toolbox.register("mate", decorator_cross(cross_over))
	toolbox.register("mutate", decorator_mut(mutation),min_index=min_index,max_index=max_index,rm_mutpb=rm_mutpb,add_mutpb=add_mutpb)
	toolbox.register("select", decorator_selection(tools.selTournament), tournsize=tourn_size)
	toolbox.register("trimmering",trimmering,length_max=length_max)
	if recover_file:
		with open(recover_file,'rb') as pk_file:
			pops=pickle.load(pk_file)
		population=recover_last_gen(pops,creator)
		start=len(pops)
	else:
		#we are now creating the population 
		population = toolbox.population(n=nb_indiv)
		pops=[]
		start=0
		


	#We are running the genetical algorithm
	for gen in range(start,nb_gen):
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
		print(top1[0])
		print(top1[0].fitness.values)


		#Registering the information in the pickle file
		pops.append([{"phen":list(indiv),"fits":indiv.fitness.values,"age":indiv.age,"id":indiv.id,"parents":indiv.parents,"mutated":indiv.mutated} for indiv in population ])




		#Selection of the individuals
		population = toolbox.select([indiv for indiv in population ], k=len(population))

		population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
		for l,ind in enumerate(population):
			ind = toolbox.trimmering(ind)
		
		if log_file:
			with open(log_file,'wb') as pickle_file:
				#pickle.dump(hparams,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
				pickle.dump(pops,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
	return pops



if __name__ == '__main__':
	run_genetic_selection(model,nb_indiv=100,nb_gen=40,min_index=0,max_index=12,rm_mutpb=0.3,add_mutpb=0.3,length_max=6,tourn_size=3,cxpb=0.5,mutpb=0.5,log_file=None)