import pickle
import random
import numpy as np
from deap import creator, base, tools, algorithms
from .Utils import decorator_cross,decorator_mut,decorator_selection

#In this program the genetic algorithm selects different indexes and forward the list of indexes to the model
#the black box must then exploit this list of indexes and return a performance metric:
#for instance: training a model on the selected indexes and returning the Q square


def generate_individual(creator,min_index=0,max_index=12,length=3):
	"""
	Function to generate a random individual
	An individual is a a set of integers with a variable length
	Arguments:
		creator (deap creator instance): used for creating an instance of individual
		min_index (Integer): min index that can be found in the list
		max_index (Integer): max index that can be found in the list
		length (Integer): Length of the generated individuals
	Outputs:
		indiv (deap Individual)
	"""
	indiv=creator.Individual(np.random.choice(list(range(min_index,max_index)),size=length,replace=False).tolist())
	indiv.parents=None
	indiv.mutated=None
	indiv.id=None
	indiv.age=0
	return indiv


def cross_over(indiv1,indiv2):
	"""
	Function to cross too individuals
	the function here takes all the genes selected in the two individuals and samples from it
	it is done twice so that we generate 2 new individuals out of the 2 previous.
	The child is composed of indexes from its parents with a length taken randomly in the length of the parents
	Arguments:
		indiv1,indiv2 (deap individuals): parents selected
	Outputs:
		tuple of 2 indiv (deap Individual)
	"""
	
	smallest = min(len(indiv1),len(indiv2))
	largest = max(len(indiv1),len(indiv2))
	#creting the children
	child1,child2 =[creator.Individual(np.random.choice(list(set(indiv1+indiv2)),size=random.randint(smallest,largest),replace=False)) for _ in range(2)]
	#for the tags to work we need the children to be tagged just like the parents
	child1.id=indiv1.id
	child2.id=indiv2.id
	return (child1,child2)


def mutation(indiv,min_index=0,max_index=12,rm_mutpb=0.3,add_mutpb=0.3):
	"""
	Function to mutate an individual
	this function is adding randomly (with probability add_mutpb) an index that is not present in the initial genotype
	it removes randomly (with probability rm_mutpb) a gene or multiple genes in the individual
	Arguments:
		indiv (Deap individual): individual to mutate
		min_index (Integer): min index that can be found in the list
		max_index (Integer): max index that can be found in the list
		rm_mutpb (Float between 0 and 1): probability for geometric law to remove n genes
		add_mutpb (Float between 0 and 1): probability for geometric law to add n genes
	Outputs:
		tuple of one indiv (deap Individual)
	"""


	try:
		#we add randomly genes
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

def trimmering(indiv,length_max=15):
	"""
	Function to make sure that the individual will be smaller than a max length
	it is removing random index until the individual has reached the desired size
	Arguments:
		indiv (Deap individual): individual to trimmer
		length_max (Integer): maximum length of an individual

	"""
	while len(indiv)>length_max:
		indiv.pop(random.randint(0,len(indiv)-1))
	return indiv,


#function for evaluating the individual using kwargs
def evaluate(indiv,model,**kwargs):
	return model(indiv,**kwargs),

def create_deap_classes(creator,weights):
	"""
	Function creating basic fitness and individual classes
	"""
	#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
	creator.create("FitnessMax", base.Fitness, weights=weights)
	#creating individual class with attributes
	creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)


def create_tools(toolbox,model,creator,min_index,max_index,length_init,rm_mutpb,add_mutpb,tourn_size,length_max,**kwargs):
	"""
	Function creating operators
	Arguments:
		toolbox (deap toolbox instance): base.toolbox
		model (python callable) evaluation function 
		creator (Deap creator): creator with classes Individual ans FitnessMax
		min_index (Integer): min index that can be found in the list
		max_index (Integer): max index that can be found in the list
		length_init (Integer): Length of the generated individuals
		rm_mutpb (Float between 0 and 1): probability for geometric law to remove n genes
		add_mutpb (Float between 0 and 1): probability for geometric law to add n genes
		tourn_size (Integer): size of the tournament
		length_max (Integer): maximum length of an individual


	"""
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



#the function that will find the best features
def run_genetic_selection(model,weights,nb_indiv,nb_gen,min_index,max_index,rm_mutpb=0.5,add_mutpb=0.5,length_max=20,tourn_size=3,length_init=5,cxpb=0.5,mutpb=0.2,log_file=None,**kwargs):
	
	#Creating classes for the genetic algorithm
	create_deap_classes(creator,weights)
	

	#toolbox object from deap: allows to define functions in one line for operations on the individual
	toolbox = base.Toolbox()
	create_tools(toolbox,model,creator,min_index,max_index,length_init,rm_mutpb,add_mutpb,tourn_size,length_max,**kwargs)

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
		population = toolbox.select(population, k=len(population))

		population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
		for l,ind in enumerate(population):
			ind = toolbox.trimmering(ind)
		
		if log_file:
			with open(log_file,'wb') as pickle_file:
				#pickle.dump(hparams,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
				pickle.dump(pops,pickle_file,protocol=pickle.HIGHEST_PROTOCOL )
	return pops

# test function for assesing the ga
# if the Ga works it should select only the highest indexes
def model(indiv):
	return sum(indiv)


if __name__ == '__main__':
	run_genetic_selection(model,(1,),nb_indiv=100,nb_gen=40,min_index=0,max_index=12,rm_mutpb=0.3,add_mutpb=0.3,length_max=6,tourn_size=3,cxpb=0.5,mutpb=0.5,log_file=None)