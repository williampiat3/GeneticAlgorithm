
from Utils.dna_translator import DNA_creator,HybridDNA
from deap import creator, base, tools, algorithms
import itertools
import multiprocessing
import time
import pickle
import json
import random
from Utils.utils import generate,decorator_cross,decorator_mut,decorator_selection,recover_last_gen,evaluate,hybrid_cx,hybrid_mut
from Utils.custom_fitnesses import ConstrainedFitness, FitnessReg

# def assess_constraints(ind,translator,constraints):
# 	"""
# 	function to assess the constraints and assign them to the individuals
# 	Arguments:
# 		ind (deap individual): individual to assess
# 		translator (Hybrid_DNA class): for translating genotype in phenotype
# 		constraints (array of callables): callables that return True if the constraint is respected, False otherwise
# 	"""
# 	if translator.is_dna_viable(ind):
# 		phen = translator.dna_to_phen(ind)
# 		const=[const(phen,ind.fitness.values) for const in constraints]
# 	else:
# 		const = [False for _ in constraints]
# 	return const

def creation_deap_classes(creator,weights):
	"""
	Function creating basic fitness and individual classes
	"""
	#weight is the weights given to the different fitnesses (incase you have a multiobjective optimization to make)
	#the Fitness that we are using is not the lexicographic one, we are creating a constrained one
	creator.create("FitnessHidden", FitnessReg,weights=weights)
	creator.create("FitnessMax", ConstrainedFitness,base_fit=creator.FitnessHidden)

	 
	#creating the individual with extra arguments (parents, mutated, id and age for logs)
	#these parameters are changed by the decorators of the operators to trace which operator was applied to whom
	creator.create("Individual", list, fitness=creator.FitnessMax,parents=None,mutated=None,id=None,age=0)

def creation_tools(toolbox,model,translator,creator,weights,discrete_cx,discrete_cx_kwargs,continuous_cx,continuous_cx_kwargs,discrete_mut,discrete_mut_kwargs,continuous_mut,continuous_mut_kwargs,selection_op,selection_kwargs,nb_threads,constraints,**kwargs):
	"""
	Function creating operators for the continous ga
	Warning the deap creator has to be instanciated with the class Individual created (call function creation deap classes first)
	Parameters:
		model (python callable) evaluation function 
		translator (Continuous_DNA instance): translator for interfacing genotype and phenotype
		creator (Deap creator): creator with classes Individual ans FitnessMax
		weights (tuple): weights of the metrics of the python callable
		discrete_cx (python callable): discrete cross over operation
		discrete_cx_kwargs (python dictionnary): kwargs for the dicrete cross over
		continuous_cx (python callable): continuous cross over
		continuous_cx_kwargs (python dictionnary): kwargs for the continuous cross over
		discrete_mut (python callable): discrete mutation operator
		discrete_mut_kwargs (python dictionnary): kwargs for discrete mutation
		continuous_mut (python callable): continuous mutation operator
		continuous_mut_kwargs (python dictionnary): kwargs for continuous mutation
		selection_op (python callable): selection operator
		selection_kwargs (python dictionnary): kwargs for the selection
		nb_threads (Integer): number of process to run in parallel
		constraints (list of callables): callables that return True if the constraint is respected, False otherwise
		**kwargs: extra arguments for the model
	"""
	# generation operation: relies on the function generate_random of the translator
	toolbox.register("individual", generate,translator=translator,creator=creator)
	# a population is a repetition of individuals
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	#funtion that retruns the fitness from an individual
	toolbox.register("evaluate", evaluate,model=model,translator=translator,weights=weights,**kwargs)
	#mate function decorated for logs, mixing continuous and discrete operators
	toolbox.register("mate",decorator_cross( hybrid_cx),discrete_cx=discrete_cx ,discrete_cx_kwargs=discrete_cx_kwargs,continuous_cx=continuous_cx,continuous_cx_kwargs=continuous_cx_kwargs)
	#mutation function decorated for logging on which individual the mutation was done
	toolbox.register("mutate", decorator_mut(hybrid_mut), discrete_mut=discrete_mut ,discrete_mut_kwargs=discrete_mut_kwargs,continuous_mut=continuous_mut,continuous_mut_kwargs=continuous_mut_kwargs)
	#selection operator decorated for changing logs
	toolbox.register("select",decorator_selection(selection_op), **selection_kwargs)
		

	#Multiprocessing if the user specified multiple threads
	if nb_threads > 1:
		pool = multiprocessing.Pool(processes=nb_threads)
		toolbox.register("map",pool.map)


#Compared to the previous hybrid GA this one computes the violation factor
#a constraint is a function that takes as the input the parameters and the fitnesses
def run_constrained_hybrid_ga(model,weights,hparams,NGEN=40,nb_indiv=80,nb_threads=1,cxpb=0.7, mutpb=0.4,discrete_cx=tools.cxTwoPoint ,discrete_cx_kwargs={},continuous_cx=tools.cxBlend,continuous_cx_kwargs={"alpha":0.1},discrete_mut=tools.mutFlipBit,discrete_mut_kwargs={"indpb":0.05},continuous_mut=tools.mutGaussian,continuous_mut_kwargs={"mu":0.,"sigma":0.1, "indpb":0.05},selection_op=tools.selTournament,selection_kwargs={"tournsize":3},stric_interval=False,log_file=None,recover_file=None,gray_code=True,constraints=[],**kwargs):
	if recover_file:
		with open(recover_file,'rb') as pickle_file:
			hparams=pickle.load(pickle_file)
			translator=HybridDNA(hparams,stric_interval=stric_interval,gray_code=gray_code)
			pops=pickle.load(pickle_file)

			#creation of deap classes
			creation_deap_classes(creator,weights)
			#toolbox object from deap: allows to define functions in one line for operations on the individual
			toolbox = base.Toolbox()
			#creating operators
			creation_tools(toolbox,model,translator,creator,weights,discrete_cx,discrete_cx_kwargs,continuous_cx,continuous_cx_kwargs,discrete_mut,discrete_mut_kwargs,continuous_mut,continuous_mut_kwargs,selection_op,selection_kwargs,nb_threads,constraints,**kwargs)
			
			population=recover_last_gen(pops,creator,translator)

			#Performing selection and cross and mutation to create the new generation

			population = toolbox.select([indiv for indiv in population if translator.is_dna_viable(indiv)], k=nb_indiv)
			population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
	else:
		translator=HybridDNA(hparams,stric_interval=stric_interval,gray_code=gray_code)
		#creation of deap classes
		creation_deap_classes(creator,weights)
		#toolbox object from deap: allows to define functions in one line for operations on the individual
		toolbox = base.Toolbox()
		#creating operators
		creation_tools(toolbox,model,translator,creator,weights,discrete_cx,discrete_cx_kwargs,continuous_cx,continuous_cx_kwargs,discrete_mut,discrete_mut_kwargs,continuous_mut,continuous_mut_kwargs,selection_op,selection_kwargs,nb_threads,constraints,**kwargs)
		
		#we are now creating the population 
		population = toolbox.population(n=nb_indiv)
		
		pops=[]
	
	init_integer=len(pops)
	# running the algorithm
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

		for ind in population:
			if translator.is_dna_viable(ind):
				phen = translator.dna_to_phen(ind)
				ind.fitness.constraints=[const(phen,ind.fitness.values) for const in constraints]
			else:
				ind.fitness.constraints = [False for _ in constraints]
		

		# Printing the best individual
		top1 = tools.selBest(population, k=1)
		print(translator.dna_to_phen(top1[0]))
		print(top1[0].fitness.values)


		#Registering the information in the pickle file
		pops.append([{"phen":translator.dna_to_phen(indiv),"fits":indiv.fitness.values,"age":indiv.age,"id":indiv.id,"parents":indiv.parents,"mutated":indiv.mutated,"constraints":indiv.fitness.constraints} for indiv in population if translator.is_dna_viable(indiv)])



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
	pops = run_constrained_hybrid_ga(test_func,(1.,),hparams,NGEN=100,constraints=[lambda hparam,y: hparam["param4"]<80,lambda hparam,y: hparam["param3"]<7],log_file="results_constraints.pk")
	print(pops[0])