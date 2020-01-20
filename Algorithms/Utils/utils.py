import itertools
import numpy as np


def get_all_combinations(feed_dict):
	"""
	Depreciated util function for discrete parameter space
	"""
	output=[]
	keys=feed_dict.keys()
	iterables=list(map(lambda x: feed_dict[x],keys))
	for t in itertools.product(*iterables):
		output.append(dict(zip(keys, t)))
	return output
#function that generates a random individual using the dna translator
def generate(translator,creator):
	indiv=creator.Individual(translator.generate_random())
	indiv.parents=None
	indiv.mutated=None
	indiv.id=None
	indiv.age=0
	return indiv


#funtion that retruns the fitness from an individual
#the output is a tuple to match the format of the weights given just above in fitnessMax definition
def evalOneMax(individual,model,translator,**kwargs):
	if (translator.is_dna_viable(individual)==False):
		return -np.inf,
	else:
		phen=translator.dna_to_phen(individual)
		accuracy=model(phen,**kwargs)
		return accuracy,


#Building a decorator for cross operator so that it registers the data of the parents in the children
def decorator_cross(cross_op):
	def wrapper(indiv1, indiv2,*args,**kwargs):
		# Pré-traitement
		indiv1, indiv2 = cross_op(indiv1, indiv2,*args,**kwargs)
		indiv1.parents = (indiv1.id,indiv2.id)
		indiv2.parents = (indiv1.id,indiv2.id)
		indiv1.age=0
		indiv2.age=0
		return indiv1, indiv2
		# Post-traitement
	return wrapper

#Building a decarator for mutation operators so that it registers the mutation operator in the class individual
def decorator_mut(mut_op):
	def wrapper(indiv,*args,**kwargs):
		indiv, = mut_op(indiv,*args,**kwargs)
		indiv.mutated=indiv.id
		indiv.age=0
		return indiv,
	return wrapper

#decorator selection
def decorator_selection(selec_op):
	def wrapper(indivs,*args,**kwargs):
		indivs = selec_op(indivs,*args,**kwargs)
		for indiv in indivs:
			indiv.mutated=None
			indiv.parents=None
		return indivs
	return wrapper

#Function to recover the population with the same attributes than before it transforms the log file into individuals (deap object)
def recover_last_gen(pops,creator,translator):

	last_gen = pops[-1]
	last_pop = []
	#for each last entry in the log file we create an Individual with the right parameters
	for ind in last_gen:
		indiv = creator.Individual(translator.phen_to_dna(ind["phen"]))
		indiv.id = ind["id"]
		indiv.parents = ind["parents"]
		indiv.mutated = ind["mutated"]
		indiv.age = ind["age"]
		indiv.fitness.values = ind["fits"]
		last_pop.append(indiv)
	return last_pop

def evalOneMaxMulti(individual,model,translator,weights,**kwargs):
	if (translator.is_dna_viable(individual)==False):
		return tuple(map(lambda x: -np.inf*abs(x)/x,weights))
	
	else:
		phen=translator.dna_to_phen(individual)
		score=model(phen,**kwargs)
		return score

#Hybrid operators for the hybrid ga

# Assessment operator 
def evaluate(indiv,model,translator,weights,**kwargs):
	if (translator.is_dna_viable(indiv)==False):
		return tuple(map(lambda x: -np.inf*abs(x)/x,weights))
	else:
		hparam = translator.dna_to_phen(indiv)
		results = model(hparam,**kwargs)
		return results

#Hybrid cross over using 2 deap operators
def hybrid_cx(indiv1,indiv2,discrete_cx ,discrete_cx_kwargs,continuous_cx,continuous_cx_kwargs):
	indiv1[0],indiv2[0]=discrete_cx(indiv1[0],indiv2[0],**discrete_cx_kwargs)
	indiv1[1],indiv2[1]=continuous_cx(indiv1[1],indiv2[1],**continuous_cx_kwargs)
	return indiv1,indiv2

# Hybrid mutation using 2 deap operators
def hybrid_mut(indiv,discrete_mut,discrete_mut_kwargs,continuous_mut,continuous_mut_kwargs):
	indiv[0],=discrete_mut(indiv[0],**discrete_mut_kwargs)
	indiv[1],=continuous_mut(indiv[1],**continuous_mut_kwargs)
	return indiv,


