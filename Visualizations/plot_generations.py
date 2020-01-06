import matplotlib.pyplot as plt
import numpy as np
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Function to load a pickle file that contains the population and the hyperparameter space
def load_pop(path):
	with open(path,'rb') as pickle_file:
		hparams=pickle.load(pickle_file)
		pop=pickle.load(pickle_file)
	return hparams,pop

def clip(a):
	return np.clip(np.array(a),0,1)
#it is important to write as well the hyperparameter space so that we can plot all components

# this function is analysing a population and extracting the values
#hparams is the hyperparameters,pop is one population, contractor is the operator that takes a list of values
#and return one element. It could be the mean, the max, the min, the median, a quantile...
def repartition_caracter_in_pop(hparams,pop,contractor=np.mean):
	#we are getting here all the parameters names
	keys=list(hparams.keys())
	#initializing results that will concatenante the distributions
	results=dict(zip(keys,[{} for i in range(len(keys))]))
	#initializing means that will average the fitnesses for one gene
	means = dict(zip(keys,[{} for i in range(len(keys))]))

	#Still initializing the objects
	for key in keys:
		for gene in hparams[key]:
			results[key][str(gene)]=0
			means[key][str(gene)]=[]

	#looping in the population to extract the caracteristics
	for indiv in pop:
		for caracter in indiv["phen"]:
			#incrementing the value for the gene
			results[caracter][str(indiv["phen"][caracter])]+=1
			#adding the fitnesses to be contracted
			means[caracter][str(indiv["phen"][caracter])].append(indiv["fits"])
	# looping in the mean dictionnary to contract the values of the mean 
	for key in keys:
		for gene in hparams[key]:
			if means[key][str(gene)]!=[]:
				try:
					#unzipping 
					unzipped=list(zip(*means[key][str(gene)]))
					#clipping the fitnesses
					unzipped=list(map(clip,unzipped))
					#contracting
					means_array=list(map(contractor,unzipped))
					#overwritting the maeans dictionnary
					means[key][str(gene)]=means_array
				except TypeError:
					#in case a hyperparameter as only 1 gene
					pass
			else:
				#if the gene is not in the population
				means[key][str(gene)]=None
	return results,means


#Function to plot the generation with a dittribution in 3D
#hparams is the hyperparameter space, pops is the list of the populations, cmap is a color map object of  fitness number is which fitness you want to plot
#contractor is the operator for contracting the values for a plot
# (You need to normalize the fitnesses between 0 and 1) you can use the  lambda_ to enhance the contrast
def plot_bar_color(hparams,pops,cmap=cm.viridis,fitness_number=0,contractor=np.mean,lamdba_=lambda x: x):
	#gathering all the informations from the generation and unzipping the results	
	results,means=zip(*[repartition_caracter_in_pop(hparams,pop,contractor=contractor) for pop in pops])
	#for each hyperparameter
	for carac in hparams:
		#new figure
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#legends
		ax.set_xlabel(carac)
		ax.set_ylabel("Generation")
		ax.set_zlabel("Occurences")
		# for each generation
		for i,result in enumerate(results):
			#plotting generation i in plan y=i 
			axs=plt.bar(result[carac].keys(),result[carac].values(),zs=i,zdir='y',edgecolor='black')
			caract_fitnesses=means[i][carac]

			#Here were different tests for an inner normalization

			# fitness_values=[max(0,caract_fitnesses[gene][fitness_number]) for gene in caract_fitnesses if caract_fitnesses[gene]!= None]
			# max_gene=np.max(fitness_values)
			# min_gene=np.min(fitness_values)
			# std=np.std(fitness_values)

			#for each bar in the 2d bar chart
			for k,ax in enumerate(axs):
				fitness_values_none=[caract_fitnesses[gene] for gene in caract_fitnesses]
				#if the gene is present 
				if fitness_values_none[k]!=None:
					#set its color using the colormap and lambda_
					ax.set_facecolor(cmap(max(0,lamdba_(fitness_values_none[k][fitness_number]))))
				else:
					#if not present set the color to the lower values of the color map
					ax.set_facecolor(cmap(0))


	plt.show()

# This function gives the best fitness and the best associated individual
def best_individual(populations, fitness_index=0):
    """This function gives the best fitness and the best associated individual"""

    # Loop over population
    fitnesses = []
    individuals = []
    for pop in populations:
        # Loop over individuals
        for ind in pop:
        	# Get fitness and individual
            fitnesses.append(ind["fits"])
            individuals.append(ind["phen"])

    # Get best individual and best fitness
    best_fitness = np.max(np.array(fitnesses)[:,fitness_index])
    index_population = np.argmax(np.array(fitnesses)[:,fitness_index])
    best_individual = individuals[index_population]

    # Return results
    return best_fitness, best_individual

if __name__=='__main__':
	#pickle with hparams and pops
	path='saved_results2.pk'
	hparams,pops=load_pop(path)
	plot_bar_color(hparams,pops,cm.hot,0,contractor=np.max,lamdba_=lambda x: x)

