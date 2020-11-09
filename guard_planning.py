import numpy as np
import math 
import random
from copy import deepcopy

class Intern():
	def __init__(self,cost_factor=1,preferences=0.):
		"""
		Class to symbolize an intern 
		Args:
			cost_factor: float multiplicator of the intern's loss for instance a cost factor of 2
			will make this intern twice more costly thus he will have twice as less nights than the others
			preferences: list of 0 and 1, 1 if the person is available at that night, 0 otherwise
		"""
		self.cost_factor=math.sqrt(cost_factor)
		self.preferences=preferences


class InternPopulation():
	def __init__(self,individuals,nights):
		"""
		Class to represent a group of interns and their planning
		Args:
			individuals: list of Intern to attribute
			nights: list giving the numbers of interns needed at each night

		"""
		#initializing the planning
		self.planning = np.zeros((len(individuals),len(nights)))
		self.individuals=individuals
		self.nights=nights

		p = np.ones(len(individuals))
		#for each night we attribute random individuals however the probabilites are biased toward equilibrating the planning
		for i,night in enumerate(nights):
			#counting the number of nights each intern already has
			counts = np.sum(self.planning,axis=1)
			#for this night we give it in priority to the ones who have less nights and less cost
			probabilities = np.array([individual.preferences[i]/(10*(counts[j]*individual.cost_factor)+1) for j,individual in enumerate(self.individuals)])
			#multipling with general probs (here equal to one)
			p_local = p*probabilities
			#normalizing
			p_local = p_local/np.sum(p_local)
			#pick interns
			candidates = np.random.choice(range(len(individuals)),night,replace=False,p=p_local)
			for candidate in candidates:
				self.planning[candidate,i]=1


def geometric_distrib(samples,p,max):
	"""
	Function to rescale a geometric distribution between 0 and max
	and sample from it
	"""
	interval = np.array([p*(1-p)**(i) for i in range(max)])
	interval = interval/np.sum(interval)
	bounds = [np.sum(interval[:i]) for i in range(max+1)]
	values = np.random.rand(samples)
	return np.digitize(values,bounds)
def compute_cost_per_intern(cost_matrix,planning):
	#matrix product for computing loss
	return np.diag(np.dot(np.dot(planning,cost_matrix),planning.T))

def mutation(population,prob_mut):
	"""
	Basic mutation operator that resets the planning on a piece of the planning
	"""
	population = deepcopy(population)
	# taking k days
	k=geometric_distrib(1,1-prob_mut,len(population.nights))[0]
	#samplig 1 day
	seed = np.random.randint(len(population.nights))
	#taking neighboring days
	indexes = [seed - i for i in range(k)]
	for i in indexes:
		p = np.ones(len(population.individuals))
		counts = np.sum(population.planning,axis=1)
		#for this night we give it in priority to the ones who have less nights and less cost
		probabilities = np.array([individual.preferences[i]/(10*(counts[j]*individual.cost_factor)+1) for j,individual in enumerate(population.individuals)])
		#multipling with general probs (here equal to one)
		p_local = p*probabilities
		#normalizing
		p_local = p_local/np.sum(p_local)
		#pick interns
		candidates = np.random.choice(range(len(population.individuals)),population.nights[i],replace=False,p=p_local)
		# reseting the planning
		population.planning[:,i]=np.zeros(len(population.individuals))
		for candidate in candidates:
			population.planning[candidate,i]=1
		else:
			continue
	return population

def list_possible_changes(population):
	# Function to list the possible exchanges
	possible_exchanges=[]
	#looping on 2 individuals
	for i in range(len(population.individuals)):
		for j in range(len(population.individuals)):
			if j>=i:
				continue
			else:
				#looping on 2 days
				for k in range(len(population.nights)):
					for l in range(len(population.nights)):
						if l > k:
							continue
						else:
							#if we are on the same day we permute the people not having shifts and not having shift, otherwise on different day we echange presences
							if ((population.planning[i,k]!=population.planning[j,l]and k==l) or (population.planning[j,l]==1 and population.planning[i,k]==1 and k!=l)) and population.individuals[i].preferences[k]>=1. and population.individuals[j].preferences[l]>=1.:
								if (j,i,l,k) not in possible_exchanges:
									possible_exchanges.append((i,j,k,l))
	return possible_exchanges


def mutation2(population,prob_mut):
	#exchanging a certain number of points 
	population = deepcopy(population)
	
	N=geometric_distrib(1,1-prob_mut,population.planning.shape[1])[0]
	
	for n in range(N):
		#listing possibilities
		possible_exchanges = list_possible_changes(population)
		exchange = random.sample(possible_exchanges,1)[0]
		i,j,k,l = exchange
		copy_planning = deepcopy(population.planning)
		#exchange the value
		population.planning[i,k]=copy_planning[j,l]
		population.planning[j,l]=copy_planning[i,k]
	return population


def cross_over(population1,population2,x_prob):
	#Function to cross two plannings
	population1 = deepcopy(population1)
	population2 = deepcopy(population2)
	#pull out some indexes
	k=geometric_distrib(1,x_prob,len(population1.nights))[0]
	seed = np.random.randint(len(population1.nights))
	indexes = [seed - i for i in range(k)]
	# exchanging the indexes
	temp1 = np.copy(population1.planning[:,indexes])
	temp2 = np.copy(population2.planning[:,indexes])
	population1.planning[:,indexes]=temp2
	population2.planning[:,indexes]=temp1
	return population1,population2

def tournament(tuples,k,eletism):
	# k Tournament with n elitism individuals
	new_pop = []
	#k tournament 
	for _ in tuples[:-eletism]:
		warriors = random.sample(tuples,k)
		new_pop.append(deepcopy(max(warriors,key=lambda x:x[1])))
	#elitism
	bests = sorted(tuples,key=lambda x:x[1],reverse=False)[-eletism:]
	for best in bests:
		new_pop.append(best)
	#output indivs + scores
	new_pop,scores = zip(*new_pop)
	new_pop,scores = list(new_pop),list(scores)

	return new_pop,scores

def compute_loss(population,cost_matrix,affinity_grid=None):
	# computing the cost on a planning using cost matrix and affinity matrix
	cost_diag = compute_cost_per_intern(cost_matrix,population.planning)*[inter.cost_factor for inter in population.individuals ]
	loss = -np.sum(cost_diag**2)
	if affinity_grid is not None:
		affinities = compute_affinities(population.planning,affinity_grid)
		loss = loss + np.sum(affinities)/10
	return loss


def compute_affinities(planning,affinity_grid):
	# computing affinities the same way the cost is computed
	return np.diag(np.dot(np.dot(planning.T,affinity_grid),planning))


def run_genetic_algorithm(individuals,shifts,cost_matrix,number_of_ip,epochs,tourn_size,x_pb,x_indiv,mut_pb,mut_indiv,elitism,affinity_grid=None):
	"""
	Main function to run the genetic algorithm
	Arguments defining the problem
		individuals: list of Interns that will be assigned nights
		shifts: iterable of int that lists the number of persons needed at each night mathcing the size of the interns availabilities
		cost_matrix: matrix to compute the cost, the diagonal being the individual cost of each night and the side values are
	the extra cost of making nights i and j
	Parameters of the algorithm:
		number_ip: number of initial plannings
		epochs: number of iterations
		tourn_size: tournament size for the selection
		x_pb: probability to cross two plannings
		x_indiv: probability to cross a specific day
		mut_pb: probability to mutate a planning
		muy_indiv: probability to mute one day
		elitism: number of best individuals to keep from one generation to the next



	Kwargs:
	affinity_grid (optional): affinity matrix to add to the cost 

	"""
	IPs = [InternPopulation(individuals,shifts) for _ in range(nb_ip)]
	best_plan=0
	for epoch in range(epochs):
		print("Epoch",epoch)
		fitnesses = list(map(lambda x: compute_loss(x,cost_matrix,affinity_grid),IPs))
		tuples = list(zip(IPs,fitnesses))
		best_dude = max(tuples,key=lambda x:x[1])
		print("Best cost ",best_dude[1])
		best_plan = best_dude[0].planning
		IPs,fits = tournament(tuples,tourn_size,elitism)
		for i in range(0,nb_ip,2):
			if random.random()<x_indiv:
				IPs[i],IPs[i+1]=cross_over(IPs[i],IPs[i+1],x_pb)
		for i in range(nb_ip):
			if random.random()<mut_indiv:
				IPs[i]=mutation(IPs[i],mut_pb)
	return best_plan

if __name__ == '__main__':
	marc = Intern(preferences=(1,1,1,1,1,1))
	elise = Intern(preferences=(1,1,1,1,1,1),cost_factor=2)
	jean = Intern(preferences=(1,1,1,1,1,1))



	cost_matrix = np.array([[1,1,0,0,0,0],
						   [1,1,1,0,0,0],
						   [0,1,1,1,0,0],
						   [0,0,1,1,1,0],
						   [0,0,0,1,1,1],
						   [0,0,0,0,1,1]])

	affinity_grid = np.array([[0,1,0],
							  [1,0,0],
							  [0,0,0]])
	shifts=(1,2,1,1,1,2)

	nb_ip = 100
	epochs= 300
	tourn_size=3
	x_pb=0.2
	x_indiv=0.5
	mut_pb=0.2
	mut_indiv=0.3
	elitism = 2
	


	print(run_genetic_algorithm([marc,elise,jean],shifts,cost_matrix,nb_ip,epochs,tourn_size,x_pb,x_indiv,mut_pb,mut_indiv,elitism,affinity_grid=None))



