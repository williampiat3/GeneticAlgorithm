import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def dec_to_bin(integer):
	#The 2 is not a magic number binary strings have this format: '0b010110' we just need to remove '0b'
	return bin(integer)[2:]



def gray_code(n):
	def gray_code_recurse (g,n):
		k=len(g)
		if n<=0:
			return

		else:
			for i in range (k-1,-1,-1):
				char='1'+g[i]
				g.append(char)
			for i in range (k-1,-1,-1):
				g[i]='0'+g[i]

			gray_code_recurse (g,n-1)

	g=['0','1']
	gray_code_recurse(g,n-1)
	return g


class DNA_creator():
	#we are creating here the structure of the genotype
	#Gray code argument encodes in gray the genotype: better for neighbor mutation
	def __init__(self,hparams,gray_code=False):
		self.keys=list(hparams.keys())
		self.lengths={}
		self.hparams=hparams
		for key in self.keys:
			self.lengths[key]=len(hparams[key])
		self.gray_code=gray_code
	#function that creates a random (viable) individual 
	def generate_random(self):
		dna=""
		for key in self.keys:
			number=random.randint(0,self.lengths[key]-1)
			if self.gray_code:
				binary=gray_code((self.lengths[key]-1).bit_length())[number]
			else:
				binary=str(dec_to_bin(number))
				binary='0'*((self.lengths[key]-1).bit_length()-len(binary))+binary
			dna+=binary
		diction=[]
		for k in dna:
			diction.append(int(k))
		return diction
	#function that translates the genotype to phenotype
	def dna_to_phen(self,dna):
		phen={}
		cursor=0
		out=""
		for k in dna:
			out+=str(k)
		for key in self.keys:
			binary=out[cursor:cursor+(self.lengths[key]-1).bit_length()]
			if self.gray_code:
				index_gene=gray_code(len(binary)).index(binary)
			else:
				index_gene=int(binary,2)

			phen[key]=self.hparams[key][index_gene]
			cursor+=(self.lengths[key]-1).bit_length()
		return phen
	#function that translates the phenotype to genotype
	def phen_to_dna(self,phen):
		dna=""
		for key in self.keys:
			number=self.hparams[key].index(phen[key])
			if self.gray_code:
				binary=gray_code((self.lengths[key]-1).bit_length())[number]
			else:
				binary=str(dec_to_bin(number))
				binary='0'*(self.lengths[key].bit_length()-len(binary)-1)+binary
			dna+=binary
		diction=[]
		for k in dna:
			diction.append(int(k))
		return diction
		
	#function that checks if the individual is viable 
	def is_dna_viable(self,dna):
		try:
			self.dna_to_phen(dna)
			return True 
		except (IndexError,TypeError):
			return False

	def repartition_pop(self,pop):
		results=dict(zip(self.keys,[{} for i in range(len(self.keys))]))
		for key in self.keys:
			for gene in self.hparams[key]:
				results[key][str(gene)]=0
		for indiv in pop:
			for caracter in indiv:
				results[caracter][str(indiv[caracter])]+=1
		return results

# Continous DNA class that handles the continous part of the dna
class Continous_DNA():
	def __init__(self,hparams,stric_interval=False):
		self.length=len(hparams)
		self.hparams=hparams
		self.keys=list(self.hparams.keys())
		self.stric_interval=stric_interval

	# Funtion that interpolate a param value of the hypercube to the real range 
	def interpolate_param(self,param,value):
		if self.hparams[param]["scale"] == "linear":

			return value*(self.hparams[param]["range"][1]-self.hparams[param]["range"][0])+ self.hparams[param]["range"][0]
		if self.hparams[param]["scale"] == "log":
			return (math.exp(value*math.log1p(self.hparams[param]["range"][1]-self.hparams[param]["range"][0]))+self.hparams[param]["range"][0]-1)


	def interpolate_reverse_param(self,param,value):
		if self.hparams[param]["scale"] == "linear":
			return (value-self.hparams[param]["range"][0])/(self.hparams[param]["range"][1]-self.hparams[param]["range"][0])

		if self.hparams[param]["scale"] == "log":
			return (math.log1p(value-self.hparams[param]["range"][0])/math.log1p(self.hparams[param]["range"][1]-self.hparams[param]["range"][0]))

	# Function that creates a random individual in the hypercube
	def generate_random(self):
		return [random.random() for _ in self.hparams ]

	# Translation operator P2D
	def dna_to_phen(self,dna):
		interpolated_value = [self.interpolate_param(param,gene) for param,gene in zip(self.keys,dna)]
		return dict(zip(self.keys,interpolated_value))

	# Translation operator D2P
	def phen_to_dna(self,phen):
		return [self.interpolate_reverse_param(param,phen[param]) for param in self.keys]

	def is_dna_viable(self,dna):
		if not self.stric_interval:
			return True
		valid=True
		for k in range(self.length):
			valid = (dna[k]<1. and dna[k]>0.)
			if not valid:
				break
		return valid



#Hybrid DNA class that takes the two previously created class for dna handling
#the genotype is a tuple of a discrete and continuous genotype
class HybridDNA():
	def __init__(self,hparams,stric_interval=False,gray_code=True):
		self.discrete_translator=DNA_creator(hparams["discrete"],gray_code=gray_code)
		self.continuous_translator=Continous_DNA(hparams["continuous"],stric_interval=stric_interval)
		self.hparams=hparams

	def generate_random(self):
		return [self.discrete_translator.generate_random(),self.continuous_translator.generate_random()]

	# Translation operator P2D
	def dna_to_phen(self,dna):
		disc = self.discrete_translator.dna_to_phen(dna[0])
		cont = self.continuous_translator.dna_to_phen(dna[1])
		return {**disc,**cont}

	# Translation operator D2P
	def phen_to_dna(self,phen):
		phen_disc = dict([(key,phen[key]) for key in self.hparams["discrete"] ])
		phen_cont = dict([(key,phen[key]) for key in self.hparams["continuous"] ])
		disc = self.discrete_translator.phen_to_dna(phen_disc)
		cont = self.continuous_translator.phen_to_dna(phen_cont)
		return [disc,cont]

	def is_dna_viable(self,dna):
		return (self.discrete_translator.is_dna_viable(dna[0]) and self.continuous_translator.is_dna_viable(dna[1]))



if __name__=='__main__':
	hparams={
				"character1":[1,2,3],
				"character2":["relu","sig"]
	}

	translator=DNA_creator(hparams,gray_code=True)

	indiv = translator.generate_random()
	print(indiv)
	phen = translator.dna_to_phen(indiv)
	print(phen)
	print(translator.is_dna_viable([1,0,1]))
	# pops=[[translator.generate_random() for i in range(10)] for j in range(3)]
	# results=[translator.repartition_pop(pop) for pop in pops]
	# for carac in hparams:
	#	 fig = plt.figure()
	#	 ax = fig.add_subplot(111, projection='3d')
	#	 for i,result in enumerate(results):
	#		 plt.bar(result[carac].keys(),result[carac].values(),zs=i)
 
	# plt.show()

