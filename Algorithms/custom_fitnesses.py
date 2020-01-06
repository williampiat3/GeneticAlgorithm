from deap import base

class FitnessReg(base.Fitness):
	#this class only changes the comparison of the fitnesses in case of a multi fitness case 
	#it is only overwritting the comparison operators 
	"""

	FitnessesReg may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
	``!=``. The comparison of those operators is made on the sum of the weighted values.
	Maximization and minimization are taken care off by a multiplication
	between the :attr:`weights` and the fitness :attr:`values`. The comparison
	can be made between fitnesses of different size, if the fitnesses are
	equal until the extra elements, the fitness with the highest sum of weighted
	values will be greater than the other

	"""

	weights = None
	"""The weights are used in the fitness comparison. They are shared among
	all fitnesses of the same type. When subclassing :class:`Fitness`, the
	weights must be defined as a tuple where each element is associated to an
	objective. A negative weight element corresponds to the minimization of
	the associated objective and positive weight to the maximization.

	.. note::
		If weights is not defined during subclassing, the following error will
		occur at instantiation of a subclass fitness object:

		``TypeError: Can't instantiate abstract <class Fitness[...]> with
		abstract attribute weights.``
	"""

	wvalues = ()
	"""Contains the weighted values of the fitness, the multiplication with the
	weights is made when the values are set via the property :attr:`values`.
	Multiplication is made on setting of the values for efficiency.

	Generally it is unnecessary to manipulate wvalues as it is an internal
	attribute of the fitness used in the comparison operators.
	"""

	def __init__(self, values=(),**kwargs):
		if len(values) > 0:
			self.values = values

	def __le__(self, other):
		return sum(self.wvalues) <= sum(other.wvalues)

	def __lt__(self, other):
		return sum(self.wvalues) < sum(other.wvalues)

	""" Instead of comparing the fitnesses lexicographically this allow to select a point
		 on the pareto front of the multi-ojective problems this allows to create a reguralized
		 objective that tries to make the most of two worlds
	"""

class ConstrainedFitness(FitnessReg):
	#this class only changes the comparison of the fitnesses in case of a constrained GA
	"""

	ConstrainedFitness may be compared using the ``>``, ``<``, ``>=``, ``<=``, ``==``,
	``!=``. The comparison of those operators is made on the number of filled conditions and then on fitnesses
	Maximization and minimization are taken care off by a multiplication
	between the :attr:`weights` and the fitness :attr:`values`. The comparison
	can be made between fitnesses of different size, if the fitnesses are
	equal until the extra elements, the fitness with the highest sum of weighted
	values will be greater than the other

	"""

	constraints = []
	'''
	Array containing the constraints for future plots
	'''

	#building transparency properties for the weights and values
	@property
	def weights(self):
		""" Fitness weights wrapped by the constrainedClass """
		return self.base_fit.weights
	@weights.setter
	def weigths(self,weights):
		self.base_fit.weights = weights
	@weights.deleter
	def weights(self):
		self.base_fit.weights=None


	#building transparency property for the values
	@property
	def values(self):
		""" Fitness values wrapped by constrained class """
		return self.base_fit.values
	@values.setter
	def values(self,values):
		self.base_fit.values=values
	@values.deleter
	def values(self):
		del self.base_fit.values

	base_fit = None

	def __init__(self, values=()):

		if self.base_fit is None:
			raise TypeError("base_fit has to be defined before using constrained fitness")
		
		super(ConstrainedFitness, self).__init__(values=values)


	def __le__(self, other):
		if sum(self.constraints) < sum(other.constraints):
			return True
		if sum(self.constraints) == sum(other.constraints):
			return self.base_fit <= other.base_fit
		return False

	def __lt__(self, other):
		if sum(self.constraints) < sum(other.constraints):
			return True
		if sum(self.constraints) == sum(other.constraints):
			return self.base_fit < other.base_fit
		return False
	""" We are now comparing first the violation factor and then the fitnesses
	"""