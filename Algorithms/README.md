# Algorithms
Here you may find canonical approches to solve optimization problems with genetic algorithms (GAs) I will present here the different approaches. If you are completly new to GAs I suggest a bit of reading [here](https://en.wikipedia.org/wiki/Genetic_algorithm#Optimization_problems) that will help you understand the concept of GAs.

All my algorithm can output a pickle file that contains all the runs executed by the GA that can be used for creating visualizations


## Utils

General purpose utility functions and utility decorators that are common to all algorithms can be found in utils.py

For coding the genetical algorithm I needed some classes for making the interaction between the genotype and phenotype, these classes can be found in dna_translator.py

For  evaluating the models I needed to code specific Fitness objects that are available in the custom_fitnesses.py file

## Algorithms

### Simple Discrete GA

### Multicriteria GA

### Continous GA
This algorithm does optimization of a continuous space, therefore the corss over and the mutation cannot be the same than on discrete spaces: the mutation is gaussian (it perturbs randomly with some noise) and the cross operation operates on each gene: the value of the child is taken uniformly in the interval of the values of the parent.

This algorithm can be used to find an optimal set of continuous values, it allows to explore the design space in a linear scale of a logarithmic scale.

An example is provided in the ```__main__``` function on a simple quadratic function

### Hybrid GA

## Experiments

### Hybrib with variable mutation

### Genetic Feature Selection




