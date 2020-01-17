# Genetic Algorithms
<p align="center">
	<img src="./Images/main_page.png">
</p>
<p align="center">
	<img src="./Images/plot_generations.png">
</p>


In ```unit_test.py``` you can find unit tests on the three main approaches the discrete appoach, the continuous approach and the hybrid approach, they are tested against the the [Griewank function](https://deap.readthedocs.io/en/master/api/benchmarks.html#deap.benchmarks.griewank) you can change the test function but remember that the algorithm are meant for maximization.


The goal of this folder is to present different genetic algorithms that I develloped in order to tackle optimization problems in my different jobs. It includes the evolutionnary algorithms that I created but also the different visualizations that I coded/adapted in order to give some insight on how the algorithms operates.
Of course the visualization don't work for all the algorithms (that can operate on continuous or discrete spaces) and I will try to explain which one are the best fitted for each case.
This is more an experimental folder where I suggest an approach and would like to exchange about the relevance of some of my technical choices, I am open to suggestions :) .

## Introduction
These tools were implemented in order to have a more efficient way to solve black-box optimization problems. What I call a black-box is process that can be fed with a given set of inputs and that produces one or multiple outputs. The user doesn't have access to the inner workings of the black box only the output. Therefore a black-box optimization problem is a problem of finding the set of inputs that gives the global minima (or maxima) of the output without knowing how the information (ie the inputs) is processed by the black-box. This can be difficult because the black-box can have multiple local minimas, can be highly non-convex and can have zones of the input space that generate errors.

This is problematic because usually in optimization problems we can access to gradients that allow a faster (and more stable as well) convergence. However here we don't have access to them. A lot of industrial problems have to be formulated as black-box problems, some old codes that don't return their gradients, brute force searches, features selection or even hyper-optimization of statistical models can all be formulated as black box problems. I offer here a way to solve them which is not optimal but surely offers (provided the algorithms are correctly tuned) a better way than a brute force approach (I believe).

## A bit of context
The approaches for tackling black-box problem are numerous and genetic algorithm is a small part of them, [scikit learn](https://scikit-optimize.github.io/) provides a lot of optimizers already (with some of the most recent approaches) but I offer here (I hope) an easier introduction and interpretability of the inner workings of the optimization process

## Architecture
In the folder Algorithms you will find the different components of the genetic algorithm each one of them will be explained in detail. Each run of the genetic algorithm produces a pickle file that contains all the information concerning the executions made by the algorithms and the outputs. This fiel can be used to produce visualization

In the folder Vizualizations you will find some exploitations: here the script take as input the pickle file produced by the algorithm and produce the visualizations, i will try to give credit to all the  blogs that brought me ideas on the visualizations.

I built my genetic algorithms on top of [deap](https://deap.readthedocs.io/en/master/) which is a highly customizable solution for genetic algorithm and genetic programming. I learned a lot on python thanks to them. I truly advice checking out their package