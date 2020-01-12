# Visualization
One of the main tools in this folder are the visualizations that I offer on the algorithms that I provide in the Algorithm folder. These algorithms are all producing an output file (a pickle binary file) that have a specific format, the pickle file contains the parameter space and the populations
```javascript
pops=[
	[
		indiv_1_of_generation_0,
		indiv_2_of_generation_0,
		indiv3_of_generation_0,
	],
	[
		indiv_1_of_generation_1,
		indiv_2_of_generation_1,
		indiv3_of_generation_1,
	],
	]

```
where
```javascript
indiv = {"fits":fitness_tuple,
		 "phen":phenotype_of_individual,
		 "age":age_of_individual,
		 "id":id_individual,
		 "parents":couple_of_ids,
		 "mutated":id_origin_indiv
}

```
These information allow us to link an individual to the previous generation, who is linked to the previous one etc... We can therefore build the genealogic tree which is exactly what does the ```board.py``` script

## The Board

 It runs a flask local web server and serves a visualisation of the fitness (The D3.js that does the visualization is largely inspired from [this one](http://karstenahnert.com/gp/), thank you Karsten Ahnert!) all you need to do is run the following command:

``` python3 board.py --log_file=../path_pickle_file.pk``` 
This will display on a web page the evolution of the first fitness, however you might want to display the evolution of different fitnesses therefor you can add in the arguments of the http request fitness=1 to display the second fitness, you can even display the evolution of one of the parameters by putting fitness=param

