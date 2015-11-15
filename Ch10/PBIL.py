
# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The Population Based Incremental Learning algorithm
# Comment and uncomment fitness functions as appropriate (as an import and the fitnessFunction variable)

import pylab as pl
import numpy as np

#import fourpeaks as fF
import knapsack as fF

def PBIL():
	pl.ion()
	
	populationSize = 100
	stringLength = 20	
	eta = 0.005
	
	#fitnessFunction = 'fF.fourpeaks'
	fitnessFunction = 'fF.knapsack'
	p = 0.5*np.ones(stringLength)
	best = np.zeros(501,dtype=float)

	for count in range(501):
		# Generate samples
		population = np.random.rand(populationSize,stringLength)
		for i in range(stringLength):
			population[:,i] = np.where(population[:,i]<p[i],1,0)

		# Evaluate fitness
		fitness = eval(fitnessFunction)(population)

		# Pick best
		best[count] = np.max(fitness)
		bestplace = np.argmax(fitness)
		fitness[bestplace] = 0
		secondplace = np.argmax(fitness)

		# Update vector
		p  = p*(1-eta) + eta*((population[bestplace,:]+population[secondplace,:])/2)

		if (np.mod(count,100)==0):
			print count, best[count]

	pl.plot(best,'kx-')
	pl.xlabel('Epochs')
	pl.ylabel('Fitness')
	pl.show()
	#print p

PBIL()
