
# Code from Chapter 7 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# A k-Nearest Neighbour Classifier
import numpy as np

def knn(k,data,dataClass,inputs):

	nInputs = np.shape(inputs)[0]
	closest = np.zeros(nInputs)

	for n in range(nInputs):
		# Compute distances
		distances = np.sum((data-inputs[n,:])**2,axis=1)

		# Identify the nearest neighbours
		indices = np.argsort(distances,axis=0)

		classes = np.unique(dataClass[indices[:k]])
		if len(classes)==1:
			closest[n] = np.unique(classes)
		else:
			counts = np.zeros(max(classes)+1)
			for i in range(k):
				counts[dataClass[indices[i]]] += 1
			closest[n] = np.max(counts)
			 
	return closest
