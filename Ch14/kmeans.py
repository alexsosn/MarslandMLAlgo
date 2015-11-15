
# Code from Chapter 14 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class kmeans:
	""" The k-Means algorithm"""
	def __init__(self,k,data):

		self.nData = np.shape(data)[0]
		self.nDim = np.shape(data)[1]
		self.k = k
		
	def kmeanstrain(self,data,maxIterations=10):
		
		# Find the minimum and maximum values for each feature
		minima = data.min(axis=0)
		maxima = data.max(axis=0)
	
		# Pick the centre locations randomly
		self.centres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
		oldCentres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
	
		count = 0
		#print centres
		while np.sum(np.sum(oldCentres-self.centres))!= 0 and count<maxIterations:
	
			oldCentres = self.centres.copy()
			count += 1
	
			# Compute distances
			distances = np.ones((1,self.nData))*np.sum((data-self.centres[0,:])**2,axis=1)
			for j in range(self.k-1):
				distances = np.append(distances,np.ones((1,self.nData))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
	
			# Identify the closest cluster
			cluster = distances.argmin(axis=0)
			cluster = np.transpose(cluster*np.ones((1,self.nData)))
	
			# Update the cluster centres	
			for j in range(self.k):
				thisCluster = np.where(cluster==j,1,0)
				if sum(thisCluster)>0:
					self.centres[j,:] = np.sum(data*thisCluster,axis=0)/np.sum(thisCluster)
			#plot(data[:,0],data[:,1],'kx')
			#plot(centres[:,0],centres[:,1],'ro')
		return self.centres
	
	def kmeansfwd(self,data):
		
		nData = np.shape(data)[0]
		# Compute distances
		distances = np.ones((1,nData))*np.sum((data-self.centres[0,:])**2,axis=1)
		for j in range(self.k-1):
			distances = np.append(distances,np.ones((1,nData))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
	
		# Identify the closest cluster
		cluster = distances.argmin(axis=0)
		cluster = np.transpose(cluster*np.ones((1,nData)))
	
		return cluster
