
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The Kernel PCA algorithm

import numpy as np
import pylab as pl

def kernelmatrix(data,kernel,param=np.array([3,2])):
    
    if kernel=='linear':
        return np.dot(data,transpose(data))
    elif kernel=='gaussian':
        K = np.zeros((np.shape(data)[0],np.shape(data)[0]))
        for i in range(np.shape(data)[0]):
            for j in range(i+1,np.shape(data)[0]):
                K[i,j] = np.sum((data[i,:]-data[j,:])**2)
                K[j,i] = K[i,j]
        return np.exp(-K**2/(2*param[0]**2))
    elif kernel=='polynomial':
        return (np.dot(data,np.transpose(data))+param[0])**param[1]
    
def kernelpca(data,kernel,redDim):
    
    nData = np.shape(data)[0]
    nDim = np.shape(data)[1]
    
    K = kernelmatrix(data,kernel)
    
    # Compute the transformed data
    D = np.sum(K,axis=0)/nData
    E = np.sum(D)/nData
    J = np.ones((nData,1))*D
    K = K - J - np.transpose(J) + E*np.ones((nData,nData))
    
    # Perform the dimensionality reduction
    evals,evecs = np.linalg.eig(K) 
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices[:redDim]]
    evals = evals[indices[:redDim]]
    
    sqrtE = np.zeros((len(evals),len(evals)))
    for i in range(len(evals)):
        sqrtE[i,i] = np.sqrt(evals[i])
       
    #print shape(sqrtE), shape(data)
    newData = np.transpose(np.dot(sqrtE,np.transpose(evecs)))
    
    return newData

#data = array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.35,0.3],[0.4,0.4],[0.6,0.4],[0.7,0.45],[0.75,0.4],[0.8,0.35]])
#newData = kernelpca(data,'gaussian',2)
#plot(data[:,0],data[:,1],'o',newData[:,0],newData[:,0],'.')
#show()
