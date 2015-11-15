
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The LDA algorithm

import pylab as pl
import numpy as np
from scipy import linalg as la

def lda(data,labels,redDim):

    # Centre data
    data -= data.mean(axis=0)
    nData = np.shape(data)[0]
    nDim = np.shape(data)[1]
    
    Sw = np.zeros((nDim,nDim))
    Sb = np.zeros((nDim,nDim))
    
    C = np.cov(np.transpose(data))
    
    # Loop over classes
    classes = np.unique(labels)
    for i in range(len(classes)):
        # Find relevant datapoints
        indices = np.squeeze(np.where(labels==classes[i]))
        d = np.squeeze(data[indices,:])
        classcov = np.cov(np.transpose(d))
        Sw += np.float(np.shape(indices)[0])/nData * classcov
        
    Sb = C - Sw
    # Now solve for W and compute mapped data
    # Compute eigenvalues, eigenvectors and sort into order
    evals,evecs = la.eig(Sw,Sb)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    w = evecs[:,:redDim]
    newData = np.dot(data,w)
    return newData,w

#data = np.array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.35,0.3],[0.4,0.4],[0.6,0.4],[0.7,0.45],[0.75,0.4],[0.8,0.35]])
#labels = np.array([0,0,0,0,0,1,1,1,1])
#newData,w = lda(data,labels,2)
#print w
#pl.plot(data[:,0],data[:,1],'o',newData[:,0],newData[:,0],'.')
#pl.show()
