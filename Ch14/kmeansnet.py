
# Code from Chapter 14 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class kmeans:
    """The k-Means Algorithm implemented as a neural network"""
    def __init__(self,k,data,nEpochs=1000,eta=0.25):

        self.nData = np.shape(data)[0]
        self.nDim = np.shape(data)[1]
        self.k = k
        self.nEpochs = nEpochs
        self.weights = np.random.rand(self.nDim,self.k)
        self.eta = eta
        
    def kmeanstrain(self,data):
        # Preprocess data (won't work if (0,0,...0) is in data)
        normalisers = np.sqrt(np.sum(data**2,axis=1))*np.ones((1,np.shape(data)[0]))
        data = np.transpose(np.transpose(data)/normalisers)

        for i in range(self.nEpochs):
            for j in range(self.nData):
                activation = np.sum(self.weights*np.transpose(data[j:j+1,:]),axis=0)
                winner = np.argmax(activation)
                self.weights[:,winner] += self.eta * data[j,:] - self.weights[:,winner]            
            
    def kmeansfwd(self,data):
        best = np.zeros(np.shape(data)[0])
        for i in range(np.shape(data)[0]):
            activation = np.sum(self.weights*np.transpose(data[i:i+1,:]),axis=0)
            best[i] = np.argmax(activation)
        return best
    
