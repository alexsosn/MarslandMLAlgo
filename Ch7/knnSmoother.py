
# Code from Chapter 7 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import pylab as pl
import numpy as np

# A k-Nearest Neighbour smoother, with three different kernels
# Example is the Ruapehu dataset
def knnSmoother(k,data,testpoints,kernel):

    outputs = np.zeros(len(testpoints))
    
    for i in range(len(testpoints)):
        distances = (data[:,0]-testpoints[i])
        if kernel=='NN':
            indices = np.argsort(distances**2,axis=0)
            outputs[i] = 1./k * np.sum(data[indices[:k],1])
        elif kernel=='Epan':
            Klambda = 0.75*(1 - distances**2/k**2)
            where = (np.abs(distances)<k)
            outputs[i] = np.sum(Klambda*where*data[:,1])/np.sum(Klambda*where)
        elif kernel=='Tricube':
            Klambda = (1 - np.abs((distances/k)**3)**3)
            where = (np.abs(distances)<k)
            outputs[i] = np.sum(Klambda*where*data[:,1])/np.sum(Klambda*where)
        else:
            print('Unknown kernel')
    return outputs

data = np.loadtxt('ruapehu.dat') 
# Data is time of start and stop
# Turn into repose and duration 
t1 = data[:,0:1] 
t2 = data[:,1:2] 
repose = t1[1:len(t1),:] -t2[0:len(t2)-1,:] 
duration = t2[1:len(t2),:] -t1[1:len(t1),:]
order = np.argsort(repose,axis=0)
repose = repose[order]
duration = duration[order]
data = np.squeeze(np.concatenate((repose,duration),axis=1))
testpoints = 12.0*np.arange(1000)/1000
outputs5 = knnSmoother(5,data,testpoints,'NN')
outputs10 = knnSmoother(10,data,testpoints,'NN')

pl.plot(data[:,0],data[:,1],'ko',testpoints,outputs5,'k-',linewidth=3)
pl.plot(testpoints,outputs10,'k--',linewidth=3)
pl.legend(('Data','NN, k=5','NN, k=10'))
pl.xlabel('Repose (years)')
pl.ylabel('Duration (years)')

pl.figure(2)
outputs5 = knnSmoother(2,data,testpoints,'Epan')
outputs10 = knnSmoother(4,data,testpoints,'Epan')

pl.plot(data[:,0],data[:,1],'ko',testpoints,outputs5,'k-',linewidth=3)
pl.plot(testpoints,outputs10,'k--',linewidth=3)
pl.legend(('Data','Epanechnikov, lambda=2','Epanechnikov, lambda=4'))
pl.xlabel('Repose (years)')
pl.ylabel('Duration (years)')

pl.figure(3)
outputs5 = knnSmoother(2,data,testpoints,'Tricube')
outputs10 = knnSmoother(4,data,testpoints,'Tricube')

pl.plot(data[:,0],data[:,1],'ko',testpoints,outputs5,'k-',linewidth=3)
pl.plot(testpoints,outputs10,'k--',linewidth=3)
pl.legend(('Data','Tricube, lambda=2','Tricube, lambda=4'))
pl.xlabel('Repose (years)')
pl.ylabel('Duration (years)')


pl.show()
