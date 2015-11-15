
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Various dimensionality reductions running on the Iris dataset
import pylab as pl
import numpy as np

iris = np.loadtxt('../3 MLP/iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),iris.min(axis=0)*np.ones((1,5))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]
labels = iris[:,4:]
iris = iris[:,:4]

order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
labels = labels[order,0]

w0 = np.where(labels==0)
w1 = np.where(labels==1)
w2 = np.where(labels==2)

import lda
newData,w = lda.lda(iris,labels,2)
print np.shape(newData)
pl.plot(iris[w0,0],iris[w0,1],'ok')
pl.plot(iris[w1,0],iris[w1,1],'^k')
pl.plot(iris[w2,0],iris[w2,1],'vk')
pl.axis([-1.5,1.8,-1.5,1.8])
pl.axis('off')
pl.figure(2)
pl.plot(newData[w0,0],newData[w0,1],'ok')
pl.plot(newData[w1,0],newData[w1,1],'^k')
pl.plot(newData[w2,0],newData[w2,1],'vk')
pl.axis([-1.5,1.8,-1.5,1.8])
pl.axis('off')

import pca
x,y,evals,evecs = pca.pca(iris,2)
pl.figure(3)
pl.plot(y[w0,0],y[w0,1],'ok')
pl.plot(y[w1,0],y[w1,1],'^k')
pl.plot(y[w2,0],y[w2,1],'vk')
pl.axis('off')

import kernelpca
newData = kernelpca.kernelpca(iris,'gaussian',2)
pl.figure(4)
pl.plot(newData[w0,0],newData[w0,1],'ok')
pl.plot(newData[w1,0],newData[w1,1],'^k')
pl.plot(newData[w2,0],newData[w2,1],'vk')
pl.axis('off')

import factoranalysis
newData = factoranalysis.factoranalysis(iris,2)
#print newData
pl.figure(5)
pl.plot(newData[w0,0],newData[w0,1],'ok')
pl.plot(newData[w1,0],newData[w1,1],'^k')
pl.plot(newData[w2,0],newData[w2,1],'vk')
pl.axis('off')

import lle
print np.shape(iris)
a,b,newData = lle.lle(iris,2,12)
print np.shape(newData)
print newData[w0,:]
print "---"
print newData[w1,:]
print "---"
print newData[w2,:]

pl.plot(newData[w0,0],newData[w0,1],'ok')
pl.plot(newData[w1,0],newData[w1,1],'^k')
pl.plot(newData[w2,0],newData[w2,1],'vk')
pl.axis('off')

import isomap
print labels
newData,newLabels = isomap.isomap(iris,2,100)
print np.shape(newData)
print newLabels
w0 = np.where(newLabels==0)
w1 = np.where(newLabels==1)
w2 = np.where(newLabels==2)
pl.plot(newData[w0,0],newData[w0,1],'ok')
pl.plot(newData[w1,0],newData[w1,1],'^k')
pl.plot(newData[w2,0],newData[w2,1],'vk')
pl.axis('off')

print "Done"

pl.show()
