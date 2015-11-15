
# Code from Chapter 8 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import pylab as pl

iris = np.loadtxt('iris_proc.data',delimiter=',')
#iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),iris.min(axis=0)*np.ones((1,5))),axis=0).max(axis=0)
#iris[:,:4] = iris[:,:4]/imax[:4]

target = -np.ones((np.shape(iris)[0],3),dtype=float);
indices = np.where(iris[:,4]==0) 
target[indices,0] = 1.
indices = np.where(iris[:,4]==1)
target[indices,1] = 1.
indices = np.where(iris[:,4]==2)
target[indices,2] = 1.

# Randomly order the data
#order = range(np.shape(iris)[0])
#np.random.shuffle(order)
#iris = iris[order,:]
#target = target[order,:]

train = iris[::2,0:4]
traint = target[::2]
test = iris[1::2,0:4]
testt = target[1::2]

#print train.max(axis=0), train.min(axis=0)

# Train the machines
output = np.zeros((np.shape(test)[0],3))

import svm
reload(svm)

# Learn the full data
#svm0 = svm.svm(kernel='linear')
#svm0 = svm.svm(kernel='poly',C=0.1,degree=3)
svm0 = svm.svm(kernel='rbf')
svm0.train_svm(train,np.reshape(traint[:,0],(np.shape(train[:,:2])[0],1)))
output[:,0] = svm0.classifier(test,soft=True).T

#svm1 = svm.svm(kernel='linear')
#svm1 = svm.svm(kernel='poly',C=0.1,degree=3)
svm1 = svm.svm(kernel='rbf')
svm1.train_svm(train,np.reshape(traint[:,1],(np.shape(train[:,:2])[0],1)))
output[:,1] = svm1.classifier(test,soft=True).T

#svm2 = svm.svm(kernel='linear')
#svm2 = svm.svm(kernel='poly',C=0.1,degree=3)
svm2 = svm.svm(kernel='rbf')
svm2.train_svm(train,np.reshape(traint[:,2],(np.shape(train[:,:2])[0],1)))
output[:,2] = svm2.classifier(test,soft=True).T

# Make a decision about which class
# Pick the one with the largest margin
bestclass = np.argmax(output,axis=1)
print bestclass
print iris[1::2,4]
err = np.where(bestclass!=iris[1::2,4])[0]
print err
print float(np.shape(testt)[0] - len(err))/ (np.shape(testt)[0]) , "test accuracy"


# Plot 2D version is below
#svm0 = svm.svm(kernel='linear')
svm0 = svm.svm(kernel='poly',degree=3)
#svm0 = svm.svm(kernel='rbf')
svm0.train_svm(train[:,:2],np.reshape(traint[:,0],(np.shape(train[:,:2])[0],1)))
output[:,0] = svm0.classifier(test[:,:2],soft=True).T

#svm1 = svm.svm(kernel='linear')
svm1 = svm.svm(kernel='poly',degree=3)
#svm1 = svm.svm(kernel='rbf')
svm1.train_svm(train[:,:2],np.reshape(traint[:,1],(np.shape(train[:,:2])[0],1)))
output[:,1] = svm1.classifier(test[:,:2],soft=True).T

#svm2 = svm.svm(kernel='linear')
svm2 = svm.svm(kernel='poly',degree=3)
#svm2 = svm.svm(kernel='rbf')
svm2.train_svm(train[:,:2],np.reshape(traint[:,2],(np.shape(train[:,:2])[0],1)))
output[:,2] = svm2.classifier(test[:,:2],soft=True).T


# Make a decision about which class
# Pick the one with the largest margin
bestclass = np.argmax(output,axis=1)
print bestclass
print iris[1::2,4]
err = np.where(bestclass!=iris[1::2,4])[0]
print err
print float(len(err))/ (np.shape(testt)[0]) , "test accuracy"

print z
# Make a plot
pl.figure()
step=0.01
f0,f1  = np.meshgrid(np.arange(np.min(train[:,0])-0.5, np.max(train[:,0])+0.5, step), np.arange(np.min(train[:,1])-0.5, np.max(train[:,1])+0.5, step))

out = np.zeros((np.shape(f0.ravel())[0],3))
out[:,0] = svm0.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T
out[:,1] = svm1.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T
out[:,2]= svm2.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T
out = np.argmax(out[:,:3],axis=1)
print out

out = out.reshape(f0.shape)
pl.contourf(f0, f1, out, cmap=pl.cm.Paired)
#pl.axis('off')

# Plot also the training points
#traint = np.where(traint==-1,0,1)
pl.plot(train[svm0.sv,0],train[svm0.sv,1],'o',markerfacecolor=None,markeredgecolor='r',markeredgewidth=3)
pl.scatter(train[:, 0], train[:, 1], c=iris[::2,4], cmap=pl.cm.Paired)
#pl.plot(train[:, 0], train[:, 1],'o', c=traint, cmap=pl.cm.Paired)


