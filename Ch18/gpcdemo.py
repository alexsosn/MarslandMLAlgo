
# Code from Chapter 18 of Machine Learning: An Algorithmic Perspective (2nd Edition)
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

target = -np.ones((np.shape(iris)[0],3));
indices = np.where(iris[:,4]==0) 
target[indices,0] = 1
indices = np.where(iris[:,4]==1)
target[indices,1] = 1
indices = np.where(iris[:,4]==2)
target[indices,2] = 1

# Randomly order the data
order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

train = iris[::2,0:4]
traint = target[::2]
test = iris[1::2,0:4]
testt = target[1::2]

#print train.max(axis=0), train.min(axis=0)

# Train the machines
output = np.zeros((np.shape(test)[0],3))

theta =np.zeros((3,1))
theta[0] = 1.0 #np.random.rand()*3
theta[1] = 0.7 #np.random.rand()*3
theta[2] = 0.

import gpc
import scipy.optimize as so

args = (train[:,:2],traint[:,0])
newTheta0 = so.fmin_cg(gpc.logPosterior, theta, fprime=gpc.gradLogPosterior, args=[args], gtol=1e-4,maxiter=20,disp=1)
pred = np.squeeze(np.array([gpc.predict(np.reshape(i,(1,2)),train,traint,newTheta0) for i in test[:,:2]]))
output[:,0] = np.reshape(np.where(pred[:,0]<0,-1,1),(np.shape(pred)[1],1))
print np.sum(np.abs(output-testt))

args = (train[:,:2],traint[:,1])
newTheta1 = so.fmin_cg(gpc.logPosterior, theta, fprime=gpc.gradLogPosterior, args=[args], gtol=1e-4,maxiter=20,disp=1)
pred = np.squeeze(np.array([gpc.predict(np.reshape(i,(1,2)),train,traint,newTheta0) for i in test[:,:2]]))
output[:,1] = np.reshape(np.where(pred[:,0]<0,-1,1),(np.shape(pred)[1],1))
print np.sum(np.abs(output-testt))

args = (train[:,:2],traint[:,2])
newTheta2 = so.fmin_cg(gpc.logPosterior, theta, fprime=gpc.gradLogPosterior, args=[args], gtol=1e-4,maxiter=20,disp=1)
pred = np.squeeze(np.array([gpc.predict(np.reshape(i,(1,2)),train,traint,newTheta0) for i in test[:,:2]]))
output[:,2] = np.reshape(np.where(pred[:,0]<0,-1,1),(np.shape(pred)[1],1))
print np.sum(np.abs(output-testt))

#err1 = np.where((output==1.) & (test==-1.))[0]
#err2 = np.where((output==-1.) & (test==1.))[0]
#print "Class 1 errors ",len(err1)," from ",len(test[test==1])
#print "Class 2 errors ",len(err2)," from ",len(test[test==-1])
#print "Test accuracy ",1. -(float(len(err1)+len(err2)))/ (len(test[test==1]) + len(test[test==-1]))


#svm1 = svm.svm(kernel='linear',sigma=3.)
svm1 = svm.svm(kernel='poly',degree=1.)
#svm1 = svm.svm(kernel='rbf',gamma=0.7,sigma=2.)
svm1.train_svm(train[:,:2],traint[:,1])
output[:,1] = svm1.classifier(test[:,:2],soft=True).T

#svm2 = svm.svm(kernel='linear',sigma=3.)
svm2 = svm.svm(kernel='poly',degree=1.)
#svm2 = svm.svm(kernel='rbf',gamma=0.7,sigma=2.)
svm2.train_svm(train[:,:2],traint[:,2])
output[:,2] = svm2.classifier(test[:,:2],soft=True).T

# Make a decision about which class
# Pick the one with the largest margin
bestclass = np.argmax(output,axis=1)
err = np.where(bestclass!=iris[1::2,4])[0]
print len(err), np.shape(iris)[0]/2.

# Make a plot
pl.figure()
step=0.01
f0,f1  = np.meshgrid(np.arange(np.min(train[:,0])-0.5, np.max(train[:,0])+0.5, step), np.arange(np.min(train[:,1])-0.5, np.max(train[:,1])+0.5, step))

out = np.zeros((np.shape(f0.ravel())[0],3))
out[:,0] = svm0.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T
out[:,1] = svm1.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T
out[:,2]= svm2.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T
out = np.argmax(out[:,:3],axis=1)

# Put the result into a color plot
out = out.reshape(f0.shape)
pl.contourf(f0, f1, out, cmap=pl.cm.Paired)
#pl.axis('off')

# Plot also the training points
#traint = np.where(traint==-1,0,1)
pl.plot(train[svm0.sv,0],train[svm0.sv,1],'o',markerfacecolor=None,markeredgecolor='r',markeredgewidth=3)
pl.scatter(train[:, 0], train[:, 1], c=iris[::2,4], cmap=pl.cm.Paired)
#pl.plot(train[:, 0], train[:, 1],'o', c=traint, cmap=pl.cm.Paired)
