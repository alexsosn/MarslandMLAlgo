
# Code from Chapter 8 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import pylab as pl

# Make some 2D data
linsep = False
overlap = True

if linsep:
	# Case 1: Linearly separable
	if overlap:
		cov = [[2.0,1.0],[1.0,2.0]]
	else:
		cov = [[0.8,0.6],[0.6,0.8]]

	train0 = np.random.multivariate_normal([0.,2.], cov, 100)
	train1 = np.random.multivariate_normal([2.,0.], cov, 100)
	train = np.concatenate((train0,train1),axis=0)
	test0 = np.random.multivariate_normal([0.,2.], cov, 20)
	test1 = np.random.multivariate_normal([2.,0.], cov, 20)
	test = np.concatenate((test0,test1),axis=0)

	labeltrain0 = np.ones((np.shape(train0)[0],1))
	labeltrain1 = -np.ones((np.shape(train1)[0],1))
	labeltrain = np.concatenate((labeltrain0,labeltrain1),axis=0)
	labeltest0 = np.ones((np.shape(test0)[0],1))
	labeltest1 = -np.ones((np.shape(test1)[0],1))
	labeltest = np.concatenate((labeltest0,labeltest1),axis=0)

else:
	# Case 2: Not linearly separable
	cov = [[1.5,1.0],[1.0,1.5]]
	train0a = np.random.multivariate_normal([-1.,2.], cov, 50)
	train0b = np.random.multivariate_normal([1.,-1.], cov, 50)
	train0 = np.concatenate((train0a,train0b),axis=0)
	train1a = np.random.multivariate_normal([4.,-4.], cov, 50)
	train1b = np.random.multivariate_normal([-4.,4.], cov, 50)
	train1 = np.concatenate((train1a,train1b),axis=0)
	train = np.concatenate((train0,train1),axis=0)

	test0a = np.random.multivariate_normal([-1.,2.], cov, 50)
	test0b = np.random.multivariate_normal([1.,-1.], cov, 50)
	test0 = np.concatenate((test0a,test0b),axis=0)
	test1a = np.random.multivariate_normal([4.,-4.], cov, 50)
	test1b = np.random.multivariate_normal([-4.,4.], cov, 50)
	test1 = np.concatenate((test1a,test1b),axis=0)
	test = np.concatenate((test0,test1),axis=0)

	labeltrain0 = np.ones((np.shape(train0)[0],1))
	labeltrain1 = -np.ones((np.shape(train1)[0],1))
	labeltrain = np.concatenate((labeltrain0,labeltrain1),axis=0)
	labeltest0 = np.ones((np.shape(test0)[0],1))
	labeltest1 = -np.ones((np.shape(test1)[0],1))
	labeltest = np.concatenate((labeltest0,labeltest1),axis=0)

pl.figure()
pl.plot(train0[:,0], train0[:,1], "o",color="0.75")
pl.plot(train1[:,0], train1[:,1], "s",color="0.25")

import svm
reload(svm)

svm = svm.svm(kernel='linear',C=0.1)
#svm = svm.svm(kernel='rbf')
#svm = svm.svm(kernel='poly',C=0.1,degree=4)

print np.shape(train), np.shape(labeltrain)
svm.train_svm(train, labeltrain)
pl.scatter(svm.X[:,0], svm.X[:,1], s=200,color= 'k')

predict = svm.classifier(test,soft=False)
correct = np.sum(predict == labeltest)
print correct, np.shape(predict)
print float(correct)/np.shape(predict)[0]*100., "test accuracy"

# Classify points over 2D space to fit contour
x,y = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
xx = np.reshape(np.ravel(x),(2500,1))
yy = np.reshape(np.ravel(y),(2500,1))
points = np.concatenate((xx,yy),axis=1)
outpoints = svm.classifier(points,soft=True).reshape(np.shape(x))
pl.contour(x, y, outpoints, [0.0], colors='k', linewidths=1, origin='lower')
pl.contour(x, y, outpoints + 1, [0.0], colors='grey', linewidths=1, origin='lower')
pl.contour(x, y, outpoints - 1, [0.0], colors='grey', linewidths=1, origin='lower')

pl.axis("tight")
pl.show()

