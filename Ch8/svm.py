
# Code from Chapter 8 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import cvxopt as cvxopt
from cvxopt import solvers
import pylab as pl

# Todo:
# comments to explain matrix transforms
# a decent example

class svm:

	def __init__(self,kernel='linear',C=None,sigma=1.,degree=1.,threshold=1e-5):
		self.kernel = kernel
		if self.kernel == 'linear':
			self.kernel = 'poly'
			self.degree = 1.
		self.C = C
		self.sigma = sigma
		self.degree = degree
		self.threshold = threshold

	def build_kernel(self,X):
		self.K = np.dot(X,X.T)

		if self.kernel=='poly':
			self.K = (1. + 1./self.sigma*self.K)**self.degree

		elif self.kernel=='rbf':
			self.xsquared = (np.diag(self.K)*np.ones((1,self.N))).T
			b = np.ones((self.N,1))
			self.K -= 0.5*(np.dot(self.xsquared,b.T) + np.dot(b,self.xsquared.T))
			self.K = np.exp(self.K/(2.*self.sigma**2))

	def train_svm(self,X,targets):
		self.N = np.shape(X)[0]
		self.build_kernel(X)

		# Assemble the matrices for the constraints
		P = targets*targets.transpose()*self.K
		q = -np.ones((self.N,1))
		if self.C is None:
			G = -np.eye(self.N)
			h = np.zeros((self.N,1))
		else:
			G = np.concatenate((np.eye(self.N),-np.eye(self.N)))
			h = np.concatenate((self.C*np.ones((self.N,1)),np.zeros((self.N,1))))
		A = targets.reshape(1,self.N)
		b = 0.0

		# Call the quadratic solver
		sol = cvxopt.solvers.qp(cvxopt.matrix(P),cvxopt.matrix(q),cvxopt.matrix(G),cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))

		# Get the Lagrange multipliers out of the solution dictionary
		lambdas = np.array(sol['x'])

		# Find the (indices of the) support vectors, which are the vectors with non-zero Lagrange multipliers
		self.sv = np.where(lambdas>self.threshold)[0]
		self.nsupport = len(self.sv)
		print self.nsupport, "support vectors found" 

		# Just retain the data corresponding to the support vectors
		self.X = X[self.sv,:]
		self.lambdas = lambdas[self.sv]
		self.targets = targets[self.sv]

        	#self.b = np.sum(self.targets)
        	#for n in range(self.nsupport):
			#self.b -= np.sum(self.lambdas*self.targets.T*np.reshape(self.K[self.sv[n],self.sv],(self.nsupport,1)))
        	#self.b /= len(self.lambdas)
		#print "b=",self.b

        	self.b = np.sum(self.targets)
        	for n in range(self.nsupport):
			self.b -= np.sum(self.lambdas*self.targets*np.reshape(self.K[self.sv[n],self.sv],(self.nsupport,1)))
        	self.b /= len(self.lambdas)
		#print "b=",self.b

		#bb = 0
		#for j in range(self.nsupport):
			#tally = 0	
			#for i in range(self.nsupport):
				#tally += self.lambdas[i]*self.targets[i]*self.K[self.sv[j],self.sv[i]]
			#bb += self.targets[j] - tally
		#self.bb = bb/self.nsupport
		#print self.bb
				
		if self.kernel == 'poly':
			def classifier(Y,soft=False):
				K = (1. + 1./self.sigma*np.dot(Y,self.X.T))**self.degree

				self.y = np.zeros((np.shape(Y)[0],1))
				for j in range(np.shape(Y)[0]):
					for i in range(self.nsupport):
						self.y[j] += self.lambdas[i]*self.targets[i]*K[j,i]
					self.y[j] += self.b
				
				if soft:
					return self.y
				else:
					return np.sign(self.y)
	
		elif self.kernel == 'rbf':
			def classifier(Y,soft=False):
				K = np.dot(Y,self.X.T)
				c = (1./self.sigma * np.sum(Y**2,axis=1)*np.ones((1,np.shape(Y)[0]))).T
				c = np.dot(c,np.ones((1,np.shape(K)[1])))
				aa = np.dot(self.xsquared[self.sv],np.ones((1,np.shape(K)[0]))).T
				K = K - 0.5*c - 0.5*aa
				K = np.exp(K/(2.*self.sigma**2))

				self.y = np.zeros((np.shape(Y)[0],1))
				for j in range(np.shape(Y)[0]):
					for i in range(self.nsupport):
						self.y[j] += self.lambdas[i]*self.targets[i]*K[j,i]
					self.y[j] += self.b

				if soft:
					return self.y
				else:
					return np.sign(self.y)
		else:
			print "Error -- kernel not recognised"
			return

		self.classifier = classifier

def modified_XOR(kernel,degree,C,sdev):
	import svm
	sv = svm.svm(kernel,degree=degree,C=C)
	#sv = svm.svm(kernel='poly',degree=3,C=0.2)
	#sv = svm.svm(kernel='rbf',C=0.1)
	#sv = svm.svm(kernel='poly',degree=3)
	#sdev = 0.4 #0.3 #0.1

	m = 100
	X = sdev*np.random.randn(m,2)
	X[m/2:,0] += 1.
	X[m/4:m/2,1] += 1.
	X[3*m/4:,1] += 1.
	targets = -np.ones((m,1))
	targets[:m/4,0] = 1.
	targets[3*m/4:,0] = 1.
	#targets = (np.where(X[:,0]*X[:,1]>=0,1,-1)*np.ones((1,np.shape(X)[0]))).T
	
	sv.train_svm(X,targets)

	Y = sdev*np.random.randn(m,2)
	Y[m/2:,0] += 1.
	Y[m/4:m/2,1] += 1.
	Y[3*m/4:m,1] += 1.
	test = -np.ones((m,1))
	test[:m/4,0] = 1.
	test[3*m/4:,0] = 1.

	#test = (np.where(Y[:,0]*Y[:,1]>=0,1,-1)*np.ones((1,np.shape(Y)[0]))).T
	#print test.T
	output = sv.classifier(Y,soft=False)
	#print output.T
	#print test.T
	err1 = np.where((output==1.) & (test==-1.))[0]
	err2 = np.where((output==-1.) & (test==1.))[0]
	print kernel, C
	print "Class 1 errors ",len(err1)," from ",len(test[test==1])
	print "Class 2 errors ",len(err2)," from ",len(test[test==-1])
	print "Test accuracy ",1. -(float(len(err1)+len(err2)))/ (len(test[test==1]) + len(test[test==-1]))

	pl.ion()
	pl.figure()
	l1 =  np.where(targets==1)[0]
	l2 =  np.where(targets==-1)[0]
	pl.plot(X[sv.sv,0],X[sv.sv,1],'o',markeredgewidth=5)
	pl.plot(X[l1,0],X[l1,1],'ko')
	pl.plot(X[l2,0],X[l2,1],'wo')
	l1 =  np.where(test==1)[0]
	l2 =  np.where(test==-1)[0]
	pl.plot(Y[l1,0],Y[l1,1],'ks')
	pl.plot(Y[l2,0],Y[l2,1],'ws')

	step = 0.1
	f0,f1  = np.meshgrid(np.arange(np.min(X[:,0])-0.5, np.max(X[:,0])+0.5, step), np.arange(np.min(X[:,1])-0.5, np.max(X[:,1])+0.5, step))

	out = sv.classifier(np.c_[np.ravel(f0), np.ravel(f1)],soft=True).T

	out = out.reshape(f0.shape)
	pl.contour(f0, f1, out,2)

	pl.axis('off')
	pl.show()

def run_mxor():
	#for sdev in [0.1]:
	for sdev in [0.1, 0.3, 0.4]:
		modified_XOR('linear',1,None,sdev)
		modified_XOR('linear',1,0.1,sdev)
		modified_XOR('poly',3,None,sdev)
		modified_XOR('poly',3,0.1,sdev)
		modified_XOR('rbf',0,None,sdev)
		modified_XOR('rbf',0,0.1,sdev)


