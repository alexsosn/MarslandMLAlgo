
# Code from Chapter 16 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import pylab as pl

def EKF_update(Q,R,y,x,t,Sig,B=None,u=None):

	if B is None:
		xpred = f(x,t).reshape(len(x),1)
	else:
		xpred = f(x,t).reshape(len(x),1) + np.dot(B,u)

	A = np.array(jac_f(x,t))
	#A = jac_f(x,t).reshape(1,len(x))
	SigPred = np.dot(A,np.dot(Sig,A.T)) + Q

	H = jac_h(xpred).reshape(1,len(x))
	e = y - h(xpred)
	Sinv = 1./(np.dot(H,np.dot(SigPred,H.T)) + R)
	#Sinv = np.linalg.inv(np.dot(H,np.dot(SigPred,H.T)) + R)
	K = np.dot(SigPred,np.dot(H.T,Sinv))

	xnew = xpred + np.dot(K,e)
	SigNew = np.dot((1 - np.dot(K,H)),SigPred)

	return xnew.T,SigNew
	
def EKF(y,Q,R,x0,Sig0,B=None,u=None):

	obs_size,T = np.shape(y)
	state_size = np.shape(Q)[0]

	x = np.zeros((state_size,T))
	Sig = np.zeros((state_size,state_size,T))

	[x[:,0],Sig[:,:,0]] = EKF_update(Q,R,y[:,0].reshape(len(y),1),x0,0,Sig0,B,u)
	for t in range(1,T):
		prevx = x[:,t-1].reshape(state_size,1)
		prevSig = Sig[:,:,t-1]
		[x[:,t],Sig[:,:,t]] = EKF_update(Q,R,y[:,t].reshape(len(y),1),prevx,t,prevSig,B,u)

	return x,Sig

def f(x,t):
	return np.array([x[1],x[2],0.5*x[0]*(x[1]+x[2])]).T

def jac_f(x,t):
	return np.array([[0,0,0.5*(x[1]+x[2])],[1,0,0.5*x[0]],[0,1,0.5*x[0]]])
	#return x

def jac_h(x):
	#return np.array([x[2]*np.sin(x[0]), 0, np.cos(x[0])])
	return np.array([1.0,1.0,0.0]) #np.array([0.4*x]).reshape(1,1)
	#return np.array([1.0,0.0,0.0]) #np.array([0.4*x]).reshape(1,1)

def h(x):
	return x[0]+x[1]

def EKF_demo():

	state_size = 3
	observation_size = 1

	Q = 0.1*np.eye(state_size)
	R = 0.1*np.eye(observation_size)
	x0 = np.array([0,0,1])
	Sig0 = np.eye(state_size)
	
	T = 20
	state = np.zeros((state_size,T))
	y = np.zeros((observation_size,T))
	state[:,0] = x0.T

	for t in range(1,T):
		state[:,t] = f(state[:,t-1],t) + np.random.multivariate_normal(np.zeros((len(Q))),Q)
		y[:,t] = h(state[:,t]) + np.sqrt(R)*np.random.randn()
		#state[:,t] = np.dot(A,state[:,t-1]) + np.random.multivariate_normal(np.zeros((len(Q))),Q)
		#y[:,t] = h(state[:,t]) + np.random.randn()

	[xfilt,Sigfilt] = EKF(y,Q,R,x0,Sig0)

	dfilt = state[0,:] - xfilt[0,:]
	#dfilt = state[[0,1],:] - xfilt[[0,1],:]
	mse_filt = np.sqrt(np.sum(dfilt**2))
	print mse_filt

	ypred = np.zeros((T))
	for t in range(T):
		ypred[t] = h(xfilt[:,t])

	print Sigfilt
	pl.figure()
	pl.plot(np.arange(T),np.squeeze(y),'*')
	pl.plot(np.arange(T),ypred,'k-')
	
