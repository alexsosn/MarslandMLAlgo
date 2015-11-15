
# Code from Chapter 16 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# A basic Hidden Markov Model
import numpy as np

scaling = False

def HMMfwd(pi,a,b,obs):

	nStates = np.shape(b)[0]
	T = np.shape(obs)[0]

	alpha = np.zeros((nStates,T))

	alpha[:,0] = pi*b[:,obs[0]]

	for t in range(1,T):
		for s in range(nStates):
			alpha[s,t] = b[s,obs[t]] * np.sum(alpha[:,t-1] * a[:,s])

	c = np.ones((T))
	if scaling:
		for t in range(T):
			c[t] = np.sum(alpha[:,t])
			alpha[:,t] /= c[t]
	return alpha,c

def HMMbwd(a,b,obs,c):

	nStates = np.shape(b)[0]
	T = np.shape(obs)[0]

	beta = np.zeros((nStates,T))

	beta[:,T-1] = 1.0 #aLast

	for t in range(T-2,-1,-1):
		for s in range(nStates):
			beta[s,t] = np.sum(b[:,obs[t+1]] * beta[:,t+1] * a[s,:])

	for t in range(T):
		beta[:,t] /= c[t]
	#beta[:,0] = b[:,obs[0]] * np.sum(beta[:,1] * pi)
	return beta

def Viterbi(pi,a,b,obs):

	nStates = np.shape(b)[0]
	T = np.shape(obs)[0]

	path = np.zeros(T)
	delta = np.zeros((nStates,T))
	phi = np.zeros((nStates,T))

	delta[:,0] = pi * b[:,obs[0]]
	phi[:,0] = 0

	for t in range(1,T):
		for s in range(nStates):
			delta[s,t] = np.max(delta[:,t-1]*a[:,s])*b[s,obs[t]]
			phi[s,t] = np.argmax(delta[:,t-1]*a[:,s])

	path[T-1] = np.argmax(delta[:,T-1])
	for t in range(T-2,-1,-1):
		path[t] = phi[path[t+1],t+1]

	return path,delta, phi

def BaumWelch(obs,nStates):

	T = np.shape(obs)[0]
	xi = np.zeros((nStates,nStates,T))

	# Initialise pi, a, b randomly
	pi = 1./nStates*np.ones((nStates))
	a = np.random.rand(nStates,nStates)
	b = np.random.rand(nStates,np.max(obs)+1)

	tol = 1e-5
	error = tol+1
	maxits = 100
	nits = 0
	while ((error > tol) & (nits < maxits)):
		nits += 1
		oldpi = pi.copy()
		olda = a.copy()
		oldb = b.copy()

		# E step
		alpha,c = HMMfwd(pi,a,b,obs)
		beta = HMMbwd(a,b,obs,c) 

		for t in range(T-1):
			for i in range(nStates):
				for j in range(nStates):
					xi[i,j,t] = alpha[i,t]*a[i,j]*b[j,obs[t+1]]*beta[j,t+1]
			xi[:,:,t] /= np.sum(xi[:,:,t])

		# The last step has no b, beta in
		for i in range(nStates):
			for j in range(nStates):
				xi[i,j,T-1] = alpha[i,T-1]*a[i,j]
		xi[:,:,T-1] /= np.sum(xi[:,:,T-1])

		# M step
		for i in range(nStates):
			pi[i] = np.sum(xi[i,:,0])
			for j in range(nStates):
				a[i,j] = np.sum(xi[i,j,:T-1])/np.sum(xi[i,:,:T-1])
	
			for k in range(max(obs)):
				found = (obs==k).nonzero()
				b[i,k] = np.sum(xi[i,:,found])/np.sum(xi[i,:,:])

		error = (np.abs(a-olda)).max() + (np.abs(b-oldb)).max() 
		print nits, error, 1./np.sum(1./c), np.sum(alpha[:,T-1])

	return pi, a, b	
		
def evenings():
	pi = np.array([0.25, 0.25, 0.25, 0.25])
	a = np.array([[0.05,0.7, 0.05, 0.2],[0.1,0.4,0.3,0.2],[0.1,0.6,0.05,0.25],[0.25,0.3,0.4,0.05]])
	b = np.array([[0.3,0.4,0.2,0.1],[0.2,0.1,0.2,0.5],[0.4,0.2,0.1,0.3],[0.3,0.05,0.3,0.35]])
	
	obs = np.array([3,1,1,3,0,3,3,3,1,1,0,2,2])
	print Viterbi(pi,a,b,obs)[0]
	alpha,c = HMMfwd(pi,a,b,obs)
	print np.sum(alpha[:,-1])

def test():
	np.random.seed(4)
	pi = np.array([0.25,0.25,0.25,0.25])
	aLast = np.array([0.25,0.25,0.25,0.25])
	#a = np.array([[.7,.3],[.4,.6]] )
	a = np.array([[.4,.3,.1,.2],[.6,.05,.1,.25],[.7,.05,.05,.2],[.3,.4,.25,.05]])
	#b = np.array([[.2,.4,.4],[.5,.4,.1]] )
	b = np.array([[.2,.1,.2,.5],[.4,.2,.1,.3],[.3,.4,.2,.1],[.3,.05,.3,.35]])
	obs = np.array([0,0,3,1,1,2,1,3])
	#obs = np.array([2,0,2])
	HMMfwd(pi,a,b,obs)
	Viterbi(pi,a,b,obs)
	print BaumWelch(obs,4)
	
def biased_coins():
	a = np.array([[0.4,0.6],[0.9,0.1]])
	b = np.array([[0.49,0.51],[0.85,0.15]])
	pi = np.array([0.5,0.5])

	obs = np.array([0,1,1,0,1,1,0,0,1,1,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0])	
	print Viterbi(pi,a,b,obs)[0]

	print BaumWelch(obs,2)

