
# Code from Chapter 16 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import pylab as pl

def systematic(w,N):
	# Systematic resampling
	N2 = np.shape(w)[0]
	# One too many to make sure it is >1
	samples = np.random.rand(N+1)
	indices = np.arange(N+1)
	u = (samples+indices)/(N+1) 
	cumw = np.cumsum(w)
	keep = np.zeros((N))
	# ni copies of particle xi where ni = number of u between ws[i-1] and ws[i]
	j = 0
	for i in range(N2):
		while((u[j]<cumw[i]) & (j<N)):
			keep[j] = i
			j+=1

	return keep

def pf(x,y,sigma,T,N):

	particles = np.ones((N,T))
	particlepred = np.ones((N,T))
	ypred = np.ones((N,T))
	weights = np.ones((N,T))

	# Main loop
	for t in range(1,T):
	
		# importance sampling
		particlepred[:,t] = ffun(particles[:,t-1],t) + np.random.randn(N)
		ypred[:,t] = hfun(particlepred[:,t],t)
		weights[:,t] = 1./np.sqrt(sigma) * np.exp(-0.5/sigma * (y[t] - ypred[:,t])**2) + 1e-99
		weights[:,t] /= np.sum(weights[:,t])
		
		# selection
		sys= True
		if sys:
			keep = systematic(weights[:,t],N)
		else:	
			# Residual resampling
			# Add a little bit because of a rounding error!
			Ncopies = np.floor(weights[:,t]*N + 1e-10)
			keep = np.zeros((N))
			j = 0
			for i in range(N):
				keep[j:j+Ncopies[i]] = i
				j+=Ncopies[i]
				
			Nleft = int(N - np.sum(Ncopies))
			# Rest by systematic resampling
			if Nleft > 0:
				print "sys resample"
				probs = (weights[:,t]*N - Ncopies)/Nleft
				extrakeep = systematic(probs,Nleft)
				keep[j:] = extrakeep
		
		particles[:,t] = particlepred[keep.astype('int'),t]

	return particles, particlepred, ypred, weights

def ffun(x,t):
	return 1 + np.sin(4e-2*np.pi*t) + 0.5*x

def hfun(x,t):
	if t<30:
		return x**2/5.0
	else:
		return x/2. - 2.

def pf_demo():

	T = 50
	N = 10	
	sigma = 1.0
	
	x = np.zeros((T))
	x[1] = 1
	y = np.zeros((T))
	for t in range(T):
		x[t] = ffun(x[t-1],t) + np.random.randn(1)
		y[t] = hfun(x[t],t) + np.sqrt(sigma)*np.random.randn(1,1)

	p, pp, yp, w = pf(x,y,sigma,T,N)

	pl.figure()
	time = np.arange(T)
	pl.plot(time,y,'k+')
	pl.plot(time,x,'k:')
	#pl.plot(time,np.mean(p,axis=0),'.')
	pl.plot(time[:30],hfun(np.mean(p,axis=0)[:30],0),'k')
	pl.plot(time,p.T,'k.')
	pl.plot(time[30:],hfun(np.mean(p,axis=0)[30:],40),'k')
	#pl.axis([-0.5,9.5,-1,7])
	pl.legend(['Observation','Process','Output','Particles'])

