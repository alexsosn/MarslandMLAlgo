
# Code from Chapter 16 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

# 2D particle filter tracking. Euclidean distance based
import numpy as np
import pylab as pl

def systematic(w,N):
	# Systematic resampling
	# One too many to make sure it is >1
	samples = np.random.rand(N+1)
	indices = np.arange(N+1)
	u = (samples+indices)/N 
	cumw = np.cumsum(w)
	Ncopies = np.zeros((N))
	keep = np.zeros((N))
	# ni copies of particle xi where ni = number of u between ws[i-1] and ws[i]
	j = 0
	for i in range(N):
		while((u[j]<cumw[i]) & (j<N)):
			keep[j] = i
			Ncopies[i]+=1
			j+=1

	return keep

def pf(x0,xdot,sigma,T,N,width):

	# Sample x0 from prior p(x0)
	particles = np.zeros((N,2,T+1))
	x = np.zeros((2,T+1))
	x[:,0] = x0
	particles[:,:,0] = x0
	particlepred = np.zeros((N,2,T))
	particlepred[:,:,0] = x0+np.random.uniform(-width,width,(N,2))
	print particlepred[:,:,0]
	weights = np.ones((N,T))

	# Main loop
	for t in range(0,T):
	
		# importance sampling
		particlepred[:,:,t] = particles[:,:,t] + np.random.uniform(-width,width,(N,2))
		#print particlepred[:,:,t]

		print x[:,t]
		print x[:,t] - particlepred[:,:,t]
		weights[:,t] = np.sum((x[:,t] - particlepred[:,:,t])**2 + 1e-99,axis=1)
		print weights[:,t]
		weights[:,t] = 1./np.sum((x[:,t] - particlepred[:,:,t])**2 + 1e-99,axis=1)
		print weights[:,t]
		#weights[:,t] = np.sum(1./np.sqrt(sigma) * np.exp(-0.5/sigma * (x[:,t] - particlepred[:,:,t])**2) + 1e-99,axis=1)
		weights[:,t] /= np.sum(weights[:,t])
		print weights[:,t]
		
		# selection
		resample = False
		if 1./sum(weights[:,t]**2) < N/2.:
			print "Resampling"
			resample = True
		sys= True
		if resample:
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
		else:
			keep = range(N)
	
		print keep
		# output
		for i in range(N):
			particles[i,:,t+1] = particlepred[keep[i],:,t]
			#print "here"
		print x[:,t]
		print particlepred[:,:,t]
		#x[:,t+1] = x[:,t] + xdot*np.random.uniform(-1,1,(1,2))
		x[:,t+1] = x[:,t] + xdot #+ np.random.uniform(-1,1,(1,2))
		#print particles[:,:,t]

	return particles, x,weights

def pf_demo():

	x0 = np.array([10,12])
	xdot = np.array([10,8])

	np.random.seed(3)
	T = 15
	N = 30
	sigma = 1.0

	[particles,x,weights] = pf(x0,xdot,sigma,T,N,15)
	x = x[:,:T]
	particles = particles[:,:,:T]
	#print particles
	#print x

	dfilt = x[[0,1],:] - particles[[0,1],:]
	mse_filt = np.sqrt(np.sum(dfilt**2))

	#plot_track(x,y,xfilt,Pfilt)
	plot_position(x,particles,T)
	
def plot_position(x,particles,T):
	import time

	pl.ion()
	pl.figure()
	colours = pl.cm.gray(np.linspace(0, 1, T))

	#for t in [0,5,10,14]:
	for t in range(T):
		#print particles[:,:,t]
		pl.plot(x[0,t],x[1,t],'x',color=colours[t],ms=10.)
		pl.plot(particles[:,0,t],particles[:,1,t],'o',color=colours[t])
	#pl.plot(particles[:,0,5],particles[:,1,5],'go')
	#pl.plot(particles[:,0,10],particles[:,1,10],'co')
	#pl.plot(particles[:,0,14],particles[:,1,14],'ko')
	pl.xlim((0,150))
	pl.ylim((0,150))

def plot_track(x,y,Kx,P):
	fig = pl.figure()
	ax = fig.add_subplot(111, aspect='equal')
	pl.plot(x[0,:],x[1,:],'ks-')
	pl.plot(y[0,:],y[1,:],'k*')
	pl.plot(Kx[0,:],Kx[1,:],'kx:')

	obs_size,T = np.shape(y)

	from matplotlib.patches import Ellipse
	# Axes of ellipse are eigenvectors of covariance matrix, lengths are square roots of eigenvalues
	ellsize = np.zeros((obs_size,T))
	ellangle = np.zeros((T))
	for t in range(T):
		[evals,evecs] = np.linalg.eig(P[:2,:2,t])
		ellsize[:,t] = np.sqrt(evals)	
		ellangle[t] = np.angle(evecs[0,0]+0.j*evecs[0,1])
		
	ells = [Ellipse(xy=[Kx[0,t],Kx[1,t]] ,width=ellsize[0,t],height=ellsize[1,t], angle=ellangle[t]) for t in range(T)]
	for e in ells:
		ax.add_artist(e)
		e.set_alpha(0.1)
		e.set_facecolor([0.7,0.7,0.7])
