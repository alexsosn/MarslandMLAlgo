
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The Isomap algorithm
import pylab as pl
import numpy as np

def swissroll():
	# Make the swiss roll dataset
	N = 1000
	noise = 0.05

	t = 3.*np.pi/2 * (1. + 2.*np.random.rand(1,N))
	h = 21. * np.random.rand(1,N)
	data = np.concatenate((t*np.cos(t),h,t*np.sin(t))) + noise*np.random.randn(3,N)	
	return np.transpose(data), np.squeeze(t)

def isomap(data,newdim=2,K=12,labels=None):

	ndata = np.shape(data)[0]
	ndim = np.shape(data)[1]
	d = np.zeros((ndata,ndata),dtype=float)
	
	# Compute the distance matrix
	# Inefficient -- not matrices
	for i in range(ndata):
		for j in range(i+1,ndata):
			for k in range(ndim):
				d[i,j] += (data[i,k] - data[j,k])**2
			d[i,j] = np.sqrt(d[i,j])
			d[j,i] = d[i,j]

	# K-nearest neighbours
	indices = d.argsort()
	#notneighbours = indices[:,K+1:]
	neighbours = indices[:,:K+1]
	# Alternative: epsilon
	# epsilon = 0.1
	#neighbours = where(d<=epsilon)
	#notneighbours = where(d>epsilon)

	h = np.ones((ndata,ndata),dtype=float)*np.inf
	for i in range(ndata):
		h[i,neighbours[i,:]] = d[i,neighbours[i,:]]

	# Compute the full distance matrix over all paths
	print "Floyd's algorithm"
	for k in range(ndata):
		for i in range(ndata):
			for j in range(ndata):
				if h[i,j] > h[i,k] + h[k,j]:
					h[i,j] = h[i,k] + h[k,j]

#	print "Dijkstra's algorithm"
#	q = h.copy()
#	for i in range(ndata):
#		for j in range(ndata):
#			k = np.argmin(q[i,:])
#			while not(np.isinf(q[i,k])):
#				q[i,k] = np.inf
#				for l in neighbours[k,:]:
#					possible = h[i,l] + h[l,k]
#					if possible < h[i,k]:
#						h[i,k] = possible
#				k = np.argmin(q[i,:])
#	print "Comnlete"

	# remove lines full of infs 
	x = np.isinf(h[:,0]).nonzero()
	if np.size(x)>0:	
		print x
		if x[0][0]>0:
			new = h[0:x[0][0],:]
			newlabels = labels[0:x[0][0]]
			start = 1
		else:
			new = h[x[0][0]+1,:]
			newlabels = labels[x[0][0]+1]
			start = 2
		for i in range(start,size(x)):
			new = np.concatenate((new,h[x[0][i-1]+1:x[0][i],:]),axis=0)
			newlabels = np.concatenate((newlabels,labels[x[0][i-1]+1:x[0][i]]),axis=0)
		new = np.concatenate((new,h[x[0][i]+1:,:]),axis=0)
		newlabels = np.concatenate((newlabels,labels[x[0][i]+1:]),axis=0)

		new2 = new[:,0:x[0][0]]
		if x[0][0]>0:
			new2 = new[:,0:x[0][0]]
			start = 1
		else:
			new2 = new[:,x[0][0]+1]
			start = 2
		for i in range(start,size(x)):
			new2 = np.concatenate((new2,new[:,x[0][i-1]+1:x[0][i]]),axis=1)
		new2 = np.concatenate((new2,new[:,x[0][i]+1:]),axis=1)

		g = new2.copy()
		ndata = ndata - size(x)
	else:
		g = h.copy()
		newlabels = labels
	
	# Map computations, following by the dimensionality reduction
	M = -0.5*(g**2 - np.transpose(np.sum(g*g,axis=0) * np.ones((ndata,1))/ndata) - np.ones((ndata,1))* np.sum(g*g,axis=0)/ndata + np.sum(np.sum(g*g))/ndata**2)

	eval,evec = np.linalg.eig(M)
	eval = np.real(eval)
	ind = np.argsort(eval)
	eval = np.real(np.diag(eval[ind[-1::-1]]))
	evec = evec[:,ind[-1::-1]]
	y = np.real(np.dot(evec,np.transpose((np.sqrt(eval)))))
	print np.shape(y)
	print np.shape(eval), np.shape(evec)
	return y, newlabels

data,t = swissroll()
y,u = isomap(data)

t -= t.min()
t /= t.max()
#pl.scatter(y[:,0],y[:,1],c=t,cmap=pl.cm.jet)
pl.scatter(y[:,1],y[:,2],s=50,c=t,cmap=pl.cm.gray)
#pl.scatter(data[:,0],data[:,1],s=50,c=t,cmap=pl.cm.gray)
 
pl.show()
