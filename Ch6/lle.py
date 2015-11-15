
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The Locally Linear Embedding algorithm, and the swissroll example
import pylab as pl
import numpy as np

def swissroll():
	# Make the swiss roll dataset
	N = 1000
	noise = 0.05

	t = 3*np.pi/2 * (1 + 2*np.random.rand(1,N))
	h = 21 * np.random.rand(1,N)
	data = np.concatenate((t*np.cos(t),h,t*np.sin(t))) + noise*np.random.randn(3,N)	
	return np.transpose(data), np.squeeze(t)

def lle(data,nRedDim=2,K=12):

	ndata = np.shape(data)[0]
	ndim = np.shape(data)[1]
	d = np.zeros((ndata,ndata),dtype=float)
	
	# Inefficient -- not matrices
	for i in range(ndata):
		for j in range(i+1,ndata):
			for k in range(ndim):
				d[i,j] += (data[i,k] - data[j,k])**2
			d[i,j] = np.sqrt(d[i,j])
			d[j,i] = d[i,j]

	indices = d.argsort(axis=1)
	neighbours = indices[:,1:K+1]

	W = np.zeros((K,ndata),dtype=float)

	for i in range(ndata):
		Z  = data[neighbours[i,:],:] - np.kron(np.ones((K,1)),data[i,:])
		C = np.dot(Z,np.transpose(Z))
		C = C+np.identity(K)*1e-3*np.trace(C)
		W[:,i] = np.transpose(np.linalg.solve(C,np.ones((K,1))))
		W[:,i] = W[:,i]/np.sum(W[:,i])

	M = np.eye(ndata,dtype=float)
	for i in range(ndata):
		w = np.transpose(np.ones((1,np.shape(W)[0]))*np.transpose(W[:,i]))
		j = neighbours[i,:]
		#print shape(w), np.shape(np.dot(w,np.transpose(w))), np.shape(M[i,j])
		ww = np.dot(w,np.transpose(w))
		for k in range(K):
			M[i,j[k]] -= w[k]
			M[j[k],i] -= w[k]
			for l in range(K):
			     M[j[k],j[l]] += ww[k,l]
	
	evals,evecs = np.linalg.eig(M)
	ind = np.argsort(evals)
	y = evecs[:,ind[1:nRedDim+1]]*np.sqrt(ndata)
	return evals,evecs,y

data,t = swissroll()
evals,evecs,y = lle(data)

t -= t.min()
t /= t.max()
pl.scatter(y[:,0],y[:,1],s=50,c=t,cmap=pl.cm.gray)
pl.axis('off')
pl.show()
