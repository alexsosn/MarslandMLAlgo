
# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The conjugate gradients algorithm
import numpy as np

def Jacobian(x):
	#return np.array([.4*x[0],2*x[1]])
	return np.array([x[0], 0.4*x[1], 1.2*x[2]])

def Hessian(x):
	#return np.array([[.2,0],[0,1]])
	return np.array([[1,0,0],[0,0.4,0],[0,0,1.2]])

def CG(x0):

	i=0
	k=0

	r = -Jacobian(x0)
	p=r

	betaTop = np.dot(r.transpose(),r)
	beta0 = betaTop

	iMax = 3
	epsilon = 10**(-2)
	jMax = 5

	# Restart every nDim iterations
	nRestart = np.shape(x0)[0]
	x = x0

	while i < iMax and betaTop > epsilon**2*beta0:
		j=0
		dp = np.dot(p.transpose(),p)
		alpha = (epsilon+1)**2
		# Newton-Raphson iteration
		while j < jMax and alpha**2 * dp > epsilon**2:
			# Line search
			alpha = -np.dot(Jacobian(x).transpose(),p) / (np.dot(p.transpose(),np.dot(Hessian(x),p)))
			print "N-R",x, alpha, p
			x = x + alpha * p
			j += 1
		print x
		# Now construct beta
		r = -Jacobian(x)
		print "r: ", r
		betaBottom = betaTop
		betaTop = np.dot(r.transpose(),r)
		beta = betaTop/betaBottom
		print "Beta: ",beta
		# Update the estimate
		p = r + beta*p
		print "p: ",p
		print "----"
		k += 1
		
		if k==nRestart or np.dot(r.transpose(),p) <= 0:
			p = r
			k = 0
			print "Restarting"
		i +=1

	print x

x0 = np.array([-2,2,-2])
CG(x0)
