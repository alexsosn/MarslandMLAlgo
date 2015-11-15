
# Code from Chapter 11 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The basic TD(0) algorithm with the Europe example

import numpy as np

def TDZero():

	R = np.array([[-5,0,-np.inf,-np.inf,-np.inf,-np.inf],[0,-5,0,0,-np.inf,-np.inf],[-np.inf,0,-5,0,-np.inf,100],[-np.inf,0,0,-5,0,-np.inf],[-np.inf,-np.inf,-np.inf,0,-5,100],[-np.inf,-np.inf,0,-np.inf,-np.inf,0]])
	t = np.array([[1,1,0,0,0,0],[1,1,1,1,0,0],[0,1,1,1,0,1],[0,1,1,1,1,0],[0,0,0,1,1,1],[0,0,1,0,1,1]])

	nStates = np.shape(R)[0]
	nActions = np.shape(R)[1]
	Q = np.random.rand(nStates,nActions)*0.1-0.05
	mu = 0.7
	gamma = 0.4
	epsilon = 0.1
	nits = 0

	while nits < 1000:
		# Pick initial state
		s = np.random.randint(nStates)
		# Stop when the accepting state is reached
		while s!=5:
			# epsilon-greedy
			if (np.random.rand()<epsilon):
				indices = np.where(t[s,:]!=0)
				pick = np.random.randint(np.shape(indices)[1])
				a = indices[0][pick]
				#print s,a
			else:
				a = np.argmax(Q[s,:])

			r = R[s,a]
			# For this example, new state is the chosen action
			sprime = a
			#print "here"
			Q[s,a] += mu * (r + gamma*np.max(Q[sprime,:]) - Q[s,a])
			s = sprime

		nits = nits+1

	print Q

TDZero()
