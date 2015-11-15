
# Code from Chapter 11 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The basic SARSA algorithm with the Europe example

import numpy as np
def SARSA():

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
        # epsilon-greedy
        if (np.random.rand()<epsilon):
            indices = np.where(t[s,:]!=0)
            pick = np.random.randint(np.shape(indices)[1])
            a = indices[0][pick]
        else:
            a = np.argmax(Q[s,:])
                
        # Stop when the accepting state is reached
        while s!=5:
            r = R[s,a]
            # For this example, new state is the chosen action
            sprime = a
            
            # epsilon-greedy
            if (np.random.rand()<epsilon):
                indices = np.where(t[sprime,:]!=0)
                pick = np.random.randint(np.shape(indices)[1])
                aprime = indices[0][pick]
                #print s,a
            else:
                aprime = np.argmax(Q[sprime,:])
            #print "here", Q[sprime,aprime], Q[s,a], s, a
            
            Q[s,a] += mu * (r + gamma*Q[sprime,aprime] - Q[s,a])

            s = sprime
            a = aprime
            
        nits = nits+1

    print Q

SARSA()
