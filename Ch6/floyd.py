
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import time

def floyd():
    
    ndata = 100
    neighbours = np.zeros((ndata,10))
    g = np.random.rand(ndata,ndata)
    for i in range(ndata):
        neighbours[i,:] = np.random.randint(0,100,10) 
    
    t0 = time.time()
    print "Floyd's algorithm"
    for k in range(ndata):
        for i in range(ndata):
            for j in range(ndata):
                if g[i,j] > g[i,k] + g[k,j]:
                    g[i,j] = g[i,k] + g[k,j]
    
    t1 = time.time()
    print "Complete"
    print t1-t0
    x = g.copy()

    t2 = time.time()
    q = g.copy()
    for i in range(ndata):
        for j in range(ndata):
            k = np.argmin(q[i,:])
            while not(np.isnan(q[i,k])):
                q[i,k] = np.nan
                for l in neighbours[k,:]:
                    possible = q[i,l] + q[l,k]
                    if possible < q[i,k]:
                        g[i,k] = possible
                k = np.argmin(q[i,:])
    t3 = time.time()
    y = g
    print t3-t2
    return x,y
