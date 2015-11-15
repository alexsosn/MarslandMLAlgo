
# Code from Chapter 15 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The Sampling-Importance-Resampling algorithm
import pylab as pl
import numpy as np

def p(x):
    return 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 

def q(x):
    return 4.0

def sir(n):
    
    sample1 = np.zeros(n)
    w = np.zeros(n)
    sample2 = np.zeros(n)
    
    # Sample from q
    sample1 = np.random.rand(n)*4

    # Compute weights
    w = p(sample1)/q(sample1)
    w /= np.sum(w)

    # Sample from sample1 according to w
    cumw = np.zeros(len(w))
    cumw[0] = w[0]
    for i in range(1,len(w)):
        cumw[i] = cumw[i-1]+w[i]
    
    u = np.random.rand(n)
    
    index = 0
    for i in range(n):
        indices = np.where(u<cumw[i])
        sample2[index:index+np.size(indices)] = sample1[i]
        index += np.size(indices)
        u[indices]=2
    return sample2

x = np.arange(0,4,0.01)
x2 = np.arange(-0.5,4.5,0.1)
realdata = 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 
box = np.ones(len(x2))*0.8
box[:5] = 0
box[-5:] = 0
pl.plot(x,realdata,'k',lw=6)
pl.plot(x2,box,'k--',lw=6)

import time
t0=time.time()
samples = sir(10000)
t1=time.time()
print t1-t0
pl.hist(samples,15,normed=1,fc='k')
pl.xlabel('x',fontsize=24)
pl.ylabel('p(x)',fontsize=24)
pl.axis([-0.5,4.5,0,1])
pl.show()
