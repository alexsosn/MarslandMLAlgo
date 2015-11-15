
# Code from Chapter 15 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The basic importance sampling algorithm
import pylab as pl
import numpy as np

def qsample():
    return np.random.rand()*4.

def p(x):
    return 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 

def q(x):
    return 4.0

def importance(nsamples):
    
    samples = np.zeros(nsamples,dtype=float)
    w = np.zeros(nsamples,dtype=float)
    
    for i in range(nsamples):
            samples[i] = qsample()
            w[i] = p(samples[i])/q(samples[i])
                
    return samples, w

x = np.arange(0,4,0.01)
x2 = np.arange(-0.5,4.5,0.1)
realdata = 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 
box = np.ones(len(x2))*0.8
box[:5] = 0
box[-5:] = 0
pl.plot(x,realdata,'k',lw=6)
pl.plot(x2,box,'k--',lw=6)

samples,w = importance(5000)
pl.hist(samples,normed=1,fc='k')
#pl.xlabel('x',fontsize=24)
#pl.ylabel('p(x)',fontsize=24)
pl.show()
