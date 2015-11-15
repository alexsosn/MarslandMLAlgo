
# Code from Chapter 15 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The Box-Muller algorithm for constructing pseudo-random Gaussian-distributed numbers

import pylab as pl
import numpy as np

def boxmuller(n):
    
    x = np.zeros((n,2))
    y = np.zeros((n,2))
    
    for i in range(n):
        x[i,:] = np.array([2,2])
        x2 = x[i,0]*x[i,0]+x[i,1]*x[i,1]
        while (x2)>1:
            x[i,:] = np.random.rand(2)*2-1
            x2 = x[i,0]*x[i,0]+x[i,1]*x[i,1]

        y[i,:] = x[i,:] * np.sqrt((-2*np.log(x2))/x2)
    
    y = np.reshape(y,2*n,1)
    return y

y = boxmuller(1000)
pl.hist(y,normed=1,fc='k')
x = np.arange(-4,4,0.1)
pl.plot(x,1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2),'k',lw=6)
pl.xlabel('x',fontsize=24)
pl.ylabel('p(x)',fontsize=24)
pl.show()
    
    
