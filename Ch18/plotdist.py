
# Code from Chapter 18 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import pylab as pl
import numpy as np

# Weibull and Gaussian fits

pl.ion()
pl.figure()
l = 1.
k = 2.
x = np.random.random(27)*2.
x = np.concatenate((x,np.random.rand(9)+2.))
xx = k/l * (x/l)**(k-1) * np.exp(-(x/l)**k) + np.random.random(36)*0.2-0.1
pl.plot(x,xx,'o')

pl.figure()
pl.plot(x,xx,'o')
x = np.arange(0,3,0.01)
s = 0.5
mu = 0.7
y = 1/(np.sqrt(2*np.pi)*s) * np.exp(-0.5*(x-mu)**2/s**2)
pl.plot(x,y,'k')


z = k/l * (x/l)**(k-1) * np.exp(-(x/l)**k)
pl.plot(x,z,'r--')
