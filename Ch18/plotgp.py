
# Code from Chapter 18 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import pylab as pl

a = 1. + 0.25*np.random.randn(10)
#a = 0.25*np.random.randn(10)
b = 1. + np.random.randn(10)

#x = np.linspace(0,2,100)
#f = lambda x: np.exp(ai*x) * np.cos(bi*x)
x = np.linspace(-2,2,100)
f = lambda x: np.exp(-(ai*a)x**2)

pl.figure()
for i in range(10):
	ai = a[i] 
	bi = b[i]
	pl.plot(x,f(x),'k-')
	pl.xlabel('x')
	pl.ylabel('f(x)')

x1 = 0.5
y1 = 0.5
x2 = 1.5
y2 = 1.0


a = 1.+0.25*np.random.randn(10000)
#a = 0.25*np.random.randn(10000)
b = 1. + np.random.randn(10000)

nbins = 25
pl.figure()
f1 = np.exp(-a**2)
#f1 = np.exp(a) * np.cos(b)
c = pl.hist(f1,bins=nbins)
p = c[0]/np.max(c[0])
pl.figure()
pl.plot(c[1][:nbins],p,'-k')
pl.xlabel('f(1)')
pl.ylabel('Pr(f(1))')




