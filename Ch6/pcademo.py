
# Code from Chapter 6 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# A simple example of PCA
import pylab as pl
import numpy as np

import pca

x = np.random.normal(5,.5,1000)
y = np.random.normal(3,1,1000)
a = x*np.cos(np.pi/4) + y*np.sin(np.pi/4)
b = -x*np.sin(np.pi/4) + y*np.cos(np.pi/4)

pl.plot(a,b,'.')
pl.xlabel('x')
pl.ylabel('y')
pl.title('Original dataset')
data = np.zeros((1000,2))
data[:,0] = a
data[:,1] = b

x,y,evals,evecs = pca.pca(data,1)
print y
pl.figure()
pl.plot(y[:,0],y[:,1],'.')
pl.xlabel('x')
pl.ylabel('y')
pl.title('Reconstructed data after PCA')
pl.show()
