
# Code from Chapter 2 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Plots a 1D Gaussian function
import pylab as pl
import numpy as np

gaussian = lambda x: 1/(np.sqrt(2*np.pi)*1.5)*np.exp(-(x-0)**2/(2*(1.5**2)))
x = np.arange(-5,5,0.01)
y = gaussian(x)
pl.ion()
pl.plot(x,y,'k',linewidth=3)
pl.xlabel('x')
pl.ylabel('y(x)')
pl.axis([-5,5,0,0.3])
pl.title('Gaussian Function (mean 0, standard deviation 1.5)')
pl.show()
