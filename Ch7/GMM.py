
# Code from Chapter 7 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import pylab as pl
import numpy as np

def GMM():

    """ Fits two Gaussians to data using the EM algorithm """
    N = 100
    pl.ion()
    
    y = 1.*np.zeros(N)
    # Set up data
    out1 = np.random.normal(6,1,N)
    out2 = np.random.normal(1,1,N)
    choice = np.random.rand(N)
    
    w = [choice>=0.5]
    y[w] = out1[w]	
    w = [choice<0.5]
    y[w] = out2[w]
    
    pl.clf()
    pl.hist(y,fc='0.5')
    
    # Now do some learning

    # Initialisation
    mu1 = y[np.random.randint(0,N-1,1)]
    mu2 = y[np.random.randint(0,N-1,1)]
    s1 = np.sum((y-np.mean(y))**2)/N
    s2 = s1
    pi = 0.5

    # EM loop
    count = 0
    gamma = 1.*np.zeros(N)
    nits = 20

    ll = 1.*np.zeros(nits)
	
    while count<nits:
        count = count + 1

    	# E-step
        for i in range(N):
            gamma[i] = pi*np.exp(-(y[i]-mu1)**2/(2*s1))/ (pi * np.exp(-(y[i]-mu1)**2/(2*s1)) + (1-pi)* np.exp(-(y[i]-mu2)**2/2*s2))
        
    	# M-step
        mu1 = np.sum((1-gamma)*y)/np.sum(1-gamma)
        mu2 = np.sum(gamma*y)/np.sum(gamma)
        s1 = np.sum((1-gamma)*(y-mu1)**2)/np.sum(1-gamma)
        s2 = np.sum(gamma*(y-mu2)**2)/np.sum(gamma)
        pi = np.sum(gamma)/N
        	
        ll[count-1] = np.sum(np.log(pi*np.exp(-(y[i]-mu1)**2/(2*s1)) + (1-pi)*np.exp(-(y[i]-mu2)**2/(2*s2))))

    x = np.arange(-2,8.5,0.1)
    y = 35*pi*np.exp(-(x-mu1)**2/(2*s1)) + 35*(1-pi)*np.exp(-(x-mu2)**2/(2*s2))

    pl.plot(x,y,'k',linewidth=4)
    pl.figure(), pl.plot(ll,'ko-')
    pl.show()
    
GMM()
