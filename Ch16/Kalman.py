
# Code from Chapter 16 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The 1D Kalman filter

import pylab as pl
import numpy as np

def Kalman(obs=None,mu_init=np.array([-0.37727]),cov_init=0.1*np.ones((1)),nsteps=50):

    ndim = np.shape(mu_init)[0]
    
    if obs==None:
        mu_init = np.tile(mu_init,(1,nsteps))
        cov_init = np.tile(cov_init,(1,nsteps))
        obs = np.random.normal(mu_init,cov_init,(ndim,nsteps))
    
    Sigma_x = np.eye(ndim)*1e-5
    A = np.eye(ndim)
    H = np.eye(ndim)
    mu_hat = 0
    cov = np.eye(ndim)
    R = np.eye(ndim)*0.01
    
    m = np.zeros((ndim,nsteps),dtype=float)
    ce = np.zeros((ndim,nsteps),dtype=float)
    
    for t in range(1,nsteps):
        # Make prediction
        mu_hat_est = np.dot(A,mu_hat)
        cov_est = np.dot(A,np.dot(cov,np.transpose(A))) + Sigma_x

        # Update estimate
        error_mu = obs[:,t] - np.dot(H,mu_hat_est)
        error_cov = np.dot(H,np.dot(cov,np.transpose(H))) + R
        K = np.dot(np.dot(cov_est,np.transpose(H)),np.linalg.inv(error_cov))
        mu_hat = mu_hat_est + np.dot(K,error_mu)
        #m[:,:,t] = mu_hat
        m[:,t] = mu_hat
        if ndim>1:
            cov = np.dot((np.eye(ndim) - np.dot(K,H)),cov_est)
        else:
            cov = (1-K)*cov_est 
        ce[:,t] = cov                                
    
    pl.figure()
    pl.plot(obs[0,:],'ko',ms=6)
    pl.plot(m[0,:],'k-',lw=3)
    pl.plot(m[0,:]+20*ce[0,:],'k--',lw=2)
    pl.plot(m[0,:]-20*ce[0,:],'k--',lw=2)
    pl.legend(['Noisy Datapoints','Kalman estimate','Covariance'])
    pl.xlabel('Time')
    
    
    pl.show()
    
Kalman()
