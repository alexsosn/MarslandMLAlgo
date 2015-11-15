
# Code from Chapter 16 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import pylab as pl

def Kalman_update(A,H,Q,R,y,x,Sig,B=None,u=None):

	if B is None:
		xpred = np.dot(A,x) 
	else:
		xpred = np.dot(A,x) + np.dot(B,u)

	SigPred = np.dot(A,np.dot(Sig,A.T)) + Q

	e = y - np.dot(H,xpred)
	Sinv = np.linalg.inv(np.dot(H,np.dot(SigPred,H.T)) + R)
	K = np.dot(SigPred,np.dot(H.T,Sinv))

	xnew = xpred + np.dot(K,e)
	SigNew = np.dot((np.eye(np.shape(A)[0]) - np.dot(K,H)),SigPred)

	return xnew.T,SigNew
	
def Kalman_smoother_update(A,Q,B,u,xs_t,Sigs_t,xfilt,Sigfilt,Sigfilt_t):

	if B is None:
		xpred = np.dot(A,xfilt)
	else:
		xpred = np.dot(A,xfilt) + np.dot(B,u)

	SigPred = np.dot(A,np.dot(Sigfilt,A.T)) + Q
	J = np.dot(Sigfilt,np.dot(A.T,np.linalg.inv(SigPred)))
	xs = xfilt + np.dot(J,(xs_t - xpred))
	Sigs = Sigfilt + np.dot(J,np.dot((Sigs_t - SigPred),J.T))

	return xs.T, Sigs


def Kalman_filter(y,A,H,Q,R,x0,Sig0,B=None,u=None):

	obs_size,T = np.shape(y)
	state_size = np.shape(A)[0]

	x = np.zeros((state_size,T))
	Sig = np.zeros((state_size,state_size,T))

	[x[:,0],Sig[:,:,0]] = Kalman_update(A,H,Q,R,y[:,0].reshape(len(y),1),x0,Sig0,B,u)
	for t in range(1,T):
		prevx = x[:,t-1].reshape(state_size,1)
		prevSig = Sig[:,:,t-1]
		[x[:,t],Sig[:,:,t]] = Kalman_update(A,H,Q,R,y[:,t].reshape(len(y),1),prevx,prevSig,B,u)

	return x,Sig

def Kalman_smoother(y,A,H,Q,R,x0,Sig0,B=None,u=None):

	obs_size,T = np.shape(y)
	state_size = np.shape(A)[0]

	xs = np.zeros((state_size,T))
	Sigs = np.zeros((state_size,state_size,T))

	[xfilt,Sigfilt] = Kalman_filter(y,A,H,Q,R,x0,Sig0,B,u)

	xs[:,T-1] = xfilt[:,T-1]
	Sigs[:,:,T-1] = Sigfilt[:,:,T-1]

	for t in range(T-2,-1,-1):
		[xs[:,t],Sigs[:,:,t]] = Kalman_smoother_update(A,Q,B,u,xs[:,t+1].reshape(len(xs),1),Sigs[:,:,t+1],xfilt[:,t].reshape(len(xfilt),1),Sigfilt[:,:,t],Sigfilt[:,:,t+1])

	return xs,Sigs

def lds_sample(A,H,Q,R,state0,T):
	# x(t+1) = Ax(t) +  state_noise(t), state_noise ~ N(O,Q), x(0) = state0
	# y(t) = Hx(t) + obs_noise(t), obs_noise~N(O,R)

	state_noise_samples = np.random.multivariate_normal(np.zeros((len(Q))),Q,T).T
	obs_noise_samples = np.random.multivariate_normal(np.zeros((len(R))),R,T).T

	x = np.zeros((np.shape(H)[1],T))
	y = np.zeros((np.shape(H)[0],T))

	x[:,0] = state0.T
	y[:,0] = np.dot(H,x[:,0]) + obs_noise_samples[:,0]

	for t in range(1,T):
		x[:,t] = np.dot(A,x[:,t-1]) + state_noise_samples[:,t]
		y[:,t] = np.dot(H,x[:,t-1]) + obs_noise_samples[:,t]

	return [x,y]

def Kalman_demo():
	state_size = 4
	observation_size = 2
	A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
	H = np.array([[1,0,0,0],[0,1,0,0]],dtype=float)

	Q = 0.1*np.eye((state_size))
	R = np.eye(observation_size,dtype=float)

	x0 = np.array([[10],[10],[1],[0]],dtype=float)
	Sig0 = 10. * np.eye(state_size)

	np.random.seed(3)
	T = 15
	
	[x,y] = lds_sample(A,H,Q,R,x0,T)

	[xfilt,Sigfilt] = Kalman_filter(y,A,H,Q,R,x0,Sig0)
	[xsmooth,Sigsmooth] = Kalman_smoother(y,A,H,Q,R,x0,Sig0)

	dfilt = x[[0,1],:] - xfilt[[0,1],:]
	mse_filt = np.sqrt(np.sum(dfilt**2))

	dsmooth = x[[0,1],:] - xsmooth[[0,1],:]
	mse_smooth = np.sqrt(np.sum(dsmooth**2))

	plot_track(x,y,xfilt,Sigfilt)
	plot_track(x,y,xsmooth,Sigsmooth)
	
def plot_track(x,y,Kx,Sig):
	fig = pl.figure()
	ax = fig.add_subplot(111, aspect='equal')
	pl.plot(x[0,:],x[1,:],'ks-')
	pl.plot(y[0,:],y[1,:],'k*')
	pl.plot(Kx[0,:],Kx[1,:],'kx:')
	pl.legend(('True','Observed','Filtered'))

	obs_size,T = np.shape(y)

	from matplotlib.patches import Ellipse
	# Axes of ellipse are eigenvectors of covariance matrix, lengths are square roots of eigenvalues
	ellsize = np.zeros((obs_size,T))
	ellangle = np.zeros((T))
	for t in range(T):
		[evals,evecs] = np.linalg.eig(Sig[:2,:2,t])
		ellsize[:,t] = np.sqrt(evals)	
		ellangle[t] = np.angle(evecs[0,0]+0.j*evecs[0,1])
		
	ells = [Ellipse(xy=[Kx[0,t],Kx[1,t]] ,width=ellsize[0,t],height=ellsize[1,t], angle=ellangle[t]) for t in range(T)]
	for e in ells:
		ax.add_artist(e)
		e.set_alpha(0.1)
		e.set_facecolor([0.7,0.7,0.7])
	pl.xlabel('x')
	pl.ylabel('y')


def Kalman_demo1d():

	x0 = np.array([-0.37727])
	Sig0 = 0.1*np.ones((1))
	T = 50	

        y = np.random.normal(x0,Sig0,(1,T))

    	A = np.eye(1)
    	H = np.eye(1)
    	Q = np.eye(1)*1e-5
    	R = np.eye(1)*0.01
    
    	xfilt = np.zeros((1,T),dtype=float)
    	Sigfilt = np.zeros((1,T),dtype=float)

	[xfilt,Sigfilt] = Kalman_filter(y,A,H,Q,R,x0,Sig0)
	xfilt = np.squeeze(xfilt)
	Sigfilt = np.squeeze(Sigfilt)

    	pl.figure()
	time = np.arange(T)
    	pl.plot(time,y[0,:],'ko',ms=6)
    	pl.plot(time,xfilt,'k-',lw=3)
    	pl.plot(time,xfilt+20*Sigfilt,'k--',lw=2)
    	pl.plot(time,xfilt-20*Sigfilt,'k--',lw=2)
    	pl.legend(['Noisy Datapoints','Kalman estimate','20*Covariance'])
    	pl.xlabel('Time')
