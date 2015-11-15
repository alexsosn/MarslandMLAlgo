
# Code from Chapter 18 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import pylab as pl
import numpy as np
import scipy.optimize as so


def kernel(data1,data2,theta,wantderiv=True,measnoise=1.):
	# Uses exp(theta) to ensure positive hyperparams
	theta = np.squeeze(theta)
	theta = np.exp(theta)
	# Squared exponential
	if np.ndim(data1) == 1:
		d1 = np.shape(data1)[0]
		n = 1
	else:
		(d1,n) = np.shape(data1)

	d2 = np.shape(data2)[0]
	sumxy = np.zeros((d1,d2))
	for d in range(n):
		D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
		D2 = [data2[:,d]] * np.ones((d1,d2))
		sumxy += (D1-D2)**2*theta[d+1]

	k = theta[0] * np.exp(-0.5*sumxy) 
	#k = theta[0]**2 * np.exp(-sumxy/(2.0*theta[1]**2)) 

	#print k
	#print measnoise*theta[2]**2*np.eye(d1,d2)
	if wantderiv:
		K = np.zeros((d1,d2,len(theta)+1))
		K[:,:,0] = k + measnoise*theta[2]*np.eye(d1,d2)
		K[:,:,1] = k 
		K[:,:,2] = -0.5*k*sumxy
		K[:,:,3] = theta[2]*np.eye(d1,d2)
		return K
	else:	
		return k + measnoise*theta[2]*np.eye(d1,d2)

def kernel2(data1,data2,theta,wantderiv=True,measnoise=1.):
	theta = np.squeeze(theta)
	theta2 = 0.3
	# Squared exponential
	(d1,n) = np.shape(data1)
	d2 = np.shape(data2)[0]
	sumxy = np.zeros((d1,d2))
	for d in range(n):
		D1 = np.transpose([data1[:,d]]) * np.ones((d1,d2))
		D2 = [data2[:,d]] * np.ones((d1,d2))
		sumxy += (D1-D2)**2

	k = theta[0]**2 * np.exp(-sumxy/(2.0*theta[1]**2)) 

	if wantderiv:
		K = np.zeros((d1,d2,len(theta)+1))
		K[:,:,0] = k + measnoise*theta[2]**2*np.eye(d1,d2)
		K[:,:,1] = 2.0 *k /theta[0]
		K[:,:,2] = k*sumxy/(theta[1]**3)
		K[:,:,3] = 2.0*theta[2]*np.eye(d1,d2)
		return K
	else:	
		return k + measnoise*theta[2]**2*np.eye(d1,d2)
		
def NRiteration(data,targets,theta):
	K = kernel(data,data,theta,wantderiv=False)
	n = np.shape(targets)[0]
	f = np.zeros((n,1))
	tol = 0.1
	phif = 1e100
	scale = 1.
	count = 0
	while True:
		count += 1
		s = np.where(f<0,f,0)
		W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
		sqrtW = np.sqrt(W)
		L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K,sqrtW)))
		p = np.exp(s)/(np.exp(s) + np.exp(s-f))
		b = np.dot(W,f) + 0.5*(targets+1) - p
		a = scale*(b - np.dot(sqrtW,np.linalg.solve(L.transpose(),np.linalg.solve(L,np.dot(sqrtW,np.dot(K,b))))))
		f = np.dot(K,a)
		oldphif = phif
		phif = np.log(p) -0.5*np.dot(f.transpose(),np.dot(np.linalg.inv(K),f)) - 0.5*np.sum(np.log(np.diag(L))) - np.shape(data)[0] /2. * np.log(2*np.pi)
		#print "loop",np.sum((oldphif-phif)**2)
		if (np.sum((oldphif-phif)**2) < tol):
			break
		elif (count > 100):
			count = 0
			scale = scale/2.
			
	s = -targets*f
	ps = np.where(s>0,s,0)
	logq = -0.5*np.dot(a.transpose(),f) -np.sum(np.log(ps+np.log(np.exp(-ps) + np.exp(s-ps)))) - np.trace(np.log(L))
	return (f,logq,a)

def predict(xstar,data,targets,theta):
	K = kernel(data,data,theta,wantderiv=False)
	n = np.shape(targets)[0]
	kstar = kernel(data,xstar,theta,wantderiv=False,measnoise=0)
	(f,logq,a) = NRiteration(data,targets,theta)
	s = np.where(f<0,f,0)
	W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
	sqrtW = np.sqrt(W)
	L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K,sqrtW)))
	p = np.exp(s)/(np.exp(s) + np.exp(s-f))
	fstar = np.dot(kstar.transpose(), (targets+1)*0.5 - p)
	v = np.linalg.solve(L,np.dot(sqrtW,kstar))	
	V = kernel(xstar,xstar,theta,wantderiv=False,measnoise=0)-np.dot(v.transpose(),v) 
	return (fstar,V)

def logPosterior(theta,args):
	data,targets = args
	(f,logq,a) = NRiteration(data,targets,theta)
	return -logq

def gradLogPosterior(theta,args):
	data,targets = args
	theta = np.squeeze(theta)
	n = np.shape(targets)[0]
	K = kernel(data,data,theta,wantderiv=True)
	(f,logq,a) = NRiteration(data,targets,theta)
	s = np.where(f<0,f,0)
	W = np.diag(np.squeeze(np.exp(2*s - f) / ((np.exp(s) + np.exp(s-f))**2)))
	sqrtW = np.sqrt(W)
	L = np.linalg.cholesky(np.eye(n) + np.dot(sqrtW,np.dot(K[:,:,0],sqrtW)))

	R = np.dot(sqrtW,np.linalg.solve(L.transpose(),np.linalg.solve(L,sqrtW)))
	C = np.linalg.solve(L,np.dot(sqrtW,K[:,:,0]))
	p = np.exp(s)/(np.exp(s) + np.exp(s-f))
	hess = -np.exp(2*s - f) / (np.exp(s) + np.exp(s-f))**2
	s2 = -0.5*np.dot(np.diag(np.diag(K[:,:,0]) - np.diag(np.dot(C.transpose(),C))) , 2*hess*(0.5-p))

	gradZ = np.zeros(len(theta))
	for d in range(1,len(theta)+1):
		s1 = 0.5*(np.dot(a.transpose(),np.dot(K[:,:,d],a))) - 0.5*np.trace(np.dot(R,K[:,:,d]))	
		b = np.dot(K[:,:,d],(targets+1)*0.5-p)
		p = np.exp(s)/(np.exp(s) + np.exp(s-f))
		s3 = b - np.dot(K[:,:,0],np.dot(R,b))
		gradZ[d-1] = s1 + np.dot(s2.transpose(),s3)

	return -gradZ

def test():
	
	pl.ion()

	data = np.array([[-2.1, -2.0, -1.9, -0.1, 0., 0.1, 1.9, 2.0, 2.1 ]]).transpose()
	labels = np.array([[-1., -1., -1., 1., 1., 1., -1., -1., -1. ]]).transpose()
	
	theta =np.zeros((3,1))
	theta[0] = 1.0 #np.random.rand()*3
	theta[1] = 0.7 #np.random.rand()*3
	theta[2] = 0.3

	args = (data,labels)

	print theta, logPosterior(theta,args)

    	result = so.fmin_cg(logPosterior, theta, fprime=gradLogPosterior, args=[args], gtol=1e-4,maxiter=100,disp=1)
	newTheta = result

	print "======="
	print newTheta, logPosterior(newTheta,args)
	print "======="

	test = np.array([[-2.2, -2.05, -1.8, -0.2, 0.05, 0.15, 1.8, 2.05, 2.01 ]]).transpose()
	tlabels = np.array([[-1., -1., -1., 1., 1., 1., -1., -1., -1. ]]).transpose()

	# Compute the mean and covariance of the data
	xstar = np.reshape(np.linspace(-5,5,100),(100,1))
	K = kernel(data,data,newTheta,wantderiv=False)
	kstar = [kernel(data,xs*np.ones((1,1)),theta,wantderiv=False,measnoise=False) for xs in xstar]
	kstar = np.squeeze(kstar)
	kstarstar = [kernel(xs*np.ones((1,1)),xs*np.ones((1,1)),theta,wantderiv=False,measnoise=False) for xs in xstar]
	kstarstar = np.squeeze(kstarstar)

	invk = np.linalg.inv(K)
	mean = np.dot(kstar,np.dot(invk,labels))
	var = kstarstar - np.diag(np.dot(kstar,np.dot(invk,kstar.transpose())))
	var = np.reshape(var,(100,1))
	pl.plot(xstar,mean,'-k')
	pl.fill_between(np.squeeze(xstar),np.squeeze(mean-2*np.sqrt(var)),np.squeeze(mean+2*np.sqrt(var)),color='0.75')
	pl.xlabel('x')
	pl.ylabel('Latent f(x)')

	#xstar = np.arange(1,1e3 + 1,1)/1e3 * 2.1 - 1.8
	pred = np.squeeze(np.array([predict(np.reshape(i,(1,1)),data,labels,newTheta) for i in test]))
	output = np.reshape(np.where(pred[:,0]<0,-1,1),(9,1))
	print np.sum(np.abs(output-tlabels))
	print pred

	#pl.figure()
	which = np.where(labels==1)
	pl.plot(data[which],labels[which],'ro')
	which = np.where(labels==-1)
	pl.plot(data[which],labels[which],'gx')

	which = np.where((tlabels==1) & (output==1))
	pl.plot(test[which],tlabels[which],'r^')
	which = np.where((tlabels==-1) & (output==-1))
	pl.plot(test[which],tlabels[which],'gv')
	
	which = np.where((tlabels==1) & (output==-1))
	pl.plot(test[which],tlabels[which],'rs')
	which = np.where((tlabels==-1) & (output==1))
	pl.plot(test[which],tlabels[which],'gs')

	pl.figure()
	pred2 = np.squeeze(np.array([predict(np.reshape(i,(1,1)),data,labels,newTheta) for i in xstar]))
	pl.plot(xstar,pred2[:,0],'k-')
	pl.fill_between(np.squeeze(xstar),np.squeeze(pred2[:,0]-2*np.sqrt(pred2[:,1])),np.squeeze(pred2[:,0]+2*np.sqrt(pred2[:,1])),color='0.75')
	pl.xlabel('x')
	pl.ylabel('$\sigma(f(x))$')

def modified_XOR(sdev=0.3):

	m = 100
	data = sdev*np.random.randn(m,2)
	data[m/2:,0] += 1.
	data[m/4:m/2,1] += 1.
	data[3*m/4:,1] += 1.
	labels = -np.ones((m,1))
	labels[:m/4,0] = 1.
	labels[3*m/4:,0] = 1.
	#labels = (np.where(X[:,0]*X[:,1]>=0,1,-1)*np.ones((1,np.shape(X)[0]))).T
	
	Y = sdev*np.random.randn(m,2)
	Y[m/2:,0] += 1.
	Y[m/4:m/2,1] += 1.
	Y[3*m/4:m,1] += 1.
	test = -np.ones((m,1))
	test[:m/4,0] = 1.
	test[3*m/4:,0] = 1.

	theta =np.zeros((3,1))
	theta[0] = 1.0 #np.random.rand()*3
	theta[1] = 0.7 #np.random.rand()*3
	theta[2] = 0.

	args = (data,labels)

	print theta, logPosterior(theta,args)

    	result = so.fmin_cg(logPosterior, theta, fprime=gradLogPosterior, args=[args], gtol=1e-4,maxiter=20,disp=1)
    	#result = so.fmin_cg(logPosterior, theta, fprime=gradLogPosterior, args=[args], gtol=1e-4,maxiter=10,disp=1)
	newTheta = result

	print "======="
	print newTheta, logPosterior(newTheta,args)
	print "======="

	#xstar = np.reshape(np.linspace(-5,5,100),(100,1))
	#K = kernel(data,data,newTheta,wantderiv=False)
	#kstar = [kernel(data,xs*np.ones((1,1)),theta,wantderiv=False,measnoise=False) for xs in xstar]
	#kstar = np.squeeze(kstar)
	#kstarstar = [kernel(xs*np.ones((2,1)),xs*np.ones((2,1)),theta,wantderiv=False,measnoise=False) for xs in xstar]
	#kstarstar = np.squeeze(kstarstar)

	#invk = np.linalg.inv(K)
	#mean = np.dot(kstar,np.dot(invk,labels))
	#var = kstarstar - np.diag(np.dot(kstar,np.dot(invk,kstar.transpose())))
	#var = np.reshape(var,(100,1))
	#pl.plot(xstar,mean,'-k')
	#pl.fill_between(np.squeeze(xstar),np.squeeze(mean-2*np.sqrt(var)),np.squeeze(mean+2*np.sqrt(var)),color='0.75')
	#pl.xlabel('x')
	#pl.ylabel('Latent f(x)')

	#xstar = np.arange(1,1e3 + 1,1)/1e3 * 2.1 - 1.8
	pred = np.squeeze(np.array([predict(np.reshape(i,(1,2)),data,labels,newTheta) for i in Y]))
	output = np.reshape(np.where(pred[:,0]<0,-1,1),(m,1))
	print np.sum(np.abs(output-test))
	#print pred

	err1 = np.where((output==1.) & (test==-1.))[0]
	err2 = np.where((output==-1.) & (test==1.))[0]
	print "Class 1 errors ",len(err1)," from ",len(test[test==1])
	print "Class 2 errors ",len(err2)," from ",len(test[test==-1])
	print "Test accuracy ",1. -(float(len(err1)+len(err2)))/ (len(test[test==1]) + len(test[test==-1]))

	pl.figure()
	l1 =  np.where(labels==1)[0]
	l2 =  np.where(labels==-1)[0]
	pl.plot(data[l1,0],data[l1,1],'ko')
	pl.plot(data[l2,0],data[l2,1],'wo')
	#l1 =  np.where(test==1)[0]
	#l2 =  np.where(test==-1)[0]
	#pl.plot(Y[l1,0],Y[l1,1],'ks')
	#pl.plot(Y[l2,0],Y[l2,1],'ws')
	pl.axis('tight')
	pl.axis('off')

	xmin = np.min(data[:,0])
	xmax = np.max(data[:,0])
	x = np.arange(xmin, xmax, 0.1)
        y = np.arange(xmin, xmax, 0.1)

	predgrid = np.zeros((len(x),len(y)))
	for i in range(len(x)):
		for j in range(len(y)):
			d = np.array([[x[i],y[j]]])
			predgrid[i,j] = predict(d,data,labels,newTheta)[0]
		
	#pgrid = np.where(predgrid<0,-1,1)
	#print predgrid
	xx,yy = np.meshgrid(x,y)
	pl.contour(xx,yy,predgrid,1)

def test2():
	
	pl.ion()

	data = np.array([[-2.1, -2.0, -1.9, -0.1, 0., 0.1, 1.9, 2.0, 2.1 ]]).transpose()
	labels = np.array([[-1., -1., -1., 1., 1., 1., -1., -1., -1. ]]).transpose()
	
	theta =np.zeros((3,1))
	theta[0] = 1.0 #np.random.rand()*3
	theta[1] = 0.7 #np.random.rand()*3
	theta[2] = 0.3

	args = (data,labels)

	print theta, logPosterior(theta,args)

    	result = so.fmin_cg(logPosterior, theta, fprime=gradLogPosterior, args=[args], gtol=1e-4,maxiter=100,disp=1)
	newTheta = result

	print "======="
	print newTheta, logPosterior(newTheta,args)
	print "======="

	test = np.array([[-2.2, -2.05, -1.8, -0.2, 0.05, 0.15, 1.8, 2.05, 2.01 ]]).transpose()
	tlabels = np.array([[-1., -1., -1., 1., 1., 1., -1., -1., -1. ]]).transpose()

	# Compute the mean and covariance of the data
	xstar = np.reshape(np.linspace(-5,5,100),(100,1))
	K = kernel(data,data,newTheta,wantderiv=False)
	kstar = [kernel(data,xs*np.ones((1,1)),theta,wantderiv=False,measnoise=False) for xs in xstar]
	kstar = np.squeeze(kstar)
	kstarstar = [kernel(xs*np.ones((1,1)),xs*np.ones((1,1)),theta,wantderiv=False,measnoise=False) for xs in xstar]
	kstarstar = np.squeeze(kstarstar)

	invk = np.linalg.inv(K)
	mean = np.dot(kstar,np.dot(invk,labels))
	var = kstarstar - np.diag(np.dot(kstar,np.dot(invk,kstar.transpose())))
	var = np.reshape(var,(100,1))
	pl.plot(xstar,mean,'-k')
	pl.fill_between(np.squeeze(xstar),np.squeeze(mean-2*np.sqrt(var)),np.squeeze(mean+2*np.sqrt(var)),color='0.75')
	pl.xlabel('x')
	pl.ylabel('Latent f(x)')

	#xstar = np.arange(1,1e3 + 1,1)/1e3 * 2.1 - 1.8
	pred = np.squeeze(np.array([predict(np.reshape(i,(1,1)),data,labels,newTheta) for i in test]))
	output = np.reshape(np.where(pred[:,0]<0,-1,1),(9,1))
	print np.sum(np.abs(output-tlabels))
	print pred

	#pl.figure()
	which = np.where(labels==1)
	pl.plot(data[which],labels[which],'ro')
	which = np.where(labels==-1)
	pl.plot(data[which],labels[which],'gx')

	which = np.where((tlabels==1) & (output==1))
	pl.plot(test[which],tlabels[which],'r^')
	which = np.where((tlabels==-1) & (output==-1))
	pl.plot(test[which],tlabels[which],'gv')
	
	which = np.where((tlabels==1) & (output==-1))
	pl.plot(test[which],tlabels[which],'rs')
	which = np.where((tlabels==-1) & (output==1))
	pl.plot(test[which],tlabels[which],'gs')

	pl.figure()
	pred2 = np.squeeze(np.array([predict(np.reshape(i,(1,1)),data,labels,newTheta) for i in xstar]))
	pl.plot(xstar,pred2[:,0],'k-')
	pl.fill_between(np.squeeze(xstar),np.squeeze(pred2[:,0]-2*np.sqrt(pred2[:,1])),np.squeeze(pred2[:,0]+2*np.sqrt(pred2[:,1])),color='0.75')
	pl.xlabel('x')
	pl.ylabel('$\sigma(f(x))$')
#test()
#test2()
#modified_XOR(sdev=0.1)
#modified_XOR(sdev=0.3)
#modified_XOR(sdev=0.4)
