
# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import scipy.optimize as so

class mlp_cg:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def mlperror(self,weights,inputs,targets):
	split = (self.nin+1)*self.nhidden
	self.weights1 = np.reshape(weights[:split],(self.nin+1,self.nhidden))
	self.weights2 = np.reshape(weights[split:],(self.nhidden+1,self.nout))
	outputs = self.mlpshortfwd(inputs)

	# Compute the error
        # Different types of output neurons
        if self.outtype == 'linear':
        	error = 0.5*np.sum((outputs-targets)**2)
        elif self.outtype == 'logistic':
		# Non-zero checks
		maxval = -np.log(np.finfo(np.float64).eps)
		minval = -np.log(1./np.finfo(np.float64).tiny - 1.)
		outputs = np.where(outputs<maxval,outputs,maxval)
		outputs = np.where(outputs>minval,outputs,minval)
		outputs = 1./(1. + np.exp(-outputs))	
    		error = - np.sum(targets*np.log(outputs) + (1 - targets)*np.log(1 - outputs))
        elif self.outtype == 'softmax':
		nout = np.shape(outputs)[1]
		maxval = np.log(np.finfo(np.float64).max) - np.log(nout)
		minval = np.log(np.finfo(np.float32).tiny)
		outputs = np.where(outputs<maxval,outputs,maxval)
		outputs = np.where(outputs>minval,outputs,minval)
            	normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            	y =  np.transpose(np.transpose(np.exp(outputs))/normalisers)
		y[y<np.finfo(np.float64).tiny] = np.finfo(np.float32).tiny
    		error = - np.sum(targets*np.log(y));

        else:
            	print "error"

	return error

    def mlpgrad(self,weights,inputs,targets):
	split = (self.nin+1)*self.nhidden
	self.weights1 = np.reshape(weights[:split],(self.nin+1,self.nhidden))
	self.weights2 = np.reshape(weights[split:],(self.nhidden+1,self.nout))
	outputs = self.mlpfwd(inputs)

	delta_out = outputs-targets
	grad_weights2 = np.dot(self.hidden.T,delta_out)

	delta_hid = np.dot(delta_out,self.weights2[1:,:].T)
	delta_hid *= (1. - self.hidden[:,1:]*self.hidden[:,1:])
	grad_weights1 = np.dot(inputs.T,delta_hid)
	
	return np.concatenate((grad_weights1.flatten(),grad_weights2.flatten()))

    def mlptrain(self,inputs,targets,niterations=100):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)

	# Put all the weights into a single row vector
	w = np.concatenate((self.weights1.flatten(),self.weights2.flatten()))

	#out = so.fmin_cg(self.mlperror, w, fprime=None, args=(inputs,targets), gtol=1e-05, maxiter=5000, full_output=True, disp=1)
	out = so.fmin_cg(self.mlperror, w, fprime=self.mlpgrad, args=(inputs,targets), gtol=1e-05, maxiter=10000, full_output=True, disp=1)

	wopt = out[0]

	# Put the updated weights back into the matrices
	split = (self.nin+1)*self.nhidden
	self.weights1 = np.reshape(wopt[:split],(self.nin+1,self.nhidden))
	self.weights2 = np.reshape(wopt[split:],(self.nhidden+1,self.nout))
    
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            	return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            	normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            	return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            	print "error"

    def mlpshortfwd(self,inputs):
        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        return np.dot(self.hidden,self.weights2);
	
    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print "Confusion matrix is:"
        print cm
        print "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100
