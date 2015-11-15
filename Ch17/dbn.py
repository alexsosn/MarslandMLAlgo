
# Code from Chapter 17 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import rbm
reload(rbm)

class dbn:

	def __init__(self,nvisible,nhidden=[100,100,50],nlabels=0,eta=0.1,momentum=0.,nCDsteps=5,nepochs=1000,useprobstate=0):
		self.layers = [rbm.rbm(nvisible,nhidden[0],nlabels=None,eta=eta,momentum=momentum,nCDsteps=nCDsteps,nepochs=nepochs)]
		self.nRBMs = len(nhidden)-1
		for i in range(1,self.nRBMs):
			self.layers.append(rbm.rbm(nhidden[i-1],nhidden[i],nlabels=None,eta=eta,momentum=momentum,nCDsteps=nCDsteps,nepochs=nepochs))
		self.layers.append(rbm.rbm(nhidden[self.nRBMs-1],nhidden[self.nRBMs],nlabels=nlabels,eta=eta,momentum=momentum,nCDsteps=nCDsteps,nepochs=nepochs))
		self.eta = eta
		self.momentum = momentum
		self.nCDsteps = nCDsteps
		self.nepochs = nepochs

		for i in range(self.nRBMs+1):
			print self.layers[i].nvisible, self.layers[i].nhidden, self.layers[i].nlabels

	def classify(self,inputs,labels):
		nextin = inputs
		for i in range(self.nRBMs):
			nextin,ph = self.compute_hidden(nextin,i)

		h, ph = self.layers[self.nRBMs].compute_hidden(nextin,1./(labels.max()+1)*np.ones(np.shape(labels)))
		v, pv, l = self.layers[self.nRBMs].compute_visible(h)
		
		print l.argmax(axis=1)
		print labels.argmax(axis=1)
		
		print 'Errors:', (l.argmax(axis=1) != labels.argmax(axis=1)).sum()

	def classify_after_greedy(self,inputs,labels):
		nextin = inputs
		for i in range(self.nRBMs):
			nextin,ph = self.layers[i].compute_hidden(nextin)

		h, ph = self.layers[self.nRBMs].compute_hidden(nextin,1./(labels.max()+1)*np.ones(np.shape(labels)))
		v, pv, l = self.layers[self.nRBMs].compute_visible(h)
		
		print l.argmax(axis=1)
		print labels.argmax(axis=1)
		
		print 'Errors:', (l.argmax(axis=1) != labels.argmax(axis=1)).sum()

	def bottom_up(self,inputs,labels):
		nextin = inputs
		for i in range(self.nRBMs):
			nextin,ph = self.compute_hidden(nextin,i)

		h, ph = self.layers[self.nRBMs].compute_hidden(nextin,np.zeros(np.shape(labels)))
		v, pv, l = self.layers[self.nRBMs].compute_visible(h)
		return h, ph, l

	def top_down(self,top_hidden):
		nextin = top_hidden
		nextin, pnextin,l = self.layers[self.nRBMs].compute_visible(nextin)
		for i in range(self.nRBMs-1,-1,-1):
			nextin, pnextin = self.compute_visible(nextin,i)
		return nextin, pnextin

	def greedy(self,inputs,labels):
		probinputs = inputs
		for i in range(self.nRBMs):
			self.layers[i].contrastive_divergence(inputs)
			inputs,probinputs = self.layers[i].compute_hidden(probinputs)
			# Can initialise the weights with the previous ones
			#if (self.layers[i].nhidden == self.layers[i+1].nhidden) and (self.layers[i].nvisible == self.layers[i+1].nvisible):
				#self.layers[i+1].weights = self.layers[i].weights
		self.layers[self.nRBMs].contrastive_divergence(probinputs,labels)
		self.sample, self.probsample = self.layers[self.nRBMs].compute_hidden(probinputs,labels)

	def compute_output(self,input):
		sample,probsample = self.layers[self.nRBMs].compute_visible(self.probsample)
		for i in range(self.nRBMs-1,-1,-1):
			sample,probsample = self.layers[i].compute_visible(probsample)
		
	def compute_visible(self,hidden,i):
		sumin = self.layers[i].visiblebias + np.dot(hidden,self.layers[i].gen.T)
		self.visibleprob = 1./(1. + np.exp(-sumin))
		self.visibleact = (self.visibleprob>np.random.rand(np.shape(self.visibleprob)[0],self.layers[i].nvisible)).astype('float')
		return [self.visibleact,self.visibleprob]

	def compute_hidden(self,visible,i):
		sumin = self.layers[i].hiddenbias + np.dot(visible,self.layers[i].rec)
		self.hiddenprob = 1./(1. + np.exp(-sumin))
		self.hiddenact = (self.hiddenprob>np.random.rand(np.shape(self.hiddenprob)[0],self.layers[i].nhidden)).astype('float')
		return [self.hiddenact,self.hiddenprob]

	def updown(self,inputs,labels):

		N = np.shape(inputs)[0]

		# Need to untie the weights
		for i in range(self.nRBMs):
			self.layers[i].rec = self.layers[i].weights.copy()
			self.layers[i].gen = self.layers[i].weights.copy()

		old_error = np.iinfo('i').max
		error = old_error
		self.eta = 0
		for epoch in range(11):
			# Wake phase
	
			v = inputs
			for i in range(self.nRBMs):
				vold = v
				h,ph = self.compute_hidden(v,i)
				v,pv = self.compute_visible(h,i)
	
				# Train generative weights
				self.layers[i].gen += self.eta * np.dot((vold-pv).T,h)/N
				self.layers[i].visiblebias += self.eta * np.mean((vold-pv),axis=0)

				v=h

			# Train the labelled RBM as normal
			self.layers[self.nRBMs].contrastive_divergence(v,labels,silent=True)

			# Sample the labelled RBM
			for i in range(self.nCDsteps):
				h,ph = self.layers[self.nRBMs].compute_hidden(v,labels)
				v,pv,pl = self.layers[self.nRBMs].compute_visible(h)

			# Compute the class error
			#print (pl.argmax(axis=1) != labels.argmax(axis=1)).sum()	

			# Sleep phase
	
			# Initialise with the last sample from the labelled RBM
			h = v
			for i in range(self.nRBMs-1,-1,-1):
				hold = h
				v, pv = self.compute_visible(h,i)
				h, ph = self.compute_hidden(v,i)
			
				# Train recognition weights
				self.layers[i].rec += self.eta * np.dot(v.T,(hold-ph))/N
				self.layers[i].hiddenbias += self.eta * np.mean((hold-ph),axis=0)

				h=v
		
			old_error2 = old_error
			old_error = error
			error = np.sum((inputs - v)**2)/N
			if (epoch%2==0): 
				print epoch, error
			if (old_error2 - old_error)<0.01 and (old_error-error)<0.01:
				break

def test_dbn():

	import dbn
	#dbn.classify()
	rb = dbn.dbn(4,[4,4,2],nlabels=2,eta=0.3,momentum=0.5,nCDsteps=1,nepochs=1001)
    	N=2500
    	v = np.zeros((2*N,4))
    	l = np.zeros((2*N,2))
    	for n in range(N):
        	r = np.random.rand()
        	if r>0.666:
            		v[n,:] = [0,1,0,0]
            		l[n,:] = [1,0]
        	elif r>0.333:
            		v[n,:] = [1,0,0,0]
            		l[n,:] = [1,0]
    	for n in range(N):
        	r = np.random.rand()
        	if r>0.666:
            		v[N+n,:] = [0,0,0,1]
            		l[N+n,:] = [0,1]
        	elif r>0.333:
            		v[N+n,:] = [0,0,1,0]
            		l[N+n,:] = [0,1]


        rb.greedy(v,l) #, delta=delta, momentum=mom)
	rb.classify_after_greedy(v,l)
	rb.updown(v,l)
	rb.classify(v,l)

def test_dbn_mnist():
	import dbn
	import cPickle, gzip

	# Load the dataset
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	indices = np.arange(np.shape(train_set[0])[0])
	np.random.shuffle(indices)
	#indices = indices[:200]
	t = train_set[0][indices,:]
	l = train_set[1][indices]
	labels = np.zeros(((np.shape(t)[0]),10))
	for i in range(20*10):
		labels[i,l[i]] = 1
	rb = dbn.dbn(28*28,[100,100,50],nlabels=10,eta=0.3,momentum=0.4,nCDsteps=3,nepochs=1000)
        rb.greedy(t,labels) #, delta=delta, momentum=mom)
	rb.classify_after_greedy(t,labels)
	rb.updown(t,labels)
	rb.classify(t,labels)

	return rb
	
def test_dbn_digs():
	import dbn
	import scipy as scipy
	import scipy.io as sio
	tmp = sio.loadmat('binaryalphadigs.mat')
	NTRAIN = 39
	CLASSES = [10, 12, 28] # A, C, S
	NCLASSES = len(CLASSES)
	I, L = 20*16, 3
	N = NTRAIN*NCLASSES

	# organize data 
	data = np.zeros((NCLASSES, NTRAIN, I))
	labels = np.zeros((NCLASSES, NTRAIN, L))
	for k in range(L): 
    		for m in range(NTRAIN):
        		data[k,m,:] = (tmp['dat'][CLASSES[k],m].ravel()).astype('d')
        		labels[k,m,k] = 1.
    	
	# prepare observations, labels
	perm = np.arange(N)
	np.random.shuffle(perm)
	v = data.reshape(N, I)[perm,:]
	l = labels.reshape(N, L)[perm,:]

	rb = dbn.dbn(20*16,[100,100,100],nlabels=3,eta=0.3,momentum=0.4,nCDsteps=3,nepochs=401)
	rb.greedy(v,l)	
	rb.classify_after_greedy(v,l)
	rb.updown(v,l)
	rb.classify(v,l)

	import pylab as pl
	pl.figure(), pl.imshow(v[5,:].reshape(20,16))
	
	toph, topph, labs = rb.bottom_up(v,l)
	obs, pobs = rb.top_down(topph)
	
	pl.figure(), pl.imshow(pobs[5,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest')
	return v, l, obs, pobs

def learn_letters():
        import scipy.io as sio

        nperclass = 39
        classes = [10, 11, 28]
        #classes = [10, 13, 28] # A, C, S
        nclasses = len(classes)

	# Read in the data and prepare it
        data = sio.loadmat('binaryalphadigs.mat')
        inputs = np.ones((nclasses, nperclass, 20*16))
        labels = np.zeros((nclasses, nperclass, nclasses))
        for k in range(nclasses):
                for m in range(nperclass):
                        inputs[k,m,:] = (data['dat'][classes[k],m].ravel()).astype('float')
                        labels[k,m,k] = 1.

	nexamples = 20
        v = inputs[:,:nexamples,:].reshape(nclasses*nexamples, 20*16)
        l = labels[:,:nexamples,:].reshape(nclasses*nexamples, nclasses)

        import pylab as pl

	# This shows a set of examples from the training set
	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(v[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')
	

	import dbn
	rb = dbn.dbn(20*16,[100,100,100],nlabels=nclasses,eta=0.3,momentum=0.4,nCDsteps=3,nepochs=600)
	rb.greedy(v,l)	
	rb.classify_after_greedy(v,l)
	rb.updown(v,l)
	rb.classify(v,l)

	toph, topph, labs = rb.bottom_up(v,l)
	obs, pobs = rb.top_down(topph)
	
	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(pobs[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(obs[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	newv = inputs[:,nexamples:,:].reshape(nclasses*(39-nexamples),20*16)
	newl = labels[:,nexamples:,:].reshape(nclasses*(39-nexamples),nclasses)
	rb.classify(newv,newl)

	toph, topph, labs = rb.bottom_up(newv,newl)
	obs, pobs = rb.top_down(topph)

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(newv[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(pobs[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(obs[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	vis = np.random.randn(np.shape(v)[0],np.shape(v)[1])*0.05

	for i in range(1000):
		toph, topph, labs = rb.bottom_up(vis,l)
		vis, pvis = rb.top_down(topph)

	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(vis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

#learn_letters()
#test_dbn_digs()

#test_dbn()
#test_dbn_mnist()
#timeit.timeit("h = (probs > np.random.rand(probs.shape[0],probs.shape[1])).astype('int')",setup="import numpy as np; probs = np.random.rand(1000,100)",number=100)

#timeit.timeit("h= np.where(probs>np.random.rand(probs.shape[0],probs.shape[1]),1,0)",setup="import numpy as np; probs = np.random.rand(1000,100)",number=100)

