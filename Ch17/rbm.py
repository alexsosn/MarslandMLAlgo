
# Code from Chapter 17 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
#import pylab as pl

class rbm:

	def __init__(self,nvisible,nhidden,nlabels=None,inputs=[],eta=0.1,momentum=0.9,decay=0.,nCDsteps=5,nepochs=1000):
		self.nvisible = nvisible
		self.nhidden = nhidden
		self.weights = np.random.randn(nvisible,nhidden)*0.1
		self.visiblebias = np.random.randn(nvisible)*0.1
		self.hiddenbias = np.random.randn(nhidden)*0.1
		self.nlabels = nlabels
		if nlabels is not None:
			self.labelweights = np.random.randn(nlabels,nhidden)*0.1
			self.labelbias = np.random.randn(nlabels)*0.1

		# cf Hinton 8.1
		if np.shape(inputs)[0]>0:
			p = np.sum(inputs,axis=1)/(np.shape(inputs)[1])
			self.visiblebias = np.log(p/(1-p))

		self.eta = eta
		self.momentum = momentum
		self.decay = decay
		self.nCDsteps = nCDsteps
		self.nepochs = nepochs

	def compute_visible(self,hidden):
		# Compute p(v=1|h,W), p(l=1|h,W)
		sumin = self.visiblebias + np.dot(hidden,self.weights.T)

		# Compute visible node activations
		self.visibleprob = 1./(1. + np.exp(-sumin))
		self.visibleact = (self.visibleprob>np.random.rand(np.shape(self.visibleprob)[0],self.nvisible)).astype('float')

		# Compute label activations (softmax)
		if self.nlabels is not None:
			sumin = self.labelbias + np.dot(hidden,self.labelweights.T)
			summax = sumin.max(axis=1)
			summax = np.reshape(summax,summax.shape+(1,)).repeat(np.shape(sumin)[1],axis=-1)			
			sumin -= summax
			normalisers = np.exp(sumin).sum(axis=1)
			normalisers = np.reshape(normalisers,normalisers.shape+(1,)).repeat(np.shape(sumin)[1],axis=-1)			
			self.labelact = np.exp(sumin)/normalisers

		return [self.visibleprob,self.visibleact,self.labelact]

	def compute_hidden(self,visible,label=[]):
		# Compute p(h=1|v,W,{l})
		if self.nlabels is not None:
			sumin = self.hiddenbias + np.dot(visible,self.weights) + np.dot(label,self.labelweights)
		else:
			sumin = self.hiddenbias + np.dot(visible,self.weights)
		self.hiddenprob = 1./(1. + np.exp(-sumin))
		self.hiddenact = (self.hiddenprob>np.random.rand(np.shape(self.hiddenprob)[0],self.nhidden)).astype('float')
		return [self.hiddenact,self.hiddenprob]

	def classify(self,inputs,labels):
		h, ph = self.compute_hidden(inputs,labels)
		#h, ph = self.compute_hidden(inputs,1./(labels.max()+1)*np.ones(np.shape(labels)))
		v, pv, l = self.compute_visible(h)
		
		print l.argmax(axis=1)
		print labels.argmax(axis=1)
		
		print 'Errors:', (l.argmax(axis=1) != labels.argmax(axis=1)).sum()

    	def energy(self, visible, hidden, labels=[]):
		if self.nlabels is not None:
        		return -np.dot(visible, self.visiblebias) - np.dot(hidden, self.hiddenbias) - np.dot(labels,self.labelbias) - (np.dot(visible, self.weights)*hidden).sum(axis=1) - (np.dot(labels,self.labelweights)*hidden).sum(axis=1)
		else:
        		return -np.dot(visible, self.visiblebias) - np.dot(hidden, self.hiddenbias) - (np.dot(visible, self.weights)*hidden).sum(axis=1)

	def contrastive_divergence(self,inputs,labels=None,dw=None,dwl=None,dwvb=None,dwhb=None,dwlb=None,silent=False):

		# Clamp input into visible nodes
		visible = inputs
		#self.visibleact = inputs
		self.labelact = labels

	        dw = 0. if dw is None else dw
	        dwl = 0. if dwl is None else dwl
	        dwvb = 0. if dwvb is None else dwvb
	        dwhb = 0. if dwhb is None else dwhb
	        dwlb = 0. if dwlb is None else dwlb
		
		for epoch in range(self.nepochs):
			# Sample the hidden variables
			self.compute_hidden(visible,labels)
	
			# Compute <vh>_0
			positive = np.dot(inputs.T,self.hiddenact)
			#positive = np.dot(inputs.T,self.hiddenprob)
			positivevb = inputs.sum(axis=0)
			positivehb = self.hiddenprob.sum(axis=0)
			if self.nlabels is not None:
				positivelabels = np.dot(labels.T,self.hiddenact)
				#positivelabels = np.dot(labels.T,self.hiddenprob)
				positivelb = labels.sum(axis=0)

			# Do limited Gibbs sampling to sample from the hidden distribution
			for j in range(self.nCDsteps):	
				self.compute_visible(self.hiddenact)
				self.compute_hidden(self.visibleact,self.labelact)

			# Compute <vh>_n
			negative = np.dot(self.visibleact.T,self.hiddenact)
			#negative = np.dot(self.visibleact.T,self.hiddenprob)
			negativevb = self.visibleact.sum(axis=0)
			negativehb = self.hiddenprob.sum(axis=0)

			if self.nlabels is not None:
				negativelabels = np.dot(self.labelact.T,self.hiddenact)
				#negativelabels = np.dot(self.labelact.T,self.hiddenprob)
				negativelb = self.labelact.sum(axis=0)
				dwl = self.eta * ((positivelabels - negativelabels) / np.shape(inputs)[0] - self.decay*self.labelweights) + self.momentum*dwl
				self.labelweights += dwl
				dwlb = self.eta * (positivelb - negativelb) / np.shape(inputs)[0] + self.momentum*dwlb
				self.labelbias += dwlb

			# Learning rule (with momentum)
			dw = self.eta * ((positive - negative) / np.shape(inputs)[0] - self.decay*self.weights) + self.momentum*dw
			self.weights += dw

			dwvb = self.eta * (positivevb - negativevb) / np.shape(inputs)[0] + self.momentum*dwvb
			self.visiblebias += dwvb
			dwhb = self.eta * (positivehb - negativehb) / np.shape(inputs)[0] + self.momentum*dwhb
			self.hiddenbias += dwhb

			error = np.sum((inputs - self.visibleact)**2)
			if (epoch%50==0) and not silent: 
				print epoch, error/np.shape(inputs)[0], self.energy(visible,self.hiddenprob,labels).sum()

			visible = inputs
			self.labelact = labels

		if not silent:
			self.compute_hidden(inputs,labels)
			print self.energy(visible,self.hiddenprob,labels).sum()
		return error 


	def cddemo(self,inputs,labels=None,dw=None,dwl=None,dwvb=None,dwhb=None,dwlb=None,silent=False):

		# Clamp input into visible nodes
		visible = inputs
		#self.visibleact = inputs
		self.labelact = labels

	        dw = 0. if dw is None else dw
	        dwl = 0. if dwl is None else dwl
	        dwvb = 0. if dwvb is None else dwvb
	        dwhb = 0. if dwhb is None else dwhb
	        dwlb = 0. if dwlb is None else dwlb
		
		for epoch in range(self.nepochs):
			print epoch
			# Sample the hidden variables
			print self.compute_hidden(visible,labels)
	
			# Compute <vh>_0
			positive = np.dot(inputs.T,self.hiddenact)
			#positive = np.dot(inputs.T,self.hiddenprob)
			positivevb = inputs.sum(axis=0)
			positivehb = self.hiddenprob.sum(axis=0)

			print positive, positivevb, positivehb

			# Do limited Gibbs sampling to sample from the hidden distribution
			for j in range(self.nCDsteps):	
				print self.compute_visible(self.hiddenact)
				print self.compute_hidden(self.visibleact,self.labelact)

			# Compute <vh>_n
			negative = np.dot(self.visibleact.T,self.hiddenact)
			#negative = np.dot(self.visibleact.T,self.hiddenprob)
			negativevb = self.visibleact.sum(axis=0)
			negativehb = self.hiddenprob.sum(axis=0)
			print negative, negativevb, negativehb

			# Learning rule (with momentum)
			dw = self.eta * ((positive - negative) / np.shape(inputs)[0] - self.decay*self.weights) + self.momentum*dw
			self.weights += dw

			dwvb = self.eta * (positivevb - negativevb) / np.shape(inputs)[0] + self.momentum*dwvb
			self.visiblebias += dwvb
			dwhb = self.eta * (positivehb - negativehb) / np.shape(inputs)[0] + self.momentum*dwhb
			self.hiddenbias += dwhb

			error = np.sum((inputs - self.visibleact)**2)
			if (epoch%50==0) and not silent: 
				print epoch, error/np.shape(inputs)[0], self.energy(visible,self.hiddenprob,labels).sum()

			visible = inputs
			self.labelact = labels

		if not silent:
			self.compute_hidden(inputs,labels)
			print self.energy(visible,self.hiddenprob,labels).sum()
		return error 
def test_rbm2():

	import rbm
	
  	r = rbm.rbm(6, 2,nCDsteps=1,momentum=0.9,nepochs = 8000)
  	#r = rbm.rbm(6, 2,nCDsteps=1,nepochs = 2)
  	inputs = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
  	r.contrastive_divergence(inputs)
  	print r.weights
	print r.visiblebias
	print r.hiddenbias
  
	test = np.array([[0,0,0,1,1,0]])
	r.compute_hidden(test)
	print r.hiddenact
	#rnd = np.random.rand(r.nhidden)
	#ans = np.where(r.hiddenprob>rnd,1,0)
	#print ans[0,1:]

def test_rbm3():

	import rbm
	
  	r = rbm.rbm(5, 2,nCDsteps=1,momentum=0.9,nepochs = 100)
  	inputs = np.array([[0,1,0,1,1],[1,1,0,1,0],[1,1,0,0,1],[1,1,0,0,1], [1,0,1,0,1],[1,1,1,0,0]])
  	r.contrastive_divergence(inputs)
  	#r.cddemo(inputs)
  	print r.weights
	print r.visiblebias
	print r.hiddenbias
	print r.hiddenact
	print "---"
  
	test = np.array([[1,1,0,0,1]])
	r.compute_hidden(test)
	print r.hiddenact

	test = np.array([[1,0]])
	r.compute_visible(test)
	print r.visibleact

	test = np.array([[1,0]])
	r.compute_visible(test)
	print r.visibleact

	test = np.array([[1,0]])
	r.compute_visible(test)
	print r.visibleact
def test_rbm():
	import rbm
	import cPickle, gzip

	# Load the dataset
	f = gzip.open('mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	r = rbm.rbm(28*28,100,inputs = train_set[0][:100,:],momentum=0.4,nCDsteps=3,nepochs=5000)
	r.contrastive_divergence(train_set[0][:100,:])
	
def test_rbm_learning():
   	import rbm
	import mdp

    	rb = rbm.rbm(4,2,nlabels=None,eta=0.3,momentum=0.9,nCDsteps=1,nepochs=500)
    	rw = mdp.utils.random_rot(max(4,2), dtype='d')[:4, :2]
	rb.weights=rw

    	# the observations consist of two disjunct patterns that never appear together
    	N=10000
    	v = np.zeros((N,4))
    	for n in range(N):
        	r = np.random.rand()
        	if r>0.666: v[n,:] = [0,1,0,1]
        	elif r>0.333: v[n,:] = [1,0,1,0]

        rb.contrastive_divergence(v) #, delta=delta, momentum=mom)

def test_labelled_rbm_learning():
	import rbm
	rb = rbm.rbm(4,4,nlabels=2,eta=0.3,momentum=0.5,nCDsteps=1,nepochs=1000)
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


        rb.contrastive_divergence(v,l) #, delta=delta, momentum=mom)

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
	

	import rbm
	rb = rbm.rbm(20*16,50,nlabels=None,eta=0.3,momentum=0.5,nCDsteps=3,nepochs=1000)
	rb.contrastive_divergence(v)

	hid,phid = rb.compute_hidden(v)
	vis,pvis,lab = rb.compute_visible(hid)
	
	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(pvis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(vis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	newv = inputs[:,nexamples:,:].reshape(nclasses*(39-nexamples),20*16)
	newl = labels[:,nexamples:,:].reshape(nclasses*(39-nexamples),nclasses)
	hid,phid = rb.compute_hidden(newv)
	vis,pvis,lab = rb.compute_visible(hid)

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(newv[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(pvis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(vis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	rb = rbm.rbm(20*16,50,nlabels=nclasses,eta=0.3,momentum=0.2,nCDsteps=3,nepochs=1500)
	rb.contrastive_divergence(v,l)
	rb.classify(v,l)
	rb.classify(newv,newl)

	hid,phid = rb.compute_hidden(v,l)
	vis,pvis,lab = rb.compute_visible(hid)

	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(vis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	hid,phid = rb.compute_hidden(newv,newl)
	vis,pvis,lab = rb.compute_visible(hid)

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(newv[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(pvis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	pl.figure() 
	for i in range(57):
		pl.subplot(6,10,i+1), pl.imshow(vis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	vis = np.random.randn(np.shape(v)[0],np.shape(v)[1])*0.05

	for i in range(1000):
		hid,phid = rb.compute_hidden(vis,l)
		vis,pvis, lab = rb.compute_visible(phid)

	pl.figure() 
	for i in range(60):
		pl.subplot(6,10,i+1), pl.imshow(vis[i,:].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

#learn_letters()

#test_labelled_rbm_learning()
#test_rbm_learning()
#test_rbm()

		
