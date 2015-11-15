
# Code from Chapter 17 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import pylab as pl

class hopfield:
	def __init__(self,inputs,synchronous=False,random=True):
		self.nneurons = np.shape(inputs)[1]
		self.weights = np.zeros((self.nneurons,self.nneurons))
		self.activations = np.zeros((self.nneurons,1))
		self.synchronous = synchronous
		self.random = random

	def set_neurons(self,input):
		self.activations = input

	def update_neurons(self):
		if self.synchronous:
			#print self.weights*self.activations
			act = np.sum(self.weights*self.activations,axis=1)
			self.activations = np.where(act>0,1,-1)
			#print self.activations
		else:
			order = np.arange(self.nneurons)
			if self.random:
				np.random.shuffle(order)
			for i in order:
				if np.sum(self.weights[i,:]*self.activations)>0:
					self.activations[i] = 1
				else:
					self.activations[i] = -1
		return self.activations

	def set_weights(self,inputs):

		ninputs = np.shape(inputs)[0]
		for i in range(self.nneurons):
			for j in range(self.nneurons):
				if i != j:
					for k in range(ninputs):
						self.weights[i,j] += inputs[k,i]*inputs[k,j]
		self.weights /= ninputs

	def compute_energy(self):
		energy = 0
		for i in range(self.nneurons):
			for j in range(self.nneurons):
				energy += self.weights[i,j]*self.activations[i]*self.activations[j]
		return -0.5*energy

	def print_net(self):
		print self.weights
		print self.compute_energy()

	def print_out(self):
		print self.activations

def learn_letters():
        import scipy.io as sio
        import pylab as pl
	pl.ion()

        nperclass = 39
	classes = np.arange(20)
        #classes = [0, 11, 17] # A, C, S
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

	inputs = np.where(inputs==0,-1,1)

        v = inputs[:,0,:].reshape(nclasses, 20*16)
        l = labels[:,0,:].reshape(nclasses, nclasses)


	# Train a Hopfield network
	import hopfield
	h = hopfield.hopfield(v[:10,:])
	h.set_weights(v[:10,:])

	# This is the training set
	pl.figure(), 
	#pl.title('Training Data')
	pl.suptitle('Training Data', fontsize=14) 
	for i in range(10):
		pl.subplot(2,5,i), pl.imshow(v[i,:].reshape(20,16),cmap=pl.cm.gray), pl.axis('off')
	
	#which = 2
	#mask = np.ones(20*16)
	#x = np.random.randint(320,size=20)
	#mask[x] = -1

	#h.set_neurons(v[which,:]*mask)
	#print h.compute_energy()

	#nrec = 3
	#new = np.zeros((320,nrec))
	#for i in range(1,nrec):
		#new[:,i] = h.update_neurons()
		#print h.compute_energy()
		#pl.figure(), pl.imshow(new[:,i].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.title('Noisy Image. Reconstruction Step %s'%i), pl.axis('off')
	
	#pl.figure(), pl.imshow((v[which,:]).reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.title('Original Image.'), pl.axis('off')
	#pl.figure(), pl.imshow((v[which,:]*mask).reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.title('Noisy Image.'), pl.axis('off')

	which = 12
	h.set_neurons(v[which,:])
	print h.compute_energy()
	pl.figure(), pl.imshow((v[which,:]).reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.title('Novel Image.'), pl.axis('off')

	nrec = 5
	new2 = np.zeros((320,nrec))
	for i in range(nrec):
		new2[:,i] = h.update_neurons()
		print h.compute_energy()
		pl.figure(), pl.imshow(new2[:,i].reshape(20,16),cmap=pl.cm.gray,interpolation='nearest'), pl.title('Novel Image. Reconstruction Step %s'%i), pl.axis('off')

def test_hopfield():

	import hopfield
	inputs = np.array([[1,1,1,1, -1,-1,-1,-1],
	[1,-1,1,-1, 1,-1,1,-1]])

	print inputs
	#h = hopfield.hopfield(inputs)
	h = hopfield.hopfield(inputs,synchronous=True)
	#h = hopfield.hopfield(inputs,random=False)

	print "Setting weights"
	h.set_weights(inputs)
	h.print_net()

	print "Input 0"
	print inputs[0,:]
	h.set_neurons(inputs[0,:])
	h.update_neurons()
	h.print_out()
	print "------"
	print "Input 1"
	print inputs[1,:]
	h.set_neurons(inputs[1,:])
	h.update_neurons()
	h.print_out()

	test_in = np.array([1,1,1,1, 1,-1,-1,-1])
	#test_in = np.array([1,1,1,-1, 1,-1,-1,-1])
	h.set_neurons(test_in)
	print h.compute_energy()
	h.print_out()
	h.update_neurons()
	print h.compute_energy()
	h.update_neurons()
	print h.compute_energy()
	h.update_neurons()
	print h.compute_energy()
	h.print_out()

def test_hopfield2():

	import hopfield
	inputs = np.array([[-1,-1,1,-1,-1, -1,-1,1,-1,-1 ,-1,-1,1,-1,-1 ,-1,-1,1,-1,-1 ,-1,-1,1,-1,-1],
	[1,1,1,1,1 ,-1,-1,-1,-1,1 ,1,1,1,1,1 ,1,-1,-1,-1,-1 ,1,1,1,1,1],
	[1,1,1,1,1 ,-1,-1,-1,-1,1 ,1,1,1,1,1 ,-1,-1,-1,-1,1 ,1,1,1,1,1] ])

	h = hopfield.hopfield(inputs)
	#h = hopfield.hopfield(inputs,synchronous=True,random=False)
	h.print_net()

	print "Updating neurons"
	h.update_neurons()
	h.print_out()

	print "Updating weights"
	h.set_weights(inputs)
	h.update_neurons()
	h.print_net()
	h.print_out()
	print "------"

	test_in = np.array([-1,-1,1,-1,-1, -1,-1,-1,-1,-1, -1,1,1,-1,-1, -1,-1,1,-1,-1, -1,-1,1,-1,-1])
	h.set_neurons(test_in)
	h.print_out()
	h.update_neurons()
	h.print_out()

def show_reverse():

	import hopfield

	inputs = np.array([[1,1,1,1, -1,-1,-1,-1, 1,1,1,1, -1,-1,-1,-1],[ 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,]])
	h = hopfield.hopfield(inputs)
	h.set_weights(inputs)
	import pylab as pl
	pl.ion()

	a = np.array([1,1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1])
	pl.figure(), pl.imshow(a.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'),pl.axis('off')
	print a.reshape(4,4)
	h.set_neurons(a)
	print h.compute_energy()
	y = h.update_neurons()
	pl.figure(), pl.imshow(y.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'),pl.axis('off')
	print h.compute_energy()
	h.update_neurons()
	print h.compute_energy()
	out = h.update_neurons()
	print h.compute_energy()

	#pl.figure(), pl.imshow(out.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'),pl.axis('off')

	a = np.array([-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,1,1])
	pl.figure(), pl.imshow(a.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'), pl.axis('off')

	print a.reshape(4,4)
	h.set_neurons(a)
	print h.compute_energy()
	x = h.activations.copy()
	print x.reshape(4,4)
	pl.figure(), pl.imshow(x.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'),pl.axis('off')
	print h.compute_energy()
	set = h.update_neurons()
	print set.reshape(4,4)
	print h.compute_energy()
	pl.figure(), pl.imshow(set.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'),pl.axis('off')
	#next = h.update_neurons()
	#print h.compute_energy()
	#pl.figure(), pl.imshow(next.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'),pl.axis('off')
	#new = h.update_neurons()
	#print h.compute_energy()
	#pl.figure(), pl.imshow(new.reshape(4,4),cmap=pl.cm.gray,interpolation='nearest'),pl.axis('off')

	pl.show()

#test_hopfield()
#learn_letters()
