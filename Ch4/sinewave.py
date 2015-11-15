
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The sinewave regression example

import pylab as pl
import numpy as np

# Set up the data
x = np.linspace(0,1,40).reshape((40,1))
t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40).reshape((40,1))*0.2
x = (x-0.5)*2

# Split into training, testing, and validation sets
train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

# Plot the data
pl.plot(x,t,'o')
pl.xlabel('x')
pl.ylabel('t')

# Perform basic training with a small MLP
import mlp
net = mlp.mlp(train,traintarget,3,outtype='linear')
net.mlptrain(train,traintarget,0.25,101)

# Use early stopping
net.earlystopping(train,traintarget,valid,validtarget,0.25)

# Test out different sizes of network
#count = 0
#out = zeros((10,7))
#for nnodes in [1,2,3,5,10,25,50]:
#    for i in range(10):
#        net = mlp.mlp(train,traintarget,nnodes,outtype='linear')
#        out[i,count] = net.earlystopping(train,traintarget,valid,validtarget,0.25)
#    count += 1
#    
#test = concatenate((test,-ones((shape(test)[0],1))),axis=1)
#outputs = net.mlpfwd(test)
#print 0.5*sum((outputs-testtarget)**2)
#
#print out
#print out.mean(axis=0)
#print out.var(axis=0)
#print out.max(axis=0)
#print out.min(axis=0)

pl.show()
