
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014
import pylab as pl
import numpy as np
import mlp
import cPickle, gzip

# Read the dataset in (code from sheet)
f = gzip.open('../2 Linear/mnist.pkl.gz','rb')
tset, vset, teset = cPickle.load(f)
f.close()

nread = 200
# Just use the first few images
train_in = tset[0][:nread,:]

# This is a little bit of work -- 1 of N encoding
# Make sure you understand how it does it
train_tgt = np.zeros((nread,10))
for i in range(nread):
    train_tgt[i,tset[1][i]] = 1

test_in = teset[0][:nread,:]
test_tgt = np.zeros((nread,10))
for i in range(nread):
    test_tgt[i,teset[1][i]] = 1

# We will need the validation set
valid_in = vset[0][:nread,:]
valid_tgt = np.zeros((nread,10))
for i in range(nread):
    valid_tgt[i,vset[1][i]] = 1

for i in [1,2,5,10,20]:  
    print "----- "+str(i)  
    net = mlp.mlp(train_in,train_tgt,i,outtype='softmax')
    net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,0.1)
    net.confmat(test_in,test_tgt)
