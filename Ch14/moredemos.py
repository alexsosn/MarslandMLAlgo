
# Code from Chapter 14 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Demonstration of the SOM algorithm on the Wine dataset (and the e-coli dataset)
import pylab as pl
import numpy as np
import som

wine = np.loadtxt('wine.data',delimiter=',')

classes = wine[:,0]
data = wine[:,1:]
data -= np.mean(data,axis=0)
data /= data.max(axis=0)

#ecoli = loadtxt('shortecoli.dat')
#classes = ecoli[:,7:]
#data = ecoli[:,:7]
#data -= mean(data,axis=0)
#data /= data.max(axis=0)

order = range(shape(data)[0])
np.random.shuffle(order)
split = int(np.round(np.shape(data)[0]/2))
train = data[order[:split],:]
target = classes[order[:split],:]

test = data[order[split:],:]
ttarget = classes[order[:split],:]

net = som.som(15,15,train,eta_b=0.3,eta_n=0.1,nSize=0.5,alpha=1,usePCA=1,useBCs=1,eta_bfinal=0.03,eta_nfinal=0.01,nSizefinal=0.05)
net.somtrain(train,12000)

best = np.zeros(shape(test)[0],dtype=int)

for i in range(shape(test)[0]):
    best[i],activation = net.somfwd(train[i,:])

#print best
#print ttarget

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(target == 0)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = pl.find(target == 1)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = pl.find(target == 2)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')

pl.figure(2)
best = np.zeros(shape(test)[0],dtype=int)

for i in range(shape(test)[0]):
    best[i],activation = net.somfwd(test[i,:])

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(ttarget == 0)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = pl.find(ttarget == 1)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = pl.find(ttarget == 2)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')

pl.show()
