
# Code from Chapter 13 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import dtree
import randomforest
tree = dtree.dtree()
forest = randomforest.randomforest()
data,classes,features = tree.read_data('car.data')
train = data[::2][:]
test = data[1::2][:]
trainc = classes[::2]
testc = classes[1::2]
f=f = forest.rf(train,trainc,features,50,100,2,maxlevel=3)
#f=f = forest.rf(train,trainc,features,100,200,2)
out = forest.rfclass(f,test)

import numpy as np

a = np.zeros(len(out))
b = np.zeros(len(out))
d = np.zeros(len(out))

for i in range(len(out)):
    if testc[i] == 'good' or testc[i]== 'v-good':
        b[i] = 1
        if out[i] == testc[i]:
            d[i] = 1
    if out[i] == testc[i]:
        a[i] = 1
print "-----"
print "Forest"
print "Number correctly predicted",np.sum(a)
print "Number of testpoints ",len(a)
print "Percentage Accuracy ",np.sum(a)/len(a)*100.0
print ""
print "Number of cars rated as good or very good", np.sum(b)
print "Number correctly identified as good or very good",np.sum(d) 
print "Percentage Accuracy",np.sum(d)/np.sum(b)*100.0
