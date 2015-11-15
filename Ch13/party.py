
# Code from Chapter 13 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Comparison of stumping and bagging on the Party dataset
import numpy as np
#import dtree
import dtw
import bagging
import randomforest

tree = dtw.dtree()
#tree = dtree.dtree()
bagger = bagging.bagger()
forest = randomforest.randomforest()
party,classes,features = tree.read_data('../6 Trees/party.data')

#w = np.random.rand((np.shape(party)[0]))/np.shape(party)[0]
w = np.ones((np.shape(party)[0]),dtype = float)/np.shape(party)[0]

f = forest.rf(party,classes,features,10,7,2,maxlevel=2)
print "RF prediction"
print forest.rfclass(f,party)

#t=tree.make_tree(party,classes,features)
t=tree.make_tree(party,w,classes,features)
#tree.printTree(t,' ')
print "Decision Tree prediction"
print tree.classifyAll(t,party)

print "Tree Stump Prediction"
print tree.classifyAll(t,party)

c=bagger.bag(party,classes,features,20)
print "Bagged Results"
print bagger.bagclass(c,party)

print "True Classes"
print classes
