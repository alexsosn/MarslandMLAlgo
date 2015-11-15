
# Code from Chapter 13 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Comparison of stumping and bagging on the mushroom dataset
import numpy as np
import dtw
import bagging
import randomforest

tree = dtw.dtree()
bagger = bagging.bagger()
forest = randomforest.randomforest()
mushroom,classes,features = tree.read_data('agaricus-lepiota.data')

w = np.ones((np.shape(mushroom)[0]),dtype = float)/np.shape(mushroom)[0]

f = forest.rf(mushroom,classes,features,10,7,2)
print forest.rfclass(f,mushroom)

t=tree.make_tree(mushroom,w,classes,features,1)
tree.printTree(t,' ')

print "Tree Stump Prediction"
print tree.classifyAll(t,mushroom)
print "True Classes"
print classes

c=bagger.bag(mushroom,classes,features,20)
print "Bagged Results"
print bagger.bagclass(c,mushroom)
