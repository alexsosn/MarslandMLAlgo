
# Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Code to run the decision tree on the Party dataset
import dtree

tree = dtree.dtree()
party,classes,features = tree.read_data('party.data')
t=tree.make_tree(party,classes,features)
tree.printTree(t,' ')

print tree.classifyAll(t,party)

for i in range(len(party)):
    tree.classify(t,party[i])


print "True Classes"
print classes
