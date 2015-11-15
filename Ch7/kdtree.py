
# Code from Chapter 7 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The code to construct and use a KD-tree
# There is a simple example at the bottom

import numpy as np

class node:
	# A passive class to hold the nodes
	pass

def makeKDtree(points, depth):
	if np.shape(points)[0]<1:
		# Have reached an empty leaf
		return None
	elif np.shape(points)[0]<2:
		# Have reached a proper leaf
		newNode = node()
		newNode.point = points[0,:]
		newNode.left = None
		newNode.right = None
		return newNode
	else:
		# Pick next axis to split on	
		whichAxis = np.mod(depth,np.shape(points)[1])
		
		# Find the median point
		indices = np.argsort(points[:,whichAxis])
		points = points[indices,:]
		median = np.ceil(float(np.shape(points)[0]-1)/2)

		# Separate the remaining points
		goLeft = points[:median,:]
		goRight = points[median+1:,:]

		# Make a new branching node and recurse
		newNode = node()
		newNode.point = points[median,:]
		newNode.left = makeKDtree(goLeft,depth+1)
		newNode.right = makeKDtree(goRight,depth+1)
		return newNode

def returnNearest(tree,point,depth):
	if tree.left is None:
		# Have reached a leaf
		distance = np.sum((tree.point-point)**2)
		return tree.point,distance,0
	else:
		# Pick next axis to split on
		whichAxis = np.mod(depth,np.shape(point)[0])

		# Recurse down the tree
		if point[whichAxis]<tree.point[whichAxis]:
			bestGuess,distance,height = returnNearest(tree.left,point,depth+1)
		else:
			bestGuess,distance,height = returnNearest(tree.right,point,depth+1)

		if height<=2:
			# Check the sibling
			if point[whichAxis]<tree.point[whichAxis]:
				bestGuess2,distance2,height2 = returnNearest(tree.right,point,depth+1)
			else:
				bestGuess2,distance2,height2 = returnNearest(tree.left,point,depth+1)
		
			# Check this node
			distance3 = np.sum((tree.point-point)**2)
			if (distance3<distance2):
				distance2 = distance3
				bestGuess2 = tree.point
			
			if (distance2<distance):
				distance = distance2
				bestGuess = bestGuess2
		return bestGuess,distance,height+1


points = np.array([[1,6],[2,2],[3,7],[5,4],[6,8],[6,1],[7,5]])
tree = makeKDtree(points,0)
print returnNearest(tree,np.array([3,5]),0)
print returnNearest(tree,np.array([4.5,2]),0)

