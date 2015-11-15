
# Code from Chapter 13 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2014

import numpy as np
import dtree

class randomforest:

    """The random forest algorithm based on the decision tree of Chapter 6"""
    def __init__(self):
        """ Constructor """
        self.tree = dtree.dtree()

        
    def rf(self,data,targets,features,nTrees,nSamples,nFeatures,maxlevel=5):
    
        nPoints = np.shape(data)[0]
        nDim = np.shape(data)[1]
        self.nSamples = nSamples
        self.nTrees = nTrees
                 
        classifiers = []
   
        for i in range(nTrees):
	    print i
            # Compute bootstrap samples
            samplePoints = np.random.randint(0,nPoints,(nPoints,nSamples))
        
            for j in range(nSamples):
                sample = []
                sampleTarget = []
                for k in range(nPoints):
                    sample.append(data[samplePoints[k,j]])
                    sampleTarget.append(targets[samplePoints[k,j]])
            # Train classifiers
            classifiers.append(self.tree.make_tree(sample,sampleTarget,features,maxlevel,forest=nFeatures))
        return classifiers
    
    def rfclass(self,classifiers,data):
        
        decision = []
        # Majority voting
        for j in range(len(data)):
            outputs = []
            #print data[j]
            for i in range(self.nTrees):
                out = self.tree.classify(classifiers[i],data[j])
                if out is not None:
                    outputs.append(out)
            # List the possible outputs
            out = []
            for each in outputs:
                if out.count(each)==0:
                    out.append(each)
            frequency = np.zeros(len(out))
        
            index = 0
            if len(out)>0:
                for each in out:
                    frequency[index] = outputs.count(each)
                    index += 1
                decision.append(out[frequency.argmax()])
            else:
                decision.append(None)
        return decision
