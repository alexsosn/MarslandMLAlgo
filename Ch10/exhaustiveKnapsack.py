
# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# An exhaustive search to solve the Knapsack problem
import numpy as np

def exhaustive():
    maxSize = 500    
    sizes = np.array([109.60,125.48,52.16,195.55,58.67,61.87,92.95,93.14,155.05,110.89,13.34,132.49,194.03,121.29,179.33,139.02,198.78,192.57,81.66,128.90])

    best = 0

    twos = np.arange(-len(sizes),0,1)
    twos = 2.0**twos
    
    for i in range(2**len(sizes)-1):
        string = np.remainder(np.floor(i*twos),2) 
        fitness = np.sum(string*sizes)
        if fitness > best and fitness<500:
            best = fitness
            bestString = string
    print best
    print bestString
          
exhaustive()
