
# Code from Chapter 15 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The linear congruential pseudo-random number generator
import numpy as np

def lcg(x0,n):
    # These choices show the periodicity very well
    # Better choices are a = 16,807 m = 2**31 -1 c = 0
    # Or m = 2**32 a = 1,664,525 c = 1,013,904,223
    a = 23
    m = 197
    c = 0
    
    rnd = np.zeros((n))
    
    rnd[0] = np.mod(a*x0 + c,m)
    
    for i in range(1,n):
        rnd[i] = np.mod(a*rnd[i-1]+c,m)
        
    return rnd
    
print lcg(3,80) 
