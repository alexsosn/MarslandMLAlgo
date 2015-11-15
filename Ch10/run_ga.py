
# Code from Chapter 10 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# A runner for the Genetic Algorithm
import ga
import pylab as pl

pl.ion()
pl.show()

plotfig = pl.figure()

ga = ga.ga(30,'fF.fourpeaks',301,100,-1,'un',4,True)
ga.runGA(plotfig)

pl.pause(0)
#pl.show()
