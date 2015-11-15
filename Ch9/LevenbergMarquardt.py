
# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The Levenberg Marquardt algorithm
import numpy as np

def function(p):
    r = np.array([10*(p[1]-p[0]**2),(1-p[0])])
    fp = np.dot(np.transpose(r),r) #= 100*(p[1]-p[0]**2)**2 + (1-p[0])**2
    J = (np.array([[-20*p[0],10],[-1,0]]))
    grad = np.dot(J.T,r.T)
    return fp,r,grad,J

def lm(p0,tol=10**(-5),maxits=100):
    
    nvars=np.shape(p0)[0]
    nu=0.01
    p = p0
    fp,r,grad,J = function(p)
    e = np.sum(np.dot(np.transpose(r),r))
    nits = 0
    while nits<maxits and np.linalg.norm(grad)>tol:
        nits += 1
        fp,r,grad,J = function(p)
        H=np.dot(np.transpose(J),J) + nu*np.eye(nvars)

        pnew = np.zeros(np.shape(p))
        nits2 = 0
        while (p!=pnew).all() and nits2<maxits:
            nits2 += 1
            dp,resid,rank,s = np.linalg.lstsq(H,grad)
            pnew = p - dp
            fpnew,rnew,gradnew,Jnew = function(pnew)
            enew = np.sum(np.dot(np.transpose(rnew),rnew))
            rho = np.linalg.norm(np.dot(np.transpose(r),r)-np.dot(np.transpose(rnew),rnew))
            rho /= np.linalg.norm(np.dot(np.transpose(grad),pnew-p))
            
            if rho>0:
                update = 1
                p = pnew
                e = enew
                if rho>0.25:
                    nu=nu/10
            else: 
                nu=nu*10
                update = 0
        print fp, p, e, np.linalg.norm(grad), nu

p0 = np.array([-1.92,2])
lm(p0)
