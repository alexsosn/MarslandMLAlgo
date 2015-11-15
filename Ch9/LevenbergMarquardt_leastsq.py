
# Code from Chapter 11 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# The Levenberg Marquardt algorithm solving a least-squares problem

import pylab as pl
import numpy as np

def function(p,x,ydata):
    fp = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x])
    r = ydata - fp
    J = np.transpose([-np.cos(p[0]*x)-p[1]*np.cos(p[0]*x)*x, p[0] * np.sin(p[1]*x)*x-np.sin(p[0]*x)])
    grad = np.dot(J.T,r.T)
    return fp,r,grad,J

def lm(p0,x,f,tol=10**(-5),maxits=100):
    
    nvars=np.shape(p0)[0]
    nu=0.01
    p = p0
    fp,r,grad,J = function(p,x,f)
    e = np.sum(np.dot(np.transpose(r),r))
    nits = 0
    while nits<maxits and np.linalg.norm(grad)>tol:
        nits += 1
        
        # Compute current Jacobian and approximate Hessian
        fp,r,grad,J = function(p,x,f)
        H=np.dot(np.transpose(J),J) + nu*np.eye(nvars)
        pnew = np.zeros(np.shape(p))
        nits2 = 0
        while (p!=pnew).all() and nits2<maxits:
            nits2 += 1
            # Compute the new estimate pnew
            #dp = np.linalg.solve(H,grad)
            dp,resid,rank,s = np.linalg.lstsq(H,grad)
            #dp = -dot(linalg.inv(H),dot(transpose(J),transpose(d)))
            pnew = p - dp[:,0]
            
            # Decide whether the trust region is good
            fpnew,rnew,gradnew,Jnew = function(pnew,x,f)
            enew = np.sum(np.dot(np.transpose(rnew),rnew))
            
            rho = np.linalg.norm(np.dot(np.transpose(r),r)-np.dot(np.transpose(rnew),rnew))
            rho /= np.linalg.norm(np.dot(np.transpose(grad),pnew-p))
            
            if rho>0:
                # Keep new estimate
                p = pnew
                e = enew
                if rho>0.25:
                    # Make trust region larger (reduce nu)
                    nu=nu/10
            else: 
                # Make trust region smaller (increase nu)
                nu=nu*10
        print p, e, np.linalg.norm(grad), nu
    return p
    
p0 = np.array([100.5,102.5]) #[ 100.0001126   101.99969709] 1078.36915936 8.87386341319e-06 1e-10 (8 itns)
#p0 = np.array([101,101]) #[ 100.88860713  101.12607589] 631.488571159 9.36938417155e-06 1e-67

p = np.array([100,102])

x = np.arange(0,2*np.pi,0.1)
y = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x]) + np.random.rand(len(x))
p = lm(p0,x,y)
y1 = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x]) #+ np.random.rand(len(x))

pl.plot(x,np.squeeze(y),'-')
pl.plot(x,np.squeeze(y1),'r--')
pl.legend(['Actual Data','Fitted Data'])
pl.show()
