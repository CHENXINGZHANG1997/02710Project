'''
Created on Apr 2, 2013

@author: Bhushan
'''
import numpy as np
import math as m
import PyML


if __name__ == '__main__':
    pass

len1 = 10
len2 = 5
A = np.empty([len1,len2], dtype = float)
B = np.empty([len1,len2], dtype = float)


def L1dist(P,Q):
    """returns the l1 norm distance between P and Q"""
    return sum(abs(P-Q))


def GERPDistance(A,B):
    """Implementation of GERP distance with multivariate time series A and B"""
    x,a = A.shape #x is the number of time points in series A
    y,b = B.shape #y is the number of time points in series B
    g = np.zeros([a]) #gap penalty vector    
    GER = np.zeros([x,y], dtype = float) #GER matrix to find the optimum score
    
    #Initialization
    GER[0,0] = 0.0
    for i in xrange(1,x):
        GER[i,0] = GER[i-1,0] + L1dist(A[i,:],g)
    for i in xrange(1,y):
        GER[0,i] = GER[0,i-1] + L1dist(B[i,:],g)
    
    #Extention    
    for i in xrange(1,x):
        for j in xrange(1,y):
            
            S = [0.0,0.0,0.0]
            S[0] = GER[i-1,j-1] + L1dist(A[i,:],B[j,:]) #match in alignment
            S[1] = GER[i,j-1] + L1dist(B[j,:], g) #gap in A
            S[2] = GER[i-1,j] + L1dist(A[i,:], g) #gap in B
            GER[i,j] = min(S)
            
    return GER[x-1,y-1]


            
def GERPKernel(A,B, sigma =10):
    """ GERP kernel using the elastic, metric GERPDistance between multivariate time series A and B """
    D = GERPDistance(A,B)
    result = m.exp((-D*D)/2*sigma*sigma)
    return result



    
    
                