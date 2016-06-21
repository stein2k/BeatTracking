import sys
import numpy as np

eps = sys.float_info.epsilon

class __Private:

    def __init__(self):

        # define prior pdf
        self.pdf = np.zeros((24,),dtype=np.float64)
        for i in range(24):
            self.pdf[i] = np.exp(-0.5*np.log2(self.reduce_denom(
                np.float64(i+1)/24.0)))
        self.pdf = self.pdf/np.sum(self.pdf)

        # define prior cdf
        self.cdf = np.cumsum(self.pdf)
    
    def rat(self,X,tol=1e-6):

        k = 0
        C = np.array([[1,0],[0,1]])

        x = X

        while True:
            
            k = k+1
            neg = x<0

            d = np.round(x)

            x = x-d

            C = np.concatenate((np.dot(C,np.array([[d],[1]])),
                np.reshape(C[:,0],(2,1))),axis=1)

            # check for exit condition
            if (np.abs(x)<np.sqrt(eps)) or (np.abs(C[0,0]/C[1,0]-X)
                < max(tol,np.abs(X)*eps)): break

            x = 1.0/x

        return (C[0,0]/np.sign(C[1,0]),np.abs(C[1,0]))

    def reduce_denom(self,X):
        return self.rat(X)[1]


__private = None
    
def mvnrnd(MU,SIGMA):

    # calculate dimensionality of multivariate normal
    # distribution
    N = np.size(MU,0)

    # find matrix A such that SIGMA = AA.T using 
    # spectral decomposition
    D,V = np.linalg.eig(SIGMA)
    A = np.dot(V,np.diag(np.sqrt(D)))
    #A = np.linalg.cholesky(SIGMA)

    # create vector of N independent standard normal
    # variates
    z = np.random.randn(N,1)

    # draw miltivariate random value from distribution
    x = MU + np.dot(A,z)

    return x

def nextBeatLocation():

    global __private
    
    # define prior pdf
    if __private is None:
        __private = __Private()

    # draw from uniform random distribution
    u = np.random.rand()

    # compute inverse transform
    #X = (((np.float64(np.array(np.where(u<__private.cdf))[0][0])+1.)
    #    / 24.0) + np.floor(np.random.exponential(0.5)))
    X = (((np.float64(np.array(np.where(u<__private.cdf))[0][0])+1.)
        / 24.0))

    return X
