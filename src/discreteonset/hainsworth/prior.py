import sys
import numpy as np

eps = sys.float_info.epsilon

class _Private:

    pdf = None
    cdf = None

    @staticmethod 
    def rat(X,tol=1e-6):

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

    @staticmethod
    def reduce_denom(X):
        return np.int64(_Private.rat(X)[1])

    @staticmethod
    def transition(X):

        if _Private.pdf is None:
            _Private.pdf = np.zeros((24,),dtype=np.float64)
            for i in range(24):
                _Private.pdf[i] = np.exp(-0.5*np.log2(_Private.reduce_denom(
                    np.float64(i+1)/24.0)))
            _Private.pdf = _Private.pdf/np.sum(_Private.pdf)

        if _Private.cdf is None:
            _Private.cdf = np.cumsum(_Private.pdf)

        # draw from uniform random distribution
        u = np.random.rand()

        # determine note type
        tmp = np.float64(np.array(np.where(u<_Private.cdf))[0][0])
        note_type = 1.0 / _Private.reduce_denom((tmp+1.0)/24.0)

        if np.abs(np.round(X/note_type)-X/note_type) < np.sqrt(eps):
            return X+note_type
        else:
            return np.ceil(X/note_type)*note_type

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

def nextBeatLocation(X=0):


    # compute inverse transform
    #X = (((np.float64(np.array(np.where(u<__private.cdf))[0][0])+1.)
    #    / 24.0) + np.floor(np.random.exponential(0.5)))
    X = _Private.transition(X)


    return X

def transitionPDF(X):

    note_type = -1

    # check for whole note
    if np.abs(X-np.floor(X))<np.sqrt(eps):
        note_type = np.int64(23)
    else:
        note_type = np.int64(24.0*(X-np.floor(X)))

    return _Private.pdf[note_type]

def observationPDF(X,SIGMA):
    p = ((np.linalg.det(SIGMA)**(-0.5)) *
        np.exp(-0.5*np.dot(X.T,np.dot(np.linalg.inv(SIGMA),
        X)))/2.0/np.pi)
    return p

def importancePDF(X,MU,SIGMA):
    p = ((np.linalg.det(SIGMA)**(-0.5)) *
        np.exp(-0.5*np.dot(np.transpose(X-MU),np.dot(
        np.linalg.inv(SIGMA),X-MU)))) / 2.0 / np.pi
    return p
