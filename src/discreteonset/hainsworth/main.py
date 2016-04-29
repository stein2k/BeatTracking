import sys
import numpy as np
import matplotlib.pyplot as plt

class BeatPrior(object):
    def __init__(self):
        self.__pdf = None
        self.__cdf = None

    def nextBeatLocation(self):
        
        # define prior pdf
        if self.__pdf is None:
            self.__pdf = np.zeros((24,),dtype=np.float64)
            for i in range(24):
                self.__pdf[i] = np.exp(-0.5*np.log2(reduce_denom(
                    np.float64(i+1)/24.0)))
            self.__pdf = self.__pdf/np.sum(self.__pdf)

        # define prior cdf
        if self.__cdf is None:
            self.__cdf = np.cumsum(self.__pdf)

        # draw from uniform random distribution
        u = np.random.rand()

        # compute inverse transform
        X = (((np.float64(np.array(np.where(u<self.__cdf))[0][0])+1.)
            / 24.0) + np.floor(np.random.exponential(0.5)))

        return X

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

def reduce_denom(X):
    return rat(X)[1]

class Particle(object):
    def __init__(self):
    
        # initialize particle
        self.theta = None

    def setup(self):
        self.setupImpl()

    def step(self,observation):

        # determine time delta between beats
        gamma = nextBeatLocation()

        # compose transition matrix
        PHI = np.array([[1.0,np.float64(gamma)],[0.,1.]])

        # sample the state
        self.theta = np.dot(PHI,self.theta)

        # update the weight
        self.weight = ((stateTransitionPrior(self.theta,prevTheta)*
            observationPrior(observation,self.theta)) /
            self.

        # update
        self.theta = [nextBeatLocation(),
            60.0/np.float64(np.random.rand]

    def setupImpl(self):

        # sample from initial prior distribution
        self.theta = np.array(((nextBeatLocation(),),
            (60.0/(140.*np.random.rand()+60.),)))

        # compute initial weight
        self.weight = ((stateTransitionPrior(self.theta[0]) *
            observationPrior()
            
            

        print "theta =", self.theta

class ParticleFilter(object):
    def __init__(self):
        self.FilterSize = 100

    def setup(self):
        self.setupImpl()

    def setupImpl(self):

        self.__particles = [None]*self.FilterSize
        for i in range(self.FilterSize):
            self.__particles[i] = Particle()
            self.__particles[i].setup()

def main():
    particleFilter = ParticleFilter()
    particleFilter.setup()

eps = sys.float_info.epsilon

_inst = BeatPrior()
nextBeatLocation = _inst.nextBeatLocation

if __name__ == '__main__':
    main()
