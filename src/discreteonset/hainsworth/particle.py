import numpy as np
import prior

class Particle(object):
    def __init__(self):

        # initialize particle
        self.theta = np.array(((0.,),(0.,)),dtype=np.float64)

        # initialize Kalman filter properties
        self.xkk = np.array(((0.,),(1.,)),dtype=np.float64)
        self.Pkk = np.eye(2)

        self.H = np.array(((1.,0.),),dtype=np.float64)

    def step(self,observation,beatlocation=None):

        # determine time delta between beats
        if beatlocation is None:
            gamma = prior.nextBeatLocation()
        else:
            gamma = beatlocation

        # compose transition matrix
        PHI = np.array(((1.0,gamma),(0.,1.)),dtype=np.float64)

        # compose covariance matrix
        Q = np.array((((gamma**3)/3.0,(gamma**2)/2.0),
            ((gamma**2)/2.0,gamma)),dtype=np.float64)

        # update covariance estimate using
        # Kalman filter

        # time update (prediction)
        x_est = np.dot(PHI,self.xkk)
        P_est = (np.dot(PHI,np.dot(self.Pkk,PHI.T)) + 
            1.0e-6*Q)

        # update
        y = observation - np.dot(self.H,x_est)
        S = (np.dot(self.H,np.dot(P_est,self.H.T)) + 
            1.0e-6*np.eye(1))
        K = np.dot(np.dot(P_est,self.H.T),np.linalg.inv(S))
        self.xkk = x_est + np.dot(K,y)
        self.Pkk = np.dot(np.eye(2)-np.dot(K,self.H),P_est)

        q = ((np.linalg.det(2.0*np.pi*self.Pkk)**(-0.5)) * 
            np.exp(-0.5*np.dot(np.dot(np.transpose(theta-self.xkk),
            self.Pkk),theta-self.xkk)))

        # draw next state from prior distribution
        randomState = prior.mvnrnd(self.xkk,self.Pkk)

        self.theta = randomState

        print "  self.theta =", self.theta

if __name__ == '__main__':

    particle = Particle()

    tempo = (1.0 / (140. * np.random.rand() + 60.))

    theta = np.array(((0.,),(tempo,)))
    particle.theta = theta

    for n in range(100):
        nextbeat = prior.nextBeatLocation()
        PHI = np.array(((1.,nextbeat),(0.,1.)),dtype=np.float64)
        theta = np.dot(PHI,theta)
        print "theta =", theta
        particle.step(theta[0,0],nextbeat)
