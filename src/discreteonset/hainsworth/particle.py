import sys
import numpy as np
import prior

eps = sys.float_info.epsilon

class Particle(object):
    def __init__(self,obj=None):

        if isinstance(obj,Particle):
            self.n = obj.n
            self.theta = obj.theta
            self.wk = obj.wk
            self.ck = obj.ck
            self.xkk = obj.xkk
            self.Pkk = obj.Pkk

        else:

            self.n = 0

            # number of internal states
            self.NumStates = 30

            # initialize particle
            self.theta = np.array(((0.,),(0.,)),dtype=np.float64)
            self.wk = 1.0
            self.ck = 0.

            # initialize Kalman filter properties
            self.xkk = np.array(((0.,),(1.,)),dtype=np.float64)
            self.Pkk = np.eye(2)

        self.H = np.array(((1.,0.),),dtype=np.float64)

    def step(self,observation,ck=None):

        # determine location of previous whole beat
        prevBeat = np.floor(self.ck)

        # randomly choos next beat location
        nextBeat = prior.nextBeatLocation()
        while (prevBeat+nextBeat)-self.ck < np.sqrt(
            eps): nextBeat = prior.nextBeatLocation()

        if ck is not None:
            next_ck = ck
            gamma = ck-self.ck
        else:
            next_ck = prevBeat+nextBeat
            gamma = prevBeat+nextBeat-self.ck

        # compose covariance matrix
        Q = np.array((((gamma**3)/3.0,(gamma**2)/2.0),
            ((gamma**2)/2.0,gamma)),dtype=np.float64)

        # compose transition matrix
        PHI = np.array(((1.0,gamma),(0.,1.)),dtype=np.float64)

        # update covariance estimate using
        # Kalman filter

        # time update (prediction)
        x_est = np.dot(PHI,self.xkk)
        P_est = (np.dot(PHI,np.dot(self.Pkk,PHI.T)) + 
            (0.06**2)*Q)

        # update
        y = observation - np.dot(self.H,x_est)
        S = (np.dot(self.H,np.dot(P_est,self.H.T)) + 
            (0.02**2)*np.eye(1))
        K = np.dot(np.dot(P_est,self.H.T),np.linalg.inv(S))
        self.xkk = x_est + np.dot(K,y)
        self.Pkk = np.dot(np.eye(2)-np.dot(K,self.H),P_est)

        # pick a new state for particle
        self.theta = prior.mvnrnd(self.xkk,self.Pkk)

        # evaluate new weight
        self.wk = self.wk * np.float64((prior.transitionPDF(next_ck) *
            prior.observationPDF(y,S)) /
            prior.importancePDF(self.theta,
            self.xkk,self.Pkk))

        self.ck = ck
        self.n = self.n + 1

if __name__ == '__main__':

    ck = np.float64(0)

    particle = Particle()

    tempo = (60.0 / (140. * np.random.rand() + 60.))

    theta = np.array(((0.,),(tempo,)))
    particle.theta = theta

    for n in range(100):

        print "n =", n

        # determine location of previous whole beat
        prevBeat = np.floor(ck)

        # randomly choos next beat location
        nextBeat = prior.nextBeatLocation()
        while (prevBeat+nextBeat)-ck < np.sqrt(
            eps): nextBeat = prior.nextBeatLocation()

        # compute random jump
        gamma = (prevBeat+nextBeat)-ck

        particle_list = [None] * 30
        for i in range(30):
            particle_list[i] = Particle(particle)

        PHI = np.array(((1.,gamma),(0.,1.)),dtype=np.float64)
        theta = np.dot(PHI,theta)

        print "theta =", theta
        for i in range(30):
            if n<20 or i==0:
                particle_list[i].step(theta[0,0],prevBeat+nextBeat)
            else:
                particle_list[i].step(theta[0,0])
    
        wk = np.empty((30,))
        for i in range(30):
            wk[i] = particle_list[i].wk

        for i in range(30):
            particle_list[i].wk = particle_list[i].wk / np.sum(wk)

        for i in range(30):
            wk[i] = particle_list[i].wk

        theta_mmse = np.array(((0.,),(0.,)))
        for i in range(30):
            theta_mmse = theta_mmse + (particle_list[i].wk * 
                particle_list[i].theta)

        print 'theta_mmse =', theta_mmse
        print 'wk =', wk

        if n > 20:
            raw_input()

        # draw next state from prior distribution
        #randomState = prior.mvnrnd(self.xkk,self.Pkk)
        #self.theta = randomState
        #print "  self.theta =", self.theta

        particle = particle_list[0]

        # update quantized score location
        ck = prevBeat+nextBeat

