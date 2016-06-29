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
            self._isSetup = obj._isSetup

        else:

            self.n = 0

            # initialize tempo to uniform random value
            tempo = 140. * np.random.rand() + 60.

            # initialize particle
            self.theta = np.array(((0.,),(tempo,)),dtype=np.float64)
            self.wk = 1.0
            self.ck = 0.

            # initialize Kalman filter properties
            self.xkk = np.array(((0.,),(1.,)),dtype=np.float64)
            self.Pkk = np.eye(2)

            # Particle protected properties
            self._isSetup = False

        self.H = np.array(((1.,0.),),dtype=np.float64)

    def step(self,observation):

        if not self._isSetup:
            self.setupImpl()

        if self.n == 0:
            
            # randomly draw tempo process from prior distribution
            tempo = 140. * np.random.rand() + 60.
            self.theta = np.array(((observation,),(tempo,)),dtype=np.float64)
            
            # randomly draw beat location from prior distribution
            self.ck = prior.nextBeatLocation()

            # initialize Kalman filter
            self.xkk = self.theta

            # update counter
            self.n = self.n + 1

            return

        # determine location of previous whole beat
        prevBeat = np.floor(self.ck)

        # randomly choose next beat location
        nextBeat = prior.nextBeatLocation()
        while (prevBeat+nextBeat)-self.ck < np.sqrt(
            eps): nextBeat = prior.nextBeatLocation()
        next_ck = prevBeat+nextBeat

        # compute random jump parameter
        gamma = prevBeat+nextBeat-self.ck

        # compose covariance matrix
        Q = np.array((((gamma**3)/3.0,(gamma**2)/2.0),
            ((gamma**2)/2.0,gamma)),dtype=np.float64)

        # compose transition matrix
        PHI = np.array(((1.0,gamma),(0.,1.)),dtype=np.float64)

        # update covariance estimate using
        # Kalman filter

        # time update (prediction)
        x_est = np.dot(PHI,self.theta)
        P_est = (np.dot(PHI,np.dot(self.Pkk,PHI.T)) + 
            1e-4*Q)

        # update
        y = observation - np.dot(self.H,x_est)
        S = (np.dot(self.H,np.dot(P_est,self.H.T)) + 
            1e-6*np.eye(1))
        K = np.dot(np.dot(P_est,self.H.T),np.linalg.inv(S))
        self.xkk = x_est + np.dot(K,y)
        self.Pkk = np.dot(np.eye(2)-np.dot(K,self.H),P_est)

        # pick a new state for particle
        self.theta = prior.mvnrnd(self.xkk,self.Pkk)

        a = prior.observationPDF(y,S)
        b = prior.importancePDF(self.theta,self.xkk,self.Pkk)
        c = prior.importancePDF(self.theta,x_est,self.Pkk)

        # evaluate new weight
        self.wk = np.float64((prior.transitionPDF(next_ck) *
            c *
            prior.observationPDF(y,S)) /
            b)

        self.ck = next_ck
        self.n = self.n + 1

    def setupImpl(self):
        self._isSetup = True

if __name__ == '__main__':

    ck = np.float64(0)
    tempo = (60.0 / (140. * np.random.rand() + 60.))
    theta = np.array(((0.,),(tempo,)))

    my_particles = [None] * 100
    for i in range(100):
        my_particles[i] = Particle()

    for n in range(100):

        # determine location of previous whole beat
        prevBeat = np.floor(ck)

        # randomly choos next beat location
        nextBeat = prior.nextBeatLocation()
        while (prevBeat+nextBeat)-ck < np.sqrt(
            eps): nextBeat = prior.nextBeatLocation()

        # compute random jump parameter
        gamma = (prevBeat+nextBeat)-ck

        # compute tempo process state transition
        PHI = np.array(((1.,gamma),(0.,1.)),dtype=np.float64)
        theta = np.dot(PHI,theta)

        if n == 0:
            for i in range(100):
                my_particles[i].step(theta[0][0])
            continue

        particle_list = [None] * 3000
        for i in range(100):
            for j in range(30):
                particle_list[i*30+j] = Particle(my_particles[i])

        print "theta =", theta
        for i in range(3000):
            particle_list[i].step(theta[0,0])
    
        wk = np.empty((3000,))
        for i in range(3000):
            wk[i] = particle_list[i].wk

        if np.abs(np.sum(wk)) > np.sqrt(eps):
            for i in range(3000):
                particle_list[i].wk = particle_list[i].wk / np.sum(wk)
            for i in range(3000):
                wk[i] = particle_list[i].wk

        # sort particles in descending order
        particle_list_unsorted = []
        for i in range(3000):
            particle_list_unsorted.append((particle_list[i],particle_list[i].wk))
        particle_list_sorted = sorted(particle_list_unsorted,
            key=lambda p: p[1], reverse=True)

        theta_mmse = np.array(((0.,),(0.,)))
        for i in range(100):
            theta_mmse = theta_mmse + (particle_list_sorted[i][0].wk * 
                particle_list_sorted[i][0].theta)

        print 'theta_mmse =', theta_mmse

        raw_input()

        # draw next state from prior distribution
        #randomState = prior.mvnrnd(self.xkk,self.Pkk)
        #self.theta = randomState
        #print "  self.theta =", self.theta

        my_particles = [None]*100
        for i in range(100):
            my_particles[i] = particle_list_sorted[i][0]

        # update quantized score location
        ck = prevBeat+nextBeat

