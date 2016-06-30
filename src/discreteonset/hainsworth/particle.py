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
            self.a = obj.a
            self.b = obj.b
            self.c = obj.c
            self.theta_est = obj.theta_est

        else:

            self.n = 0

            # initialize tempo to uniform random value
            tempo = (60.0 / (140. * np.random.rand() + 60.))

            # initialize particle
            self.theta = np.array(((0.,),(tempo,)),dtype=np.float64)
            self.wk = 1.0
            self.ck = 0.

            # initialize Kalman filter properties
            self.xkk = np.array(((0.,),(tempo,)),dtype=np.float64)
            self.Pkk = np.eye(2)

            # Particle protected properties
            self._isSetup = False

            self.a = None
            self.b = None
            self.c = None
            self.theta_est = None

        self.H = np.array(((1.,0.),),dtype=np.float64)

    def step(self,observation):

        if not self._isSetup:
            self.setupImpl()

        if self.n == 0:
            
            # randomly draw beat location from prior distribution
            self.ck = prior.nextBeatLocation()
            self.theta[0][0] = observation
            self.xkk[0][0] = observation

            # update counter
            self.n = self.n + 1

            return

        # determine location of previous whole beat
        prevBeat = np.floor(self.ck)

        # randomly choose next beat location
        nextBeat = prior.nextBeatLocation(self.ck)
        #nextBeat = 1

        # compute random jump parameter
        gamma = nextBeat-self.ck

        # compose covariance matrix
        Q = np.array((((gamma**3)/3.0,(gamma**2)/2.0),
            ((gamma**2)/2.0,gamma)),dtype=np.float64)

        # compose transition matrix
        PHI = np.array(((1.0,gamma),(0.,1.)),dtype=np.float64)

        self.theta_est = np.dot(PHI,self.theta)

        # update covariance estimate using
        # Kalman filter

        # time update (prediction)
        x_est = np.dot(PHI,self.xkk)
        P_est = (np.dot(PHI,np.dot(self.Pkk,PHI.T)) + 
            1e-3*Q)

        # update
        y = observation - np.dot(self.H,x_est)
        S = (np.dot(self.H,np.dot(P_est,self.H.T)) + 
            1e-4*np.eye(1))
        K = np.dot(np.dot(P_est,self.H.T),np.linalg.inv(S))
        self.xkk = x_est + np.dot(K,y)
        self.Pkk = np.dot(np.eye(2)-np.dot(K,self.H),P_est)

        # pick a new state for particle
        self.theta = prior.mvnrnd(self.xkk,self.Pkk)

        self.a = prior.observationPDF(y,S)
        self.b = prior.importancePDF(self.theta,self.xkk,self.Pkk)
        self.c = prior.importancePDF(self.theta_est,self.theta,self.Pkk)

        # evaluate new weight
        self.wk = self.wk * np.float64((prior.transitionPDF(nextBeat) *
            self.c *
            prior.observationPDF(y,S)) /
            self.b)

        self.ck = nextBeat
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

    observation_vector = []

    for n in range(100):

        # determine location of previous whole beat
        prevBeat = np.floor(ck)

        # randomly choos next beat location
        nextBeat = prior.nextBeatLocation(ck)
        #nextBeat = 1.0

        # compute random jump parameter
        gamma = nextBeat-ck

        # compute tempo process state transition
        PHI = np.array(((1.,gamma),(0.,1.)),dtype=np.float64)
        theta = np.dot(PHI,theta)

        observation_vector.append(theta[0][0])

        if n == 0:
            for i in range(100):
                my_particles[i].step(theta[0][0])
            ck = nextBeat
            continue

        particle_list = [None] * 3000
        for i in range(100):
            for j in range(30):
                particle_list[i*30+j] = Particle(my_particles[i])

        print "theta =", theta
        for i in range(3000):
            particle_list[i].step(theta[0,0])

        # sort particles in descending order
        particle_list_unsorted = []
        for i in range(3000):
            particle_list_unsorted.append((particle_list[i],particle_list[i].wk))
        particle_list_sorted = sorted(particle_list_unsorted,
            key=lambda p: p[1], reverse=True)

        # god mode
        particle_list_unsorted002 = []
        for i in range(3000):
            particle_list_unsorted002.append((particle_list[i],np.sum(np.abs(particle_list[i].theta-theta))))
        particle_list_sorted002 = sorted(particle_list_unsorted002,
            key=lambda p: p[1], reverse=False)

        # weight scaling factor
        Wk = np.float64(0)
        for i in range(100):
            Wk = Wk + particle_list_sorted[i][0].wk

        # apply weight scaling
        Neff = np.float64(0)
        if Wk > np.sqrt(eps):
            for i in range(100):
                p = particle_list_sorted[i][0]
                p.wk = p.wk / Wk
                Neff = Neff + (p.wk**2)
            Neff = 1.0 / Neff

        theta_mmse = np.array(((0.,),(0.,)))
        for i in range(100):
            theta_mmse = theta_mmse + (particle_list_sorted[i][0].wk * 
                particle_list_sorted[i][0].theta)

        print 'theta_mmse =', theta_mmse, ", ck =", nextBeat, ", Wk =", Wk, ", Neff =", Neff

        for i in range(10):
            p = particle_list_sorted002[i][0]
            print "i =", i, ", wk =", p.wk, ", p.theta =", p.theta.flatten(), ", p.theta_est =", p.theta_est.flatten(), ", (a,b,c) =", (p.a,p.b,p.c), ", ck =", p.ck
            p = particle_list_sorted[i][0]
            print "*** i =", i, ", wk =", p.wk, ", p.theta =", p.theta.flatten(), ", p.theta_est =", p.theta_est.flatten(), ", (a,b,c) =", (p.a,p.b,p.c), ", ck =", p.ck

        if n > 29:
            raw_input()

        # draw next state from prior distribution
        #randomState = prior.mvnrnd(self.xkk,self.Pkk)
        #self.theta = randomState
        #print "  self.theta =", self.theta

        my_particles = [None]*100

        for i in range(100):
            my_particles[i] = particle_list_sorted[i][0]

        if Neff < 0.1:

            print "!!!!!!!!!!!!!!!!!!! Resample"

            for i in range(100):
               p = Particle()
               p.step(observation_vector[-1])
               my_particles[i] = p
        
        # update quantized score location
        ck = nextBeat

