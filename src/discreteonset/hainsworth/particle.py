import sys
import numpy as np
import prior

eps = sys.float_info.epsilon

class Particle(object):
    def __init__(self):

        # number of internal states
        self.NumStates = 30

        # initialize particle
        self.__k = 1
        self.theta = np.array(((0.,),(0.,)),dtype=np.float64)
        self.ck = 0.

        # initialize Kalman filter properties
        self.xkk = np.array(((0.,),(1.,)),dtype=np.float64)
        self.Pkk = np.eye(2)

        self.H = np.array(((1.,0.),),dtype=np.float64)

    def step(self,observation,beatlocation=None):

        # determine location of previous whole beat
        prevBeat = np.floor(self.ck)

        # create array for set s= {1,...,S} of new
        # score locations c(s,k)
        Ck = [None] * self.NumStates
        Pk = [None] * self.NumStates
        wk = [None] * self.NumStates

        for s in range(self.NumStates):

            # randomly choos next beat location
            nextBeat = prior.nextBeatLocation()
            while (prevBeat+nextBeat)-self.ck < np.sqrt(
                eps): nextBeat = prior.nextBeatLocation()

            print "nextBeat[%d] =" % (s,) , nextBeat

            # propagate ck to new state
            Ck[s] = prevBeat+nextBeat

            # calculate random jump paramater gamma
            gamma = Ck[s]-self.ck

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
                0.2*Q)

            # update
            y = observation - np.dot(self.H,x_est)
            S = (np.dot(self.H,np.dot(P_est,self.H.T)) + 
                0.02*np.eye(1))
            K = np.dot(np.dot(P_est,self.H.T),np.linalg.inv(S))
            Ck[s] = x_est + np.dot(K,y)
            Pk[s] = np.dot(np.eye(2)-np.dot(K,self.H),P_est)

            '''
            # evaluate new weight
            if self.__k == 1:
                wk[s] = ((prior.transitionPDF(Ck[s])*
                    prior.observationPDF(observation,Ck[s][0][0])) /
                    prior.importancePDF(Pk[s]))
            else:
                wk[s] = self.wk[-1] * ((prior.transitionPDF(Ck[s])*
                    prior.observationPDF(observation,Ck[s][0][0])) /
                    prior.importancePDF(Pk[s]))
            '''

        '''
        q = ((np.linalg.det(2.0*np.pi*self.Pkk)**(-0.5)) * 
            np.exp(-0.5*np.dot(np.dot(np.transpose(theta-self.xkk),
            self.Pkk),theta-self.xkk)))
        '''

        raw_input()

        # draw next state from prior distribution
        randomState = prior.mvnrnd(self.xkk,self.Pkk)

        self.theta = randomState

        print "  self.theta =", self.theta

if __name__ == '__main__':

    particle = Particle()

    tempo = (60.0 / (140. * np.random.rand() + 60.))

    theta = np.array(((0.,),(tempo,)))
    particle.theta = theta

    for n in range(100):
        nextbeat = prior.nextBeatLocation()
        PHI = np.array(((1.,nextbeat),(0.,1.)),dtype=np.float64)
        theta = np.dot(PHI,theta)
        print "theta =", theta
        particle.step(theta[0,0],nextbeat)
