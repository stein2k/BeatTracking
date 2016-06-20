import prior

class Particle(object):
    def __init__(self):

        # initialize particle
        self.theta = None

        # initialize Kalman filter properties
        self.thetak = None
        self.P = np.eye(2)

    def setup(self):
        self.setupImpl()

    def step(self,observation):

        # determine time delta between beats
        gamma = prior.nextBeatLocation()

        # compose transition matrix
        PHI = np.array([[1.0,np.float64(gamma)],[0.,1.]])

        # propagate particle to a new location ck
        self.theta = np.dot(PHI,self.theta)

        # update covariance estimate using
        # Kalman filter

        # time update (prediction)
        theta_est = np.dot(PHI,self.thetak)
        P_est = (np.dot(PHI,np.dot(self.P,PHI.T)) +
            np.eye(2))

        # update
        y = observation - np.dot(self.H,theta_est)
        S = np.dot(self.H,np.dot(P_est,self.H.T)) + np.eye(1)
        K = np.dot(np.dot(P_est,self.H.T),np.linalg.inv(S))
        self.thetak = self.thetak + np.dot(K,y)
        self.P = np.dot(np.eye(2)-np.dot(K,self.H),P_est)

        #
        ((np.det(2.0*np.pi*self.P)**(-0.5)) *
            np.exp(-0.5*np.dot(np.dot(np.transpose(theta_est-self.theta),
            np.linalg.inv(self.P)),theta_est-self.theta)))

        # update
        self.theta = [prior.nextBeatLocation(),
            60.0/np.float64(np.random.rand]

    def setupImpl(self):

        # sample from initial prior distribution
        self.theta = np.array(((prior.nextBeatLocation(),),
            (60.0/(140.*np.random.rand()+60.),)))

        # compute initial weight
        self.weight = ((prior.stateTransitionPrior(self.theta[0]) *
            observationPrior()
