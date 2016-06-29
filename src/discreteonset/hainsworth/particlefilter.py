import sys
import numpy as np
import particle
import prior

eps = sys.float_info.epsilon

class ParticleFilter(object):
    def __init__(self):
    
        # ParticleFilter properties
        self.NumParticles = 100
        self.NumEvolutions = 30

        # ParticleFilter protected properties
        self._particle_list = None
        self._isSetup = False

    def step(self,X):
        
        if not self._isSetup:
            self.setupImpl()

        # compute number of particle states that we will
        # be tracking
        NumStates = self.NumParticles * self.NumEvolutions

        particle_states = [None] * self.NumParticles
        for k in range(self.NumParticles):

            # get particle p(k)
            pk = self._particle_list[k]

            # allocate memory for self.NumEvolutions set of
            # new particle locations for particle p(k)
            particle_states[k] = [None] * self.NumEvolutions

            # propagate particle k to a set s = {1,...,S} of new
            # locations
            for s in range(self.NumEvolutions):
    
                # copy particle p(k) to new particle p(k,s)
                ps = Particle(pk)

                # propagate particle p(k,s) to new location,
                # evaluate the new weight w(k,s) and pick new
                # state for particle
                ps.step(X)

                # add particle p(k,s) to list of particle states
                particle_states[k][s] = ps

        # compute global particle weights
        wk = np.empty((NumStates,))
        for i in range(self.NumParticles):
            for j in range(self.NumEvolutions):
                wk[i*self.NumEvolutions+j] = particle_states[i][j].wk

        # check to ensure at least one of the particles has non-zero
        # weight
        if np.abs(np.sum(wk)) > np.sqrt(eps):

            # normalize particle weights
            for i in range(self.NumParticles):
                for j in range(self.NumEvolutions):
                    particle_states[i][j].wk = (particle_states[i][j].wk / 
                        np.sum(wk))
    
            # update weight list with new particle weight values
            for i in range(self.NumParticles):
                for j in range(self.NumEvolutions):
                    wk[i*self.NumEvolutions+j] = particle_states[i][j].wk

        # sort particles in descending order
        particle_list_unsorted = []
        for i in range(self.NumParticles):
            for j in range(self.NumEvolutions):
                particle_list_unsorted.append((particle_states[i][j],
                    particle_states[i][j].wk))
        particle_list_sorted = sorted(particle_list_unsorted,
            key=lambda p: p[1], reverse=True)

        # select self.NumParticles best particles
        for i in range(self.NumParticles):
            self._particle_list[i] = particle_list_sorted[i][0]

    def setupImpl(self):

        # allocate memory for particles
        self._particle_list = [None] * self.NumParticles

        # initialize particles
        for i in range(self.NumParticles):
            self._particle_list[i] = particle.Particle()

        # indicate that object has been setup
        self._isSetup = True
