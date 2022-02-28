'''
Filename particle.py
Refer to wiki_doc: https://en.wikipedia.org/wiki/Particle_filter
Author Maxtom
'''
import bisect
import numpy as np
from random import randint, uniform
from operator import attrgetter
from src.util import log
from config import cfg
import pdb

class Particle(object):
    """ Particle class """
    def __init__(self, index, weight):
        self.index = index
        self.weight = weight
        self.track_residual = 0.0
    def __repr__(self):
        return repr((self.index, self.weight))
    def update_weighting(self, weight):
        """ Update Weighting """
        self.weight = weight
    def update_index(self, index):
        """ Update Index """
        self.index = index

class WeightedDistribution(object):
    """ Sampling from Distribution """
    def __init__(self, state):
        accum = 0.0
        self.state = [x for x in state if x.weight > 0]
        self.distribution = []
        for particle in self.state:
            accum += particle.weight
            self.distribution.append(accum)
    def pick(self):
        """ Pick up samples """
        return self.state[bisect.bisect_left(self.distribution, uniform(0, 1))]

class MSPF(object):
    """Multi scale particle filter"""
    def __init__(self, pf_num):
        self.particles = []
        self.pf_num = pf_num
        self.effective_particle = 0
        """ Particle initial """
        for _ in range(self.pf_num):
            index = randint(0, 10)
            self.particles.append(Particle(index, 1./self.pf_num))
    def particle_sampling(self, p_ids):
        """ Particle Sampling """
        if len(self.particles) != len(p_ids):
            log("Particles size is not equal to current estimation.", mode='r')
            return -1
        for particle, idx in zip(self.particles, p_ids):
            particle.update_index(idx)
        return 0
    def particle_status(self):
        """ Print current particle status """
        for particle in self.particles:
            print ("Particle {} with weighting {}".format(particle.index,\
                                                        particle.weight))
    def best_estimate(self):
        # Weight sorting
        self.particles = sorted(self.particles, \
                                key=attrgetter('weight'), reverse=True)

        # Best matches
        return self.particles[0].index

    def updating(self, data, p_dist):
        """ Reweighting """
        # Merge particles
        data = np.array(data, dtype=[('index', int), ('weight', float)])
        data.sort(axis=0, order=['index'])
        particles = []
        log("Befor merge particle {}".format(len(self.particles)), mode='y')
        # Caculate the current particle efficience
        #sum_w = np.sum(data['weight'])
        #cur_weight = data['weight']/sum_w
        #cur_eff = sum(pow(cur_weight, 2))
        #cur_eff = 1./cur_eff/len(self.particles)
        #log("Current Effective Particles rate {}".format(cur_eff), mode='r')
        #pdb.set_trace()
        # TODO need to update
        for count in range(len(data)):
            idx = int(data[count][0])
            value = data[count][1]
            # TODO: Need to figure out better merge method
            if count == 0:
                particles.append(Particle(idx, value))
            else:
                if abs(particles[-1].index - idx) <= 3: #3 p_dist
                    particles[-1].weight += value
                else:
                    particles.append(Particle(idx, value))

        particles = sorted(particles, \
                                key=attrgetter('weight'), reverse=True)
        # Without particle merge
        #self.particles = particles[:int(len(particles)/2)]
        self.particles = particles
        log("After merge particle {}".format(len(self.particles)), mode='y')
        # Weighting normalization
        sum_w = sum(p.weight for p in self.particles) + 1e-10
        for particle in self.particles:
            particle.weight = particle.weight/sum_w
        # Compute an estimate of the effective number of particles
        self.effective_particle = self.compute_effective_particles()
        ef_pf_rate = self.effective_particle/len(self.particles)
        log("Effective Particles rate {}".format(ef_pf_rate), mode='g')
        return ef_pf_rate
    def compute_effective_particles(self):
        """ Effective Particles """
        effect_particle = sum(pow(p.weight, 2) for p in self.particles)
        return 1./effect_particle
    def re_sampling(self, scale, pf_num):
        """ Resample from high weighting particles """
        # weight sorting
        self.particles = sorted(self.particles, \
                                key=attrgetter('weight'), reverse=True)
        # resample from higher weighting particles
        new_particles = []
        dist = WeightedDistribution(self.particles)
        for _ in range(pf_num):
            particle = dist.pick()
            # TODO sample around near space
            index = self.sample_around_index(particle.index, scale)
            new_particles.append(Particle(index, 1./pf_num))
        self.particles = new_particles

    def generate_new_particles(self, index, scale, pf_shift=10):
        """ Generate new particles around index """
        new_particles = []
        pf_num = 2*pf_shift+1
        if pf_num==1:
            scale = 0
        for idx in range(-pf_shift, pf_shift+1):
            # TODO sample around near space
            #index = self.sample_around_index(index, scale)
            new_index = index + scale*idx
            new_particles.append(Particle(new_index, 1./pf_num))

        self.particles = new_particles

    def generate_new_particles_specific(self, index, scale, possible_path, possible_index): 
        """ Generate new particles around index on a specipfc sequence"""
        space_length = len(possible_index) #path should be the same
        # print(possible_index)
        
        new_particles = []
        pf_num = space_length/scale
        for idx in range(pf_num):
            # TODO sample around near space
            #index = self.sample_around_index(index, scale)
            new_index = possible_index[scale*idx][0]
            print(new_index)
            new_particles.append(Particle(new_index, 1./pf_num))

        self.particles = new_particles

    def sample_around_index(self, index, scale):
        """ Sample around indexes """
        index = randint(-int(scale/2), int(scale/2)) + index
        return index
