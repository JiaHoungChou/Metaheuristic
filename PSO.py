import numpy as np
from numpy.random import rand
from numpy.random import randn
import matplotlib.pyplot as plt
from numpy import around

def objective_function(x):
    return (x[0]** 2+ x[1]- 11)** 2+ (x[0]+ x[1]** 2- 7)** 2

class particle_swarm_optimization:
    def __init__(self, objective_function, n_particles, c1, c2, w):
        self.objective_function= objective_function
        self.n_particles= n_particles
        self.c1= c1
        self.c2= c2
        self.w= w
    
    def main_run(self, bounds: list, iter: int):
        self.iteration= iter
        # n particles initialization
        particles= bounds[0][0]+ rand(len(bounds), self.n_particles)* (bounds[0][1]- bounds[0][0])
        if particles.shape[1] <= 1:
            raise ValueError("hyparameter n_particles should be larger than 1")
        
        pre_V= randn(len(bounds), self.n_particles)* 0.1

        # pbest location initialization
        pre_pbest= particles
        pre_pbest_obj= self.objective_function(pre_pbest)
        # gbest location initialization
        pre_gbest= pre_pbest[:, pre_pbest_obj.argmin()]
        pre_gbest_obj= pre_pbest_obj.min()

        # update pbest and gbest location by iteration
        for i in range(1, self.iteration+ 1):
            # set r as a random movement ratio
            r1, r2= np.random.rand(2)
            pre_V= self.w* pre_V+ self.c1* r1* (pre_pbest- particles)+ self.c2* r2* (pre_gbest.reshape(-1, 1)- particles)

            # update movement for the particles
            particles= particles+ pre_V
            new_obj= self.objective_function(particles)
            
            pre_pbest[:, (pre_pbest_obj>= new_obj)]= particles[:, (pre_pbest_obj>= new_obj)]
            pre_pbest_obj= np.array([pre_pbest_obj, new_obj]).min(axis= 0)

            pre_gbest= pre_pbest[:, pre_pbest_obj.argmin()]
            pre_gbest_obj= pre_pbest_obj.min()

            print('Iteration: %d f([%s]) = %.5f' % (i, around(pre_gbest, decimals= 2), pre_gbest_obj))
        
        best_point= around(pre_pbest.mean(axis= 1), decimals= 2)
        best_obj= new_obj.mean()
        return best_point, best_obj

if __name__== "__main__":
    n_particles= 100
    w= 0.8
    c1= c2= 0.1
    iteration= 100
    bounds= [[-5, 5], [-5, 5]]

    PSO= particle_swarm_optimization(objective_function, n_particles, c1, c2, w)
    PSO.main_run(bounds, iteration)