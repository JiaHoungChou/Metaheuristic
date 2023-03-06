import numpy as np

def objective_function(x):
    return (x[0]** 2+ x[1]- 11)** 2+ (x[0]+ x[1]** 2- 7)** 2

if __name__== "__main__":

    class ant_colony_optimization:
        def __init__(self, objective, n_pop):
            self.objective_function= objective
            self.n_pop= n_pop

        
        def propability_computation(self, tau1, tau2):
            return tau1/ (tau1+ tau2), tau2/ (tau1+ tau2)