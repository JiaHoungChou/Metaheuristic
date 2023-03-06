from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around

def objective_function(x):
    return (x[0]** 2+ x[1]- 11)** 2+ (x[0]+ x[1]** 2- 7)** 2

class differential_evaluation_algorithm:
    def __init__(self, objective_function, F: float, cr: float):
        if cr > 1 or F > 1:
            raise ValueError("the cr variable should less than 1, it's a probability for binomail crossoever")
        self.F= F
        self.cr= cr
        self.objective_function= objective_function
    def mutation(self, x, F: float):
        return x[0]+ F* (x[1]- x[2])
    def check_bounds(self, mutated, bounds):
        mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
        return mutated_bound
    def crossover(self, mutated, target, dims, cr):
        # generate a varaible from unifrom distribution
        p= rand(dims)
        # generate trail vector by binomial crossover
        trail= [mutated[i] if p[i]< cr else target[i] for i in range(dims)]
        return trail
    def main_run(self, pop_size, bounds, iter):
        if pop_size < 4:
            raise ValueError("the parameter pop should be larger than 4")
        self.pop_size= pop_size
        self.bounds= bounds
        self.iteration= iter
        # initialise population of candidate solutions randomly within the specified bounds
        # bounds[:, 0]: the first parameter bound's lower boundary 
        # bounds[:, 1]: the first parameter bound's upper boundary
        pop= bounds[:, 0]+ (rand(pop_size, len(bounds))* (bounds[:, 1]- bounds[:, 0]))
        # evaluate initial population of candidate solutions randomly within the specified bounds
        obj_all= [self.objective_function(ind) for ind in pop]
        # find the best performing vector of initial population
        best_vector= pop[argmin(obj_all)]
        best_obj= min(obj_all)
        prev_obj= best_obj
        for i in range(0, iter):
            # iterate over all candidate solutions
            for j in range(0, pop_size):
                # choose three candidates, a, b and c, that are not the current one
                candidates= [candidate for candidate in range(pop_size) if candidate != j]
                a, b, c= pop[choice(candidates, 3, replace= False)]
                # perfrom mutation
                mutated= self.mutation([a, b, c], self.F)
                # check that lower and upper bounds are retained after mutation
                mutated= self.check_bounds(mutated, bounds)
                # perform crossover
                trail= self.crossover(mutated, pop[j], len(bounds), self.cr)
                # compute objective function value for target vector
                obj_target= self.objective_function(pop[j])
                # compute objective function value for trial vector
                obj_trial= self.objective_function(trail)
                # perform selection
                if obj_trial< obj_target:
                    # replace the target vector with the trial vector
                    pop[j]= trail
                    # store the new objective function value
                    obj_all[j]= obj_trial
            # find the best performing vector at each iteration
            best_obj= min(obj_all)
            # store the lowest objective function value
            if best_obj < prev_obj:
                best_vector= pop[argmin(obj_all)]
                prev_obj= best_obj
                # report progress at each iteration
                print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
        return [best_vector, best_obj]

if __name__== "__main__":
    # define population size
    pop_size = 100
    # define lower and upper bounds for every dimension
    bounds = asarray([(-5.0, 5.0), (-5.0, 5.0)])
    # define number of iterations
    iter = 1000
    # define scale factor for mutationv
    F = 0.5
    # define crossover rate for recombination
    cr = 0.7

    DE_algorithm= differential_evaluation_algorithm(objective_function, F, cr)
    DE_algorithm.main_run(pop_size, bounds, iter)