import numpy as np
from numpy.random import rand
from numpy.random import randint

def objective_function(x):
    return (x[0]** 2+ x[1]- 11)** 2+ (x[0]+ x[1]** 2- 7)** 2

class genetic_algorithm:
    def __init__(self, objective, n_pop, n_bits, r_cross, r_mut):
        self.objective= objective
        self.n_variable_bits= n_bits
        self.r_cross= r_cross
        self.r_mut= r_mut
        self.n_bits= n_bits
        self.n_pop= n_pop
    
    # decode bitstring to numbers
    def decoder(self, bounds, n_bits, bitstring):
        decode= []
        largest= 2** n_bits
        for i in range(len(bounds)):
            # extract the substring
            start, end= i* n_bits, (i* n_bits)+ n_bits
            substring= bitstring[start: end]
            # convert bitstring to a string of chars
            chars= "".join([str(s) for s in substring])
            # convert string to integer - 2 進制
            interger= int(chars, 2)
            # scale integer to desired range - 2 進制轉數字型態
            value= bounds[i][0]+ (interger/ largest)* (bounds[i][1]- bounds[i][0]) 
            # store
            decode.append(value)
        return decode

    # tournament selection
    def selection(self, pop, scores, k= 3):
        # first random selection
        selection_ix= randint(len(pop))
        for ix in randint(0, len(pop), k- 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix= ix
        return pop[selection_ix]

    # crossover two parents to create two children
    def crossover(self, p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2= p1.copy(), p2.copy()
        # check for recombination
        if rand()< r_cross:
            # select crossover point that is not on the end of the string
            pt= randint(1, len(p1)- 2)
            # perform crossover
            c1= p1[: pt]+ p2[pt: ]
            c2= p2[: pt]+ p1[pt: ]
        return [c1, c2]
    
    # mutation operator
    def mutation(self, bitstring, r_mut):
        for i in range(0, len(bitstring)):
            # check for a mutation
            if rand()< r_mut:
                # flip the bit
                self.bitstring[i]= 1- self.bitstring[i]

    # main genetic algorithm process
    def main_run(self, bounds, n_iter):
        self.bounds= bounds
        self.n_iter= n_iter
        # initial population of random bitstring
        pop= [randint(0, 2, self.n_bits* len(bounds)).tolist() for _ in range(self.n_pop)]
        # keep track of best solution
        best, best_val= 0, self.objective(self.decoder(self.bounds, self.n_bits, pop[0]))
        # enumerate generations
        for gen in range(0, self.n_iter):
            # decode population
            decoded = [self.decoder(bounds, n_bits, p) for p in pop]
            # evaluate all candidates in the population
            scores = [self.objective(d) for d in decoded]
            # check for new best solution
            for i in range(0, self.n_pop):
                if scores[i]< best_val:
                    best, best_val= pop[i], scores[i]
                    print("-> %d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
            # select parents
            selected= [self.selection(pop, scores) for _ in range(0, self.n_pop)]
            # create the next generation
            children= []
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2= selected[i], selected[i+ 1]
                # crossover and mutation
                for c in self.crossover(p1, p2, self.r_cross):
                    self.bitstring= c
                    # mutation
                    self.mutation(self.bitstring, self.r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop= children
        return [best, best_val]


if __name__== "__main__":
    bounds= [[-5, 5], [-5, 5]]
    n_pop= 100
    n_bits= 16
    r_cross= 0.9
    r_mut= 1.0 / (float(n_bits) * len(bounds))
    n_iter= 1000

    GA= genetic_algorithm(objective= objective_function, n_pop= n_pop, n_bits= n_bits, r_cross= r_cross, r_mut= 1.0 / (float(n_bits) * len(bounds)))
    GA.main_run(bounds, n_iter)