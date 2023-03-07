from random import random
import numpy as np
import math
import matplotlib.pyplot as plt
np.random.seed(123)

def objective_function(x1, x2):
    return (x1** 2+ x2- 11)** 2+ (x1+ x2** 2- 7)** 2

def SA(fun, bound_x1: list, bound_x2: list, iteration):
    x1= np.random.uniform(low= bound_x1[0], high= bound_x1[1])
    x2= np.random.uniform(low= bound_x2[0], high= bound_x2[1])
    TMax= 1000; Tmin= 0.01; R_ratio= 0.1
    while TMax>= Tmin:
        y_converge_curve= []
        for i in range(0, iteration):
            e_c= fun(x1, x2)
            while True:
                # make a move
                x1_new= x1+ np.random.uniform(low=  -0.1, high= 0.1)* TMax
                x2_new= x2+ np.random.uniform(low=  -0.1, high= 0.1)* TMax
                if (bound_x1[0]<= x1_new and x1_new <= bound_x1[1]) and (bound_x2[0]<= x2_new and x2_new <= bound_x2[1]):
                    break
            e_n= fun(x1_new, x2_new)
            delta_e= e_n- e_c
            # good move
            if delta_e <= 0:
                x1, x2= x1_new, x2_new
            # bad move
            else:
                p= math.exp(-(e_n- e_c)/ TMax)
                r= np.random.uniform(0, 1, 1)
                # accept the bad move
                if (p> r):
                    x1, x2= x1_new, x2_new
                else:
                    x1, x2= x1, x2
            y_converge_curve.append(e_c)
        TMax= TMax* R_ratio
    
    plt.figure(num= 1, figsize= (13, 3))
    plt.title("y converge curve", fontsize= 10)
    plt.plot(np.arange(0, len(y_converge_curve)), y_converge_curve, color= "darkblue", label= "y")
    plt.scatter(np.arange(0, len(y_converge_curve)), y_converge_curve, color= "black")
    plt.grid(True)
    plt.show()
    return round(x1, 4), round(x2, 4), round(fun(x1, x2), 4)

if __name__== "__main__":
    opt_x1, opt_x2, opt_y= SA(fun= objective_function, bound_x1= [-5, 5], bound_x2= [-5, 5], iteration= 10000)
    print("-----> optimal x1: {}// optimal x2: {}// optimal y: {}".format(opt_x1, opt_x2, opt_y))