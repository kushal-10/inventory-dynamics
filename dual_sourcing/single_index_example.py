import numpy as np
from lib.dual_sourcing import *

np.random.seed(10)

ce = 1020
cr = 1000
le = 0
lr = 2
zr = 100
h = 5
b = 495
T = 50

samples = 4000
Delta_arr = [0,1,2,3,4]

optimal_z_r, optimal_Delta = single_index_zr_Delta(samples,
                                                   Delta_arr,
                                                   ce, 
                                                   cr, 
                                                   le, 
                                                   lr,
                                                   h, 
                                                   b, 
                                                   T,
                                                   zr)

S1 = DualSourcingModel(ce, 
                       cr, 
                       le, 
                       lr, 
                       h, 
                       b,
                       T, 
                       optimal_z_r,
                       optimal_z_r,
                       optimal_Delta,
                       single_index=True)

S1.simulate()  

print("total cost (order always):", S1.total_cost)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(S1.cost, '-o', label = r"cost s1")
plt.plot(S1.inventory, '-o', label = r"inventory s1")
plt.plot(S1.inventory_position, '-o', label = r"inventory position s1")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()