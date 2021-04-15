import numpy as np
import matplotlib.pyplot as plt
from lib.dual_sourcing import *

np.random.seed(10)

# parameters are taken from Table 2 from
# Scheller-Wolf, A., Veeraraghavan, S., & van Houtum, G. J. (2007). 
# Effective dual sourcing with a single index policy. 
# Working Paper, Tepper School of Business, 
# Carnegie Mellon University, Pittsburgh.)
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

S = DualSourcingModel(ce, 
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

S.simulate()  

print("total cost (single index policy):", S.total_cost)

plt.figure()
plt.plot(S.cost, '-o', label = r"cost")
plt.plot(S.inventory, '-o', label = r"inventory")
plt.plot(S.inventory_position, '-o', label = r"inventory position")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()