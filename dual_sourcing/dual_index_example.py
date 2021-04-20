import numpy as np
import matplotlib.pyplot as plt
from lib.dual_sourcing import *

np.random.seed(10)

ce = 1020
cr = 1000
le = 0
lr = 2
ze = 100
h = 5
b = 495
T = 50

samples = 4000
Delta_arr = [0,1,2,3,4]

optimal_ze, optimal_Delta = dual_index_ze_Delta(samples,
                                                Delta_arr,
                                                ce, 
                                                cr, 
                                                le, 
                                                lr,
                                                h, 
                                                b, 
                                                T,
                                                ze)

S = DualSourcingModel(ce=ce, 
                      cr=cr, 
                      le=le, 
                      lr=lr, 
                      h=h, 
                      b=b,
                      T=T, 
                      I0=optimal_ze,
                      ze=optimal_ze,
                      Delta=optimal_Delta,
                      dual_index=True)

S.simulate()  

print("total cost (dual index):", S.total_cost)

plt.figure()
plt.plot(S.cost, '-o', label = r"cost")
plt.plot(S.inventory, '-o', label = r"inventory")
plt.plot(S.inventory_position_e, '-o', label = r"inventory position (e)")
plt.plot(S.inventory_position_r, '-o', label = r"inventory position (r)")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()
