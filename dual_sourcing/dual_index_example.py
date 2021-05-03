import numpy as np
import matplotlib.pyplot as plt
from lib.dual_sourcing import *

np.random.seed(10)

ce = 20
cr = 0
le = 0
lr = 2
ze = 100
h = 5
b = 495
T = 100

samples = 4000
Delta_arr = [0,1,2,3,4,5,6]

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

print("average cost (dual index):", np.mean(S.total_cost))

plt.figure()
plt.plot(S.inventory, '-o', label = r"inventory")
plt.plot(S.inventory_position_e, '-o', label = r"inventory position (e)")
plt.plot(S.inventory_position_r, '-o', label = r"inventory position (r)")
plt.plot(S.demand, '-o', label = r"demand")
plt.plot(S.qe, '-o', label = r"expedited order")
plt.plot(S.qr, '-o', label = r"regular order")
plt.xlabel(r"time")
plt.ylabel(r"value")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(S.cost, '-o', label = r"cost")
plt.xlabel(r"time")
plt.ylabel(r"cost")
plt.legend(loc = 4, ncol = 3)
plt.tight_layout()
plt.show()
