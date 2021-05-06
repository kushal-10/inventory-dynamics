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
T = 5000

u1_arr = np.arange(4,8)
u2_arr = np.arange(8,15)
u3_arr = np.arange(4)

optimal_u1, optimal_u2, optimal_u3 = capped_dual_index_parameters(u1_arr,
                                                                  u2_arr,
                                                                  u3_arr,
                                                                  ce, 
                                                                  cr, 
                                                                  le, 
                                                                  lr,
                                                                  h, 
                                                                  b, 
                                                                  T)

T = 5000
S = DualSourcingModel(ce=ce, 
                      cr=cr, 
                      le=le, 
                      lr=lr, 
                      h=h, 
                      b=b,
                      T=T, 
                      I0=0,
                      u1=optimal_u1,
                      u2=optimal_u2,
                      u3=optimal_u3,
                      capped_dual_index=True)

S.simulate()  

print("average cost (capped dual index):", S.total_cost/T)

plt.figure()
plt.plot(S.inventory, '-o', label = r"inventory")
plt.plot(S.inventory_position, '-o', label = r"inventory position")
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
