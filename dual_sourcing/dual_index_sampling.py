import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from dual_sourcing.lib.dual_sourcing import *

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
n_samples = 100

for i in range(n_samples):
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
    I  = torch.tensor(S.inventory)
    D  = torch.tensor(S.demand)
    qe = torch.tensor(S.qe)
    qr = torch.tensor(S.qr)
    c  = torch.tensor(S.cost)
    sample = dict(
        inventory = I,
        demand = D,
        exp_order = qe,
        re_order = qr,
        cost = c
    )
    os.makedirs('../data/ds_1', exist_ok=True)
    torch.save(sample, '../data/ds_1/sample_'+str(i)+'.pt')

