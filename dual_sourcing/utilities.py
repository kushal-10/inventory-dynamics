import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from dual_sourcing.lib.dual_sourcing import dual_index_ze_Delta, DualSourcingModel


def sample_trajectories(n_trajectories,
                        optimization_samples = 100,
                        seed= 1,
                        ce = 20,
                        cr = 0,
                        le = 0,
                        lr = 2,
                        ze = 100,
                        h = 5,
                        b = 495,
                        T = 100,
                        zr=100):
                        
    np.random.seed(seed)
    Delta_arr = [0,1,2,3,4,5,6]
    optimal_ze, optimal_Delta = dual_index_ze_Delta(optimization_samples,
	                                            Delta_arr,
	                                            ce,
	                                            cr,
	                                            le,
	                                            lr,
	                                            h,
	                                            b,
	                                            T,
	                                            ze)
    # each trajectory consists of T timesteps
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3])
    qr_trajectories = torch.zeros([n_trajectories, T+1+lr])
    qe_trajectories  = torch.zeros([n_trajectories, T+1+le])
    
    for i in range(n_trajectories):
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
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        qe_trajectories[i, :] = qe
        qr_trajectories[i, :] = qr
        state_trajectories[i, :, 2] = c
        
    return state_trajectories, qr_trajectories, qe_trajectories
