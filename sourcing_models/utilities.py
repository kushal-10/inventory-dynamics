import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from sourcing_models.lib.dual_sourcing import single_index_zr_Delta, \
					      dual_index_ze_Delta, DualSourcingModel, \
                          capped_dual_index_parameters

from sourcing_models.lib.single_sourcing import SingleSourcingModel
    
def sample_trajectories_single_index(n_trajectories,
                                     optimization_samples = 100,
                                     seed = 1,
                                     ce = 20,
                                     cr = 0,
                                     le = 0,
                                     lr = 2,
                                     h = 5,
                                     b = 495,
                                     T = 100,
                                     zr = 100):
    
    Delta_arr = np.arange(0,5)

    optimal_zr, optimal_Delta = single_index_zr_Delta(optimization_samples,
                                                      Delta_arr,
                                                      ce, 
                                                      cr, 
                                                      le, 
                                                      lr,
                                                      h, 
                                                      b, 
                                                      2000,
                                                      zr)

    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le])
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr])
        
    state_trajectories = torch.zeros([n_trajectories, T+1, 3])
    
    for i in range(n_trajectories):
        S = DualSourcingModel(ce=ce, 
                              cr=cr, 
                              le=le, 
                              lr=lr, 
                              h=h, 
                              b=b,
                              T=T, 
                              I0=optimal_zr,
                              zr=optimal_zr,
                              Delta=optimal_Delta,
                              single_index=True)

        S.simulate()

        I = torch.tensor(S.inventory)
        D = torch.tensor(S.demand)
        qe = torch.tensor(S.qe)
        qr = torch.tensor(S.qr)
        c = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        qe_trajectories[i, :] = qe
        qr_trajectories[i, :] = qr
        state_trajectories[i, :, 2] = c
    
    return state_trajectories, qr_trajectories, qe_trajectories

def sample_trajectories_dual_index(n_trajectories,
                                   optimization_samples = 100,
                                   seed = 1,
                                   ce = 20,
                                   cr = 0,
                                   le = 0,
                                   lr = 2,
                                   ze = 100,
                                   h = 5,
                                   b = 495,
                                   T = 100,
                                   zr = 100):
                        
    np.random.seed(seed)
    Delta_arr = np.arange(0,7)
    optimal_ze, optimal_Delta = dual_index_ze_Delta(optimization_samples,
	                                            Delta_arr,
	                                            ce,
	                                            cr,
	                                            le,
	                                            lr,
	                                            h,
	                                            b,
	                                            2000,
	                                            ze)
    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le])
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr])
        
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3])
    
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

def sample_trajectories_capped_dual_index(n_trajectories,
                                          seed = 1,
                                          ce = 20,
                                          cr = 0,
                                          le = 0,
                                          lr = 2,
                                          h = 5,
                                          b = 495,
                                          T = 100):
                        
    np.random.seed(seed)
    u1_arr = np.arange(1,5)
    u2_arr = np.arange(8,12)
    u3_arr = np.arange(5)

    optimal_u1, optimal_u2, optimal_u3 = capped_dual_index_parameters(u1_arr,
                                                                      u2_arr,
                                                                      u3_arr,
                                                                      ce, 
                                                                      cr, 
                                                                      le, 
                                                                      lr,
                                                                      h, 
                                                                      b, 
                                                                      50000)
    # each trajectory consists of T timesteps
    if le == 0:
        qe_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qe_trajectories = torch.zeros([n_trajectories, T+le])
    
    if lr == 0:
        qr_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        qr_trajectories = torch.zeros([n_trajectories, T+lr])
        
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3])
    
    for i in range(n_trajectories):
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

def sample_trajectories_single_sourcing(n_trajectories,
                                        seed = 1,
                                        l = 2,
                                        h = 5,
                                        b = 495,
                                        T = 100,
                                        r = 10):
                        
    np.random.seed(seed)
    
    # each trajectory consists of T timesteps
    if l == 0:
        q_trajectories = torch.zeros([n_trajectories, T+1])
    else:
        q_trajectories = torch.zeros([n_trajectories, T+l])
        
    state_trajectories  = torch.zeros([n_trajectories, T+1, 3])
    
    for i in range(n_trajectories):
        S = SingleSourcingModel(l=l, 
                                h=h, 
                                b=b,
                                T=T, 
                                r=r,
                                I0=0,
                                optimal_base_stock=False)

        S.simulate()
        
        I  = torch.tensor(S.inventory)
        D  = torch.tensor(S.demand)
        q = torch.tensor(S.q)
        c  = torch.tensor(S.cost)
        state_trajectories[i, :, 0] = I
        state_trajectories[i, :, 1] = D
        q_trajectories[i, :] = q
        state_trajectories[i, :, 2] = c
        
    return state_trajectories, q_trajectories
