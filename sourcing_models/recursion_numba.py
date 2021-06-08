from collections import namedtuple
import numpy as np
import time
from sys import argv
from itertools import product
from numba import njit
from numba import types
from numba.typed import Dict

@njit
def vf_update(state, vf, demand_prob, actions, states):
    """
    Calculation of value iteration for a single update.
    state:   tuple of integers of length l=lf-le: [IPe, qr(t-l+1), ..., qr(t-1)]
    vf:      dictionary of value function per state

    The value iteration update is
    vf[state] <-- min_{(qe, qr)} {sum(s in states) prob(s| (qe, qr, state))*vf[s]}
    """

    best_action, best_cost = None, 10e9

    for qe, qr in actions:
        # Immediate cost of action
        cost = qe * ce
        # Partial state update
        ip_e = state[0] + qe + state[1]

        if state[2:]:
            pipeline = state[2:]
        else: 
            pipeline = qr

        for dem in range(d_min, d_max + 1):
            ipe_new = ip_e - dem
            # This works only for l = 2, need to work out the general case
            # this_state = (ipe_new, qr)
            # This should work for the general case
            if state[2:]:
                pass # this_state = (ipe_new, pipeline[0], qr) 
            else:
                this_state = (ipe_new, qr)
            # If we jump to a state that is not in our list, we are not playing optimal
            # so we can safely get out of here. 
            if this_state not in states:
                cost = 10e9
                break
            else:
                # Careful: qr(t-1) has not arrived yet, we need to take it out
                inv_on_hand = ipe_new - state[1]
                inv_cost = inv_on_hand * h if inv_on_hand >= 0 else -inv_on_hand * b
                cost += demand_prob[dem] * (inv_cost + vf[this_state])
        if cost < best_cost:
            best_cost = cost
            best_action = (qe, qr)

    #if not best_action:
        # If there is no best action it means we were left in a state we were not supposed 
        # to ever get there with optimal play; we can remove it
        #vf.pop(state, None)
        #states.remove(state)

    return best_cost, best_action

def main():
    """
    Value iteration function.
    
    Parameters: 
    filename (str): filename of parameter file.
    
    """
    # In problems where demand in [0, 4], the expedited inventory position is between -8 and 13
    # Note that some of the states should never be reached (the ones with high inventory and high qr)
    # If we land in such a state we will remove it
    dim_pipeline = lr - le - 1
    states = list(product(range(-8, 15 + 1), *(range(5 + 1),) * int(dim_pipeline)))
    # SW mention we never need to order more than max demand for any mode
    actions = list(product(range(5+1), range(5+1)))
    # Values can be initiated arbitrarily
    vals = np.repeat(1., len(states))
    vf_ = dict(zip(states,vals))
    
    vf = Dict.empty(key_type=types.UniTuple(types.int64, 2),value_type=types.float64)       
    for k, v in vf_.items():
        vf[k] = v 
        
    demand_prob_ = dict(zip(np.arange(d_min, d_max + 1), np.repeat(1 / (support+1), support+1)))

    demand_prob = Dict.empty(key_type=types.float64,value_type=types.float64)       
    for k, v in demand_prob_.items():
        demand_prob[k] = v
    
    max_iterations, tolerance, delta = 200000, 10e-8, 10.
    all_values = np.zeros(max_iterations)
    these_values = np.zeros(len(states))

    start_time = time.time()

    # Main value iteration loop
    for iteration in range(5000):
        # We first store each newly updated state
        
        for idx, state in enumerate(states):
            #print(idx, state, len(states))
            these_values[idx], best_action = vf_update(state, vf, demand_prob, actions, states)
        # After the minimum for each state has been calculated, we update the states
        # If done in the same loop, converge sucks even more
        for idx, state in enumerate(states):
            vf[state] = these_values[idx]

        this_average = np.mean([val for val in vf.values() if val < 10e8])
        
        all_values[iteration] = this_average/(iteration+1)

        if iteration > 1 and iteration % 100 == 0:
            print('iteration: %d average cost: %1.2f'%(iteration,all_values[iteration]))
            delta = all_values[iteration - 1] - all_values[iteration]
            if delta <= tolerance:
                break

    end_time = time.time() - start_time
    print('DP terminated after %1.2f seconds. Tolerance: %1.9f Iterations: %d'%(end_time,delta,iteration))


if __name__ == '__main__':
   ce = 20
   cr = 0
   le = 0
   lr = 2
   h = 5
   b = 495
   d_min = 0
   d_max = 4
   support = d_max - d_min
   main()
