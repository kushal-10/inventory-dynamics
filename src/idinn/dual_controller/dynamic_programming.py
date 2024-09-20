from itertools import product
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict, List
from logging import info, debug, warning, error, critical

class DynamicProgrammingController:
    def __init__(self) -> None:
        self.qf = None
        self.sourcing_model = None

    # Make it private
    # https://www.geeksforgeeks.org/private-methods-in-python/
    @staticmethod    
    @njit
    def vf_update(demand, min_demand, max_demand, ce, h, b, state, vf, actions):
        """
        vf_update is a function that calculates a single value iteration update.
        """
        support = max_demand - min_demand
        # TODO: Use pdf to replace np.reapeat
        demand_prob = demand.enumerate_support()

        best_action = None
        best_cost = 10e9

        for qe, qr in actions:
            # Immediate cost of action
            cost = qe * ce
            # Partial state update
            ip_e = state[0] + qe + state[1]
            pipeline = state[2:]

            for dem in range(int(min_demand), int(max_demand) + 1):
                ipe_new = int(ip_e) - dem
                this_state = (ipe_new,) + pipeline + (qr,)
                # If we jump to a state that is not in our list, we are not playing optimal,
                # so we can safely get out of here.
                if (this_state not in vf) or (vf[this_state] > 10e9 - 1.0):
                    cost = 10e9
                    break
                else:
                    # Careful: qr(t-1) has not arrived yet, we need to take it out
                    inv_on_hand = ipe_new - state[1]
                    inv_cost = inv_on_hand * h if inv_on_hand >= 0 else -inv_on_hand * b
                    cost += demand_prob[dem] * (inv_cost + vf[this_state])
            if cost < best_cost:
                best_cost = cost
                best_action = (qr, qe)
        return best_cost, best_action

    def get_basestock_ub(self, exp_demand, lead_time, support, h, b):
        """
        Get an upper bound on the single-source basestock level based on
        Hoeffding's inequality.
        """
        n = lead_time + 1
        base_stock_ub = n * exp_demand + support * np.sqrt(n * np.log(1 + b / h) / 2)
        return np.ceil(base_stock_ub)
    
    def fit(self, sourcing_model, max_iterations=1000000, tolerance=10e-8):
        # TODO: Check demand is uniform distributed

        # Log when sourcing_model's lead time do not correspond to the one in the controller
        # if self.regular_lead_time is None:
        #     info("Model starts training.")
        # else:
        # if sourcing_model.regular_lead_time != self.regular_lead_time:
        #     info("Regular lead time does not match the controller's previous specified regular lead time.")
        # if sourcing_model.expedited_lead_time != self.expedited_lead_time:
        #     info("Expedited lead time does not match the controller's previous specified expedited lead time.")
        # TODO: Check if the expedited_lead_time is 0
        # if sourcing_model.expedited_lead_time > 0:
        #     lr = sourcing_model.regular_lead_time - sourcing_model.expedited_lead_time
        #     le = 0
        self.sourcing_model = sourcing_model

        min_demand = int(sourcing_model.demand_generator.distribution.low)
        max_demand = int(sourcing_model.demand_generator.distribution.high - 1)
        exp_demand = (max_demand + min_demand) / 2.0
        support = max_demand - min_demand
        h = sourcing_model.holding_cost
        b = sourcing_model.shortage_cost
        ce = sourcing_model.expedited_order_cost
        le = sourcing_model.expedited_lead_time
        lr = sourcing_model.regular_lead_time

        base_e = self.get_basestock_ub(
            exp_demand=exp_demand, lead_time=le, support=support, h=h, b=b
        )
        base_r = self.get_basestock_ub(
            exp_demand=exp_demand, lead_time=lr, support=support, h=h, b=b
        )
        min_ip = int(min(base_r, base_e) - max_demand)
        max_ip = int(max(base_r, base_e))
        max_order = max_ip + min_ip
        dim_pipeline = lr - le - 1

        states_ = list(
            product(
                range(-min_ip, max_ip + 1),
                *(range(int(max_demand) + 1),) * int(dim_pipeline),
            )
        )
        states = List()
        for state in states_:
            states.append(state)

        actions_ = list(product(range(max_order), range(max_order)))
        actions = List()
        for action in actions_:
            actions.append(action)

        # Values can be initiated arbitrarily
        vals = np.repeat(1.0, len(states))
        vf_ = dict(zip(states, vals))
        vf = Dict.empty(
            key_type=types.UniTuple(types.int64, lr), value_type=types.float64
        )
        for k, v in vf_.items():
            vf[k] = v

        demand_prob = Dict.empty(key_type=types.float64, value_type=types.float64)
        demand_prob_ = dict(
            zip(
                np.arange(min_demand, max_demand + 1),
                np.repeat(1.0 / (support + 1.0), int(support + 1)),
            )
        )
        for k, v in demand_prob_.items():
            demand_prob[k] = v

        all_values = np.zeros(max_iterations, dtype=float)
        these_values = np.zeros(len(states))
        iteration_arr = []
        value_arr = []
        qf = {}
        val = 0

        for iteration in range(max_iterations):
            # We first store each newly updated state
            for idx, state in enumerate(states):
                these_values[idx] = DynamicProgrammingController.vf_update(
                    min_demand, max_demand, ce, h, b, state, vf, actions
                )[0]
            # After the minimum for each state has been calculated, we update the states
            # If done in the same loop, converge sucks even more
            for idx, state in enumerate(states):
                vf[state] = these_values[idx]

            iter_vals = np.array([val for val in vf.values() if val < 10e8])
            this_average = np.mean(iter_vals)
            
            val = this_average / (iteration + 1)
            all_values[iteration] = val

            if iteration > 1 and iteration % 100 == 0:
                iteration_arr.append(iteration)
                value_arr.append(all_values[iteration])
                delta = all_values[iteration - 1] - all_values[iteration]
                print(f"iteration: {iteration} delta: {delta}")
                if delta <= tolerance:
                    for state in states:
                        qa = DynamicProgrammingController.vf_update(min_demand, max_demand, ce, h, b, state, vf, actions)[1]

                        if qa:
                            qf[state] = qa
                    break
                    
        self.qf = qf

    def evaluate(self, current_inventory, past_regular_orders, past_expedited_orders, sourcing_model = None):
        pass

    def predict(self, current_inventory, past_regular_orders, past_expedited_orders=None, sourcing_model = None):
        regular_lead_time = self.sourcing_model.regular_lead_time

        first = current_inventory + past_regular_orders[-regular_lead_time]
        second = past_regular_orders[-regular_lead_time+1:]
        key = tuple([first]+second)
        
        return self.qf[key]
