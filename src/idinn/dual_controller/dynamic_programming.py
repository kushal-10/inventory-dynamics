from .base import BaseDualController
from itertools import product
import numpy as np
import torch
from numba import njit, types
from numba.typed import Dict, List


class DynamicProgrammingController(BaseDualController):
    def __init__(self) -> None:
        self.qf = None
        self.sourcing_model = None

    @staticmethod
    @njit
    def __vf_update(demand_prob, min_demand, max_demand, ce, h, b, state, vf, actions):
        """
        vf_update is a function that calculates a single value iteration update.
        """
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

    @staticmethod
    def __get_basestock_ub(exp_demand, lead_time, support, h, b):
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

        min_demand = int(sourcing_model.demand_generator.get_min_demand())
        max_demand = int(sourcing_model.demand_generator.get_max_demand())
        exp_demand = (max_demand + min_demand) / 2.0
        support = max_demand - min_demand
        h = sourcing_model.get_holding_cost()
        b = sourcing_model.get_shortage_cost()
        ce = sourcing_model.get_expedited_order_cost()
        le = sourcing_model.get_expedited_lead_time()
        lr = sourcing_model.get_regular_lead_time()

        base_e = DynamicProgrammingController.__get_basestock_ub(
            exp_demand=exp_demand, lead_time=le, support=support, h=h, b=b
        )
        base_r = DynamicProgrammingController.__get_basestock_ub(
            exp_demand=exp_demand, lead_time=lr, support=support, h=h, b=b
        )
        min_ip = int(min(base_r, base_e) - max_demand)
        max_ip = int(max(base_r, base_e))
        max_order = max_ip + min_ip
        dim_pipeline = lr - le - 1

        demand_prob = Dict.empty(key_type=types.float64, value_type=types.float64)
        demand_prob_ = sourcing_model.demand_generator.enumerate_support()
        for k, v in demand_prob_.items():
            demand_prob[k] = v

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

        all_values = np.zeros(max_iterations, dtype=float)
        these_values = np.zeros(len(states))
        iteration_arr = []
        value_arr = []
        qf = {}
        val = 0

        for iteration in range(max_iterations):
            # We first store each newly updated state
            for idx, state in enumerate(states):
                these_values[idx] = DynamicProgrammingController.__vf_update(
                    demand_prob, min_demand, max_demand, ce, h, b, state, vf, actions
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
                        qa = DynamicProgrammingController.__vf_update(
                            demand_prob,
                            min_demand,
                            max_demand,
                            ce,
                            h,
                            b,
                            state,
                            vf,
                            actions,
                        )[1]
                        if qa is not None:
                            qf[state] = qa
                    break

        self.qf = qf

    def predict(
        self, current_inventory, past_regular_orders, past_expedited_orders=None
    ):
        """
        past_expedited_orders is optional since expedited lead time is assumed to be 0 and the batch size is assumed to be 1.
        """
        regular_lead_time = self.sourcing_model.regular_lead_time
        first = current_inventory.squeeze() + past_regular_orders.squeeze()[-regular_lead_time]
        second = past_regular_orders.squeeze()[-regular_lead_time + 1 :]
        key = tuple([int(first)] + second.int().tolist())
        return self.qf[key]

    def reset(self):
        self.qf = None
        self.sourcing_model = None
