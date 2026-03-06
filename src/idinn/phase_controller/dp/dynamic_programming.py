import logging
from datetime import datetime
from itertools import product
from typing import Dict as TypingDict
from typing import List as TypingList
from typing import Optional, Tuple, Union, no_type_check

import numpy as np
import torch
from numba import njit, types  # type: ignore
from numba.typed import Dict, List
from tqdm import tqdm

from ...demand import UniformDemand
from ...sourcing_model import DualSourcingModel
from .base import BaseDPController

# Get root logger
logger = logging.getLogger()


class DynamicProgrammingController(BaseDPController):
    def __init__(self) -> None:
        self.sourcing_model = None
        self.qf = None
        self.vf = None
        logger.info("Initialized DynamicProgrammingController")

    @staticmethod
    @njit
    def _vf_update(
        demand_prob: TypingDict[int, float],
        min_demand: int,
        max_demand: int,
        ce: float,
        h: float,
        b: float,
        state: Tuple[int, ...],
        vf: TypingDict[Tuple[int, ...], float],
        actions: TypingList[Tuple[int, int, int]],
    ) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """
        Two-period cycle value iteration update.
        
        Even period: place (qr0, qe0). Odd period: place (qe1) only.
        State: (ip, pipeline[0], ..., pipeline[dim-1])
        """
        best_action = None
        best_cost = 10e9

        for qr0, qe0, qe1 in actions:

            # Immediate costs are fixed regardless of demand realizations
            immediate_cost = (qe0 + qe1) * ce
            expected_cost = 0.0
            valid = True

            for dem0 in range(int(min_demand), int(max_demand) + 1):
                if not valid:
                    break
                for dem1 in range(int(min_demand), int(max_demand) + 1):

                    # -------------------------------------------------------
                    # EVEN PERIOD TRANSITION
                    # qe0 arrives immediately (le=0), pipeline[0] also arrives
                    # qr0 enters the back of the pipeline
                    # -------------------------------------------------------
                    # Inventory position after qe0 and pipeline[0] arrive
                    ip_even = state[0] + qe0 + state[1]
                    pipeline_even = state[2:]  # remaining in-pipeline orders

                    # Inventory position after demand dem0 is realized
                    ipe_after_even = int(ip_even) - dem0

                    # State entering the odd period
                    # pipeline advances: pipeline_even moves up, qr0 joins the back
                    state_odd = (ipe_after_even,) + pipeline_even + (qr0,)

                    if (state_odd not in vf) or (vf[state_odd] > 10e9 - 1.0):
                        valid = False
                        break

                    # On-hand inventory after even period
                    inv_even = ipe_after_even - state[1]
                    inv_cost_even = inv_even * h if inv_even >= 0 else -inv_even * b

                    # -------------------------------------------------------
                    # ODD PERIOD TRANSITION
                    # qe1 arrives immediately (le=0), state_odd[1] also arrives
                    # No qr placed this period, so 0 enters the back of the pipeline
                    # -------------------------------------------------------
                    ip_odd = state_odd[0] + qe1 + state_odd[1]
                    pipeline_odd = state_odd[2:]  # remaining pipeline after even

                    ipe_after_odd = int(ip_odd) - dem1

                    # State entering the next even period
                    state_next = (ipe_after_odd,) + pipeline_odd + (0,)

                    if (state_next not in vf) or (vf[state_next] > 10e9 - 1.0):
                        valid = False
                        break

                    inv_odd = ipe_after_odd - state_odd[1]
                    inv_cost_odd = inv_odd * h if inv_odd >= 0 else -inv_odd * b

                    # -------------------------------------------------------
                    # Joint probability weighted cost for this demand realization
                    # -------------------------------------------------------
                    prob = demand_prob[dem0] * demand_prob[dem1]
                    expected_cost += prob * (inv_cost_even + inv_cost_odd + vf[state_next])

            if valid and (immediate_cost + expected_cost) < best_cost:
                best_cost = immediate_cost + expected_cost
                best_action = (qr0, qe0, qe1)

        return best_cost, best_action


    @staticmethod
    def _get_basestock_ub(
        exp_demand: float, lead_time: int, support: float, h: float, b: float
    ) -> float:
        """
        Get an upper bound on the single-source basestock level based on
        Hoeffding's inequality.
        """
        n = lead_time + 1
        base_stock_ub = n * exp_demand + support * np.sqrt(n * np.log(1 + b / h) / 2)
        return np.ceil(base_stock_ub)

    @no_type_check
    def fit(
        self,
        sourcing_model: DualSourcingModel,
        max_iterations: int = 1000000,
        tolerance: float = 10e-8,
        validation_freq: int = 100,
        log_freq: int = 100,
    ) -> None:
        """
        Fit the controller to the given sourcing model.

        Parameters
        ----------
        sourcing_model : DualSourcingModel
            The sourcing model to fit the controller to.
        max_iterations : int, default is 1000000
            Specifies the maximum number of iterations to run.
        tolerance : float, default is 10e-8
            Specifies the tolerance to check if the value function has converged.
        validation_freq : int, default is 100
            Specifies how many iteration to run before checking the tolerance is reached, e.g. `validation_freq=10` runs validation every 10 epochs.
        log_freq : int, default is 10
            Specifies how many training epochs to run before logging the training loss.
        """
        self.sourcing_model = sourcing_model

        # Check demand is uniform distributed
        if not isinstance(sourcing_model.demand_generator, UniformDemand):
            raise ValueError(
                "DynamicProgrammingController only supports uniform demand distribution."
            )
        # Check if the expedited_lead_time is 0
        if sourcing_model.expedited_lead_time != 0:
            raise ValueError(
                "DynamicProgrammingController only supports expedited_lead_time = 0."
            )

        start_time = datetime.now()
        logger.info(f"Starting dynamic programming at {start_time}")
        logger.info(
            f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
            f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
            f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
        )
        logger.info(
            f"Training parameters: max_iterations={max_iterations}, tolerance={tolerance}"
        )

        min_demand = int(sourcing_model.demand_generator.get_min_demand())
        max_demand = int(sourcing_model.demand_generator.get_max_demand())
        exp_demand = (max_demand + min_demand) / 2.0
        support = max_demand - min_demand
        h = sourcing_model.get_holding_cost()
        b = sourcing_model.get_shortage_cost()
        ce = sourcing_model.get_expedited_order_cost()
        le = sourcing_model.get_expedited_lead_time()
        lr = sourcing_model.get_regular_lead_time()

        base_e = DynamicProgrammingController._get_basestock_ub(
            exp_demand=exp_demand, lead_time=le, support=support, h=h, b=b
        )
        base_r = DynamicProgrammingController._get_basestock_ub(
            exp_demand=exp_demand, lead_time=lr, support=support, h=h, b=b
        )
        min_ip = int(min(base_r, base_e) - max_demand)
        max_ip = int(max(base_r, base_e))
        max_order = max_ip + min_ip
        dim_pipeline = lr - le - 1

        demand_prob = Dict.empty(key_type=types.int64, value_type=types.float64)
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

        actions_ = list(product(range(max_order), range(max_order), range(max_order)))
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

        for iteration in tqdm(range(max_iterations)):
            # We first store each newly updated state
            for idx, state in enumerate(states):
                these_values[idx] = DynamicProgrammingController._vf_update(
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

            if iteration > 1 and iteration % log_freq == 0:
                logger.info(
                    f"Epoch {iteration}/{max_iterations} - Value: {all_values[iteration]}"
                )

            if iteration > 1 and iteration % validation_freq == 0:
                iteration_arr.append(iteration)
                value_arr.append(all_values[iteration])
                delta = all_values[iteration - 1] - all_values[iteration]
                if delta <= tolerance:
                    for state in states:
                        qa = DynamicProgrammingController._vf_update(
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
        self.vf = val

        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Dynamic programming completed at {end_time}")
        logger.info(f"Total training duration: {duration}")
        logger.info(
            f"Final best cost: {self.get_average_cost(self.sourcing_model, sourcing_periods=1000, seed=42):.4f}"
        )

    def predict(
        self,
        current_inventory: Union[int, torch.Tensor],
        past_regular_orders: Optional[Union[TypingList[int], torch.Tensor]] = None,
        past_expedited_orders: Optional[Union[TypingList[int], torch.Tensor]] = None,
        output_tensor: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        """

        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory.
        past_regular_orders : list, or torch.Tensor, optional
            Past regular orders. If the length of `past_regular_orders` is lower than `regular_lead_time`, it will be padded with zeros. If the length of `past_regular_orders` is higher than `regular_lead_time`, only the last `regular_lead_time` orders will be used during inference.
        past_expedited_orders : list, or torch.Tensor, optional
            Past expedited orders. Since `expedited_lead_time` is assumed to be 0 for DynamicProgrammingController and the batch size is assumed to be 1, the input of `past_expedited_orders` is optional and will be ignored.
        output_tensor : bool, default is False
            If True, the replenishment order quantity will be returned as a torch.Tensor. Otherwise, it will be returned as an integer.

        """
        if self.sourcing_model is None:
            raise AttributeError("The controller is not trained.")

        regular_lead_time = self.sourcing_model.get_regular_lead_time()

        current_inventory = self._check_current_inventory(current_inventory)
        past_regular_orders = self._check_past_orders(
            past_regular_orders, regular_lead_time
        )

        first = (
            current_inventory.squeeze()
            + past_regular_orders.squeeze()[-regular_lead_time]
        )
        second = past_regular_orders.squeeze()[-regular_lead_time + 1 :]
        key = tuple([int(first)] + second.int().tolist())
        if output_tensor:
            return (
                torch.tensor([[self.qf[key][0]]]),
                torch.tensor([[self.qf[key][1]]]),
                torch.tensor([[self.qf[key][2]]]),
            )
        else:
            return self.qf[key]

    def get_last_cost(self, sourcing_model: DualSourcingModel) -> torch.Tensor:
        """Calculate the cost for the latest period."""
        last_regular_q = sourcing_model.get_last_regular_order()
        last_expedited_q = sourcing_model.get_last_expedited_order()
        regular_order_cost = sourcing_model.get_regular_order_cost()
        expedited_order_cost = sourcing_model.get_expedited_order_cost()
        holding_cost = sourcing_model.get_holding_cost()
        shortage_cost = sourcing_model.get_shortage_cost()
        current_inventory = sourcing_model.get_current_inventory()
        last_cost = (
            regular_order_cost * last_regular_q
            + expedited_order_cost * last_expedited_q
            + holding_cost * torch.relu(current_inventory)
            + shortage_cost * torch.relu(-current_inventory)
        )
        return last_cost

    @no_type_check
    def get_total_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the total cost."""
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        # for _ in tqdm(range(sourcing_periods), desc=f"Calculating Average cost across {sourcing_periods} Time periods"):
        for _ in tqdm(range(sourcing_periods)): # // 2 aligns with the original paper

            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()

            regular_q0, expedited_q0, expedited_q1 = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )

            sourcing_model.order(regular_q0, expedited_q0)
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()

            sourcing_model.order(0, expedited_q1)
            last_cost = self.get_last_cost(sourcing_model)
            total_cost += last_cost.mean()

        return total_cost

    @no_type_check
    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the average cost."""
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / sourcing_periods
        )


    def reset(self) -> None:
        self.qf = None
        self.vf = None
        self.sourcing_model = None
