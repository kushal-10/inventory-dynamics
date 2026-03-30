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

from ..demand import UniformDemand
from ..sourcing_model import DualSourcingModel
from .base import BaseDPController

# Get root logger
logger = logging.getLogger()


class DynamicProgrammingController(BaseDPController):
    def __init__(self, cycle_length: int = 2) -> None:
        if cycle_length not in (1, 2, 3):
            raise ValueError("cycle_length must be 1, 2, or 3")
        self.cycle_length = cycle_length
        self.sourcing_model = None
        self.qf = None
        self.vf = None
        logger.info(f"Initialized DynamicProgrammingController with cycle_length={cycle_length}")

    # ------------------------------------------------------------------
    # Bellman update functions — one per supported cycle length.
    # Action convention: (qr0, qe0, qe1, ..., qe_{N-1})
    #   qr0  — regular order placed once per cycle (period 0)
    #   qe_t — expedited order placed every period t in [0, N-1]
    # ------------------------------------------------------------------

    @staticmethod
    @njit
    def _vf_update_n1(
        demand_prob: TypingDict[int, float],
        min_demand: int,
        max_demand: int,
        ce: float,
        h: float,
        b: float,
        state: Tuple[int, ...],
        vf: TypingDict[Tuple[int, ...], float],
        actions: TypingList[Tuple[int, int]],
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Single-period Bellman update (N=1 cycle).

        Actions: (qr0, qe0).
        State: (ip, pipeline[0], ..., pipeline[dim-1])
        """
        best_action = None
        best_cost = 10e9

        for qr0, qe0 in actions:
            immediate_cost = qe0 * ce
            expected_cost = 0.0
            valid = True

            # PERIOD 0: qe0 arrives immediately (le=0), state[1] arrives, qr0 enters pipeline back
            ip0 = state[0] + qe0 + state[1]
            pipeline0 = state[2:]

            for dem0 in range(int(min_demand), int(max_demand) + 1):
                ipe0 = int(ip0) - dem0
                state_next = (ipe0,) + pipeline0 + (qr0,)

                if state_next not in vf:
                    valid = False
                    break

                inv0 = ipe0 - state[1]
                inv_cost0 = inv0 * h if inv0 >= 0 else -inv0 * b
                expected_cost += demand_prob[dem0] * (inv_cost0 + vf[state_next])

            if valid and (immediate_cost + expected_cost) < best_cost:
                best_cost = immediate_cost + expected_cost
                best_action = (qr0, qe0)

        return best_cost, best_action

    @staticmethod
    @njit
    def _vf_update_n2(
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
        Two-period cycle Bellman update (N=2).

        Period 0: place (qr0, qe0). Period 1: place (qe1) only.
        Actions: (qr0, qe0, qe1).
        State: (ip, pipeline[0], ..., pipeline[dim-1])
        """
        best_action = None
        best_cost = 10e9

        for qr0, qe0, qe1 in actions:
            immediate_cost = (qe0 + qe1) * ce
            expected_cost = 0.0
            valid = True

            for dem0 in range(int(min_demand), int(max_demand) + 1):
                if not valid:
                    break
                for dem1 in range(int(min_demand), int(max_demand) + 1):

                    # PERIOD 0: qe0 + state[1] arrive, qr0 enters pipeline back
                    ip0 = state[0] + qe0 + state[1]
                    pipeline0 = state[2:]
                    ipe0 = int(ip0) - dem0
                    state1 = (ipe0,) + pipeline0 + (qr0,)

                    if state1 not in vf:
                        valid = False
                        break

                    inv0 = ipe0 - state[1]
                    inv_cost0 = inv0 * h if inv0 >= 0 else -inv0 * b

                    # PERIOD 1: qe1 + state1[1] arrive, 0 enters pipeline back
                    ip1 = state1[0] + qe1 + state1[1]
                    pipeline1 = state1[2:]
                    ipe1 = int(ip1) - dem1
                    state_next = (ipe1,) + pipeline1 + (0,)

                    if state_next not in vf:
                        valid = False
                        break

                    inv1 = ipe1 - state1[1]
                    inv_cost1 = inv1 * h if inv1 >= 0 else -inv1 * b

                    prob = demand_prob[dem0] * demand_prob[dem1]
                    expected_cost += prob * (inv_cost0 + inv_cost1 + vf[state_next])

            if valid and (immediate_cost + expected_cost) < best_cost:
                best_cost = immediate_cost + expected_cost
                best_action = (qr0, qe0, qe1)

        return best_cost, best_action

    @staticmethod
    @njit
    def _vf_update_n3(
        demand_prob: TypingDict[int, float],
        min_demand: int,
        max_demand: int,
        ce: float,
        h: float,
        b: float,
        state: Tuple[int, ...],
        vf: TypingDict[Tuple[int, ...], float],
        actions: TypingList[Tuple[int, int, int, int]],
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """
        Three-period cycle Bellman update (N=3).

        Period 0: place (qr0, qe0). Periods 1-2: place (qe_t) only.
        Actions: (qr0, qe0, qe1, qe2).
        State: (ip, pipeline[0], ..., pipeline[dim-1])
        """
        best_action = None
        best_cost = 10e9

        for qr0, qe0, qe1, qe2 in actions:
            immediate_cost = (qe0 + qe1 + qe2) * ce
            expected_cost = 0.0
            valid = True

            for dem0 in range(int(min_demand), int(max_demand) + 1):
                if not valid:
                    break
                for dem1 in range(int(min_demand), int(max_demand) + 1):
                    if not valid:
                        break
                    for dem2 in range(int(min_demand), int(max_demand) + 1):

                        # PERIOD 0: qe0 + state[1] arrive, qr0 enters pipeline back
                        ip0 = state[0] + qe0 + state[1]
                        pipeline0 = state[2:]
                        ipe0 = int(ip0) - dem0
                        state1 = (ipe0,) + pipeline0 + (qr0,)

                        if state1 not in vf:
                            valid = False
                            break

                        inv0 = ipe0 - state[1]
                        inv_cost0 = inv0 * h if inv0 >= 0 else -inv0 * b

                        # PERIOD 1: qe1 + state1[1] arrive, 0 enters pipeline back
                        ip1 = state1[0] + qe1 + state1[1]
                        pipeline1 = state1[2:]
                        ipe1 = int(ip1) - dem1
                        state2 = (ipe1,) + pipeline1 + (0,)

                        if state2 not in vf:
                            valid = False
                            break

                        inv1 = ipe1 - state1[1]
                        inv_cost1 = inv1 * h if inv1 >= 0 else -inv1 * b

                        # PERIOD 2: qe2 + state2[1] arrive, 0 enters pipeline back
                        ip2 = state2[0] + qe2 + state2[1]
                        pipeline2 = state2[2:]
                        ipe2 = int(ip2) - dem2
                        state_next = (ipe2,) + pipeline2 + (0,)

                        if state_next not in vf:
                            valid = False
                            break

                        inv2 = ipe2 - state2[1]
                        inv_cost2 = inv2 * h if inv2 >= 0 else -inv2 * b

                        prob = demand_prob[dem0] * demand_prob[dem1] * demand_prob[dem2]
                        expected_cost += prob * (inv_cost0 + inv_cost1 + inv_cost2 + vf[state_next])

            if valid and (immediate_cost + expected_cost) < best_cost:
                best_cost = immediate_cost + expected_cost
                best_action = (qr0, qe0, qe1, qe2)

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
            f"Training parameters: max_iterations={max_iterations}, tolerance={tolerance}, "
            f"cycle_length={self.cycle_length}"
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

        # Actions: (qr0, qe0, qe1, ..., qe_{N-1}) — 1 regular + N expedited per cycle
        actions_ = list(product(range(max_order), repeat=self.cycle_length + 1))
        actions = List()
        for action in actions_:
            actions.append(action)

        # Select Bellman update for the configured cycle length
        _vf_update = {
            1: DynamicProgrammingController._vf_update_n1,
            2: DynamicProgrammingController._vf_update_n2,
            3: DynamicProgrammingController._vf_update_n3,
        }[self.cycle_length]

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
            # Store each updated value before writing back (Gauss-Seidel style)
            for idx, state in enumerate(states):
                these_values[idx] = _vf_update(
                    demand_prob, min_demand, max_demand, ce, h, b, state, vf, actions
                )[0]
            for idx, state in enumerate(states):
                vf[state] = these_values[idx]

            iter_vals = np.array([val for val in vf.values() if val < 10e8])
            this_average = np.mean(iter_vals)

            val = this_average / (iteration + 1)
            all_values[iteration] = val

            if iteration > 1 and iteration % log_freq == 0:
                logger.info(
                    f"Epoch {iteration}/{max_iterations} - Value: {all_values[iteration]:.4f}"
                )

            if iteration > 1 and iteration % validation_freq == 0:
                iteration_arr.append(iteration)
                value_arr.append(all_values[iteration])
                delta = all_values[iteration - 1] - all_values[iteration]
                if delta <= tolerance:
                    for state in states:
                        qa = _vf_update(
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
    ) -> Union[Tuple, Tuple[int, ...]]:
        """
        Parameters
        ----------
        current_inventory : int, or torch.Tensor
            Current inventory.
        past_regular_orders : list, or torch.Tensor, optional
            Past regular orders. If the length of `past_regular_orders` is lower than
            `regular_lead_time`, it will be padded with zeros.
        past_expedited_orders : list, or torch.Tensor, optional
            Ignored. Expedited lead time is assumed to be 0.
        output_tensor : bool, default is False
            If True, returns a tuple of torch.Tensors of length cycle_length + 1
            in the order (qr0, qe0, qe1, ..., qe_{N-1}).
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
        second = past_regular_orders.squeeze()[-regular_lead_time + 1:]
        key = tuple([int(first)] + second.int().tolist())

        if output_tensor:
            return tuple(torch.tensor([[v]]) for v in self.qf[key])
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
        """
        Calculate the total cost over `sourcing_periods` cycles.

        Each cycle covers `cycle_length` real periods: one regular + expedited
        order in period 0, then expedited-only orders in periods 1..N-1.
        """
        if seed is not None:
            torch.manual_seed(seed)

        total_cost = torch.tensor(0.0)
        for _ in tqdm(range(sourcing_periods)):
            current_inventory = sourcing_model.get_current_inventory()
            past_regular_orders = sourcing_model.get_past_regular_orders()
            past_expedited_orders = sourcing_model.get_past_expedited_orders()

            # actions = (qr0, qe0, qe1, ..., qe_{N-1})
            actions = self.predict(
                current_inventory,
                past_regular_orders,
                past_expedited_orders,
                output_tensor=True,
            )

            # Period 0: regular + first expedited order
            sourcing_model.order(actions[0], actions[1])
            total_cost += self.get_last_cost(sourcing_model).mean()

            # Periods 1..N-1: expedited only
            for t in range(1, self.cycle_length):
                sourcing_model.order(torch.zeros_like(actions[0]), actions[t + 1])
                total_cost += self.get_last_cost(sourcing_model).mean()

        return total_cost

    @no_type_check
    def get_average_cost(
        self,
        sourcing_model: DualSourcingModel,
        sourcing_periods: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate the average per-period cost."""
        return (
            self.get_total_cost(sourcing_model, sourcing_periods, seed)
            / (sourcing_periods * self.cycle_length)
        )

    def reset(self) -> None:
        self.qf = None
        self.vf = None
        self.sourcing_model = None
