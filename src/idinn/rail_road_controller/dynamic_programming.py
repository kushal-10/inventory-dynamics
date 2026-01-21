import logging
from datetime import datetime
from itertools import product
from typing import Dict as TypingDict, Tuple, Optional

import numpy as np
import torch
from numba import njit, types
from numba.typed import Dict, List
from tqdm import tqdm

from ..rail_road_model import RailRoadInventoryModel

logger = logging.getLogger()


class RailRoadDPController:
    """
    Dynamic Programming controller for the periodic rail-road inventory model.
    """

    def __init__(self) -> None:
        self.model: Optional[RailRoadInventoryModel] = None
        self.qf = None  # policy: (inventory, cycle_day) -> order quantity
        self.vf = None  # optimal average cost

    # ------------------------------------------------------------------
    # Bellman update
    # ------------------------------------------------------------------

    @staticmethod
    @njit
    def _vf_update(
        demand_prob: TypingDict[int, float],
        min_demand: int,
        max_demand: int,
        h: float,
        b: float,
        c: float,
        cycle_len: int,
        state: Tuple[int, int],
        vf: TypingDict[Tuple[int, int], float],
        max_order: int,
    ) -> Tuple[float, int]:

        I, tau = state
        best_cost = 1e12
        best_q = 0

        for q in range(max_order + 1):
            cost = c * q
            next_tau = (tau + 1) % cycle_len

            for d in range(min_demand, max_demand + 1):
                I_next = I + q - d
                next_state = (I_next, next_tau)

                if next_state not in vf:
                    cost = 1e12
                    break

                inv_cost = I_next * h if I_next >= 0 else -I_next * b
                cost += demand_prob[d] * (inv_cost + vf[next_state])

            if cost < best_cost:
                best_cost = cost
                best_q = q

        return best_cost, best_q

    # ------------------------------------------------------------------
    # Fit DP
    # ------------------------------------------------------------------

    def fit(
        self,
        model: RailRoadInventoryModel,
        max_iterations: int = 200_000,
        tolerance: float = 1e-6,
        validation_freq: int = 100,
        log_freq: int = 500,
    ) -> None:
        self.model = model

        start_time = datetime.now()
        logger.info(f"Starting Rail-Road DP at {start_time}")

        # Demand
        demand_prob = Dict.empty(types.int64, types.float64)
        support = model.demand_generator.enumerate_support()
        for k, v in support.items():
            demand_prob[k] = v

        min_demand = min(support)
        max_demand = max(support)
        exp_demand = sum(k * v for k, v in support.items())

        # Costs
        h = model.get_holding_cost()
        b = model.get_shortage_cost()
        cr = model.get_regular_order_cost()
        ce = model.get_expedited_order_cost()
        K = model.cycle_length

        # Truncation
        support_width = max_demand - min_demand
        base_stock = exp_demand + support_width * np.sqrt(
            np.log(1 + b / h) / 2
        )

        max_I = int(np.ceil(base_stock))
        min_I = int(-max_demand)
        max_order = max_I - min_I

        # State space
        states = List()
        for I, tau in product(range(min_I, max_I + 1), range(K)):
            states.append((I, tau))

        # Initialize value function
        vf = Dict.empty(types.UniTuple(types.int64, 2), types.float64)
        for s in states:
            vf[s] = 0.0

        all_values = np.zeros(max_iterations)

        # Value iteration
        for it in tqdm(range(max_iterations)):
            new_vf = Dict.empty(types.UniTuple(types.int64, 2), types.float64)

            for state in states:
                _, tau = state
                c = cr if tau == 0 else ce

                v, _ = self._vf_update(
                    demand_prob,
                    min_demand,
                    max_demand,
                    h,
                    b,
                    c,
                    K,
                    state,
                    vf,
                    max_order,
                )
                new_vf[state] = v

            vf = new_vf
            avg_val = np.mean(np.array(list(vf.values())))
            all_values[it] = avg_val / (it + 1)

            if it > 1 and it % validation_freq == 0:
                if abs(all_values[it] - all_values[it - 1]) <= tolerance:
                    break

        # Extract policy
        qf = {}
        for state in states:
            _, tau = state
            c = cr if tau == 0 else ce
            _, q = self._vf_update(
                demand_prob,
                min_demand,
                max_demand,
                h,
                b,
                c,
                K,
                state,
                vf,
                max_order,
            )
            qf[state] = q

        self.qf = qf
        self.vf = all_values[it]

        logger.info(
            f"Rail-Road DP completed in {datetime.now() - start_time}, "
            f"avg cost ≈ {self.vf:.4f}"
        )

    # ------------------------------------------------------------------
    # Predict (model-aware)
    # ------------------------------------------------------------------

    def predict(self) -> int:
        """
        Predict order quantity using the current model state.
        """
        if self.qf is None or self.model is None:
            raise RuntimeError("Controller not trained or model not set.")

        I = int(self.model.get_current_inventory().item())
        tau = self.model.get_current_cycle_day()

        return self.qf[(I, tau)]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.qf = None
        self.vf = None
        self.model = None

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------

    def get_average_cost(
            self,
            model: RailRoadInventoryModel,
            periods: int,
            seed: Optional[int] = None,
    ) -> float:
        """
        Evaluate average cost under the learned DP policy.
        Costs are charged at time t using the action chosen at (I_t, tau_t).
        """
        if self.qf is None:
            raise RuntimeError("Controller not trained.")

        if seed is not None:
            torch.manual_seed(seed)

        model.reset(batch_size=1)
        total_cost = 0.0

        for _ in range(periods):
            # ----- observe state (I_t, tau_t) -----
            I = model.get_current_inventory().item()
            tau = model.get_current_cycle_day()

            # ----- choose action -----
            q = self.qf[(int(I), tau)]

            # ----- cost is incurred NOW (before transition) -----
            order_cost = (
                model.get_regular_order_cost()
                if tau == 0
                else model.get_expedited_order_cost()
            )

            total_cost += order_cost * q

            # ----- system transition -----
            model.order(q)

            # ----- holding / backlog cost after demand -----
            I_next = model.get_current_inventory().item()
            total_cost += (
                    model.get_holding_cost() * max(I_next, 0)
                    + model.get_shortage_cost() * max(-I_next, 0)
            )

        return total_cost / periods

