import logging
from datetime import datetime
from itertools import product
from typing import Optional, Tuple

import numpy as np
import torch
from numba import njit, types
from numba.typed import Dict, List
from tqdm import tqdm

from ..rail_road_model_lead import RailRoadInventoryModelLeadTime

logger = logging.getLogger()


class RailRoadDPLeadTimeController:
    """
    Dynamic Programming controller for rail-road inventory
    with BOTH cycle constraint and lead time.
    """

    def __init__(self, lead_time: int) -> None:
        self.L = lead_time
        self.model: Optional[RailRoadInventoryModelLeadTime] = None
        self.qf = None
        self.vf = None

    # --------------------------------------------------
    # Bellman update (cycle + lead time)
    # --------------------------------------------------

    @staticmethod
    @njit
    def _vf_update(
        demand_prob: Dict,
        min_demand: int,
        max_demand: int,
        h: float,
        b: float,
        cr: float,
        ce: float,
        cycle_len: int,
        lead_time: int,
        state: Tuple,
        vf: Dict,
        max_order: int,
    ) -> Tuple[float, int]:

        I = state[0]
        tau = state[-1]
        pipeline = state[1:-1]

        best_cost = 1e12
        best_q = 0

        # rail orders only allowed when tau == 0
        q_range = range(max_order + 1) if tau == 0 else range(1)

        for q in q_range:
            order_cost = (cr if tau == 0 else ce) * q
            next_tau = (tau + 1) % cycle_len

            arrived = pipeline[0]
            pipeline_next = pipeline[1:] + (q,)

            cost = order_cost

            for d in range(min_demand, max_demand + 1):
                I_next = I + arrived - d
                next_state = (I_next,) + pipeline_next + (next_tau,)

                if next_state not in vf:
                    cost = 1e12
                    break

                inv_cost = I_next * h if I_next >= 0 else -I_next * b
                cost += demand_prob[d] * (inv_cost + vf[next_state])

            if cost < best_cost:
                best_cost = cost
                best_q = q

        return best_cost, best_q

    # --------------------------------------------------
    # Fit DP
    # --------------------------------------------------

    def fit(
        self,
        model: RailRoadInventoryModelLeadTime,
        max_iterations: int = 200_000,
        tolerance: float = 1e-6,
        validation_freq: int = 100,
    ) -> None:

        self.model = model
        L = self.L
        K = model.cycle_length

        start_time = datetime.now()
        logger.info(f"Starting Rail-Road DP (cycle + lead time) at {start_time}")

        # -----------------------------
        # Demand distribution
        # -----------------------------
        demand_prob = Dict.empty(types.int64, types.float64)
        support = model.demand_generator.enumerate_support()
        for k, v in support.items():
            demand_prob[k] = v

        min_demand = min(support)
        max_demand = max(support)
        exp_demand = sum(k * v for k, v in support.items())

        # -----------------------------
        # Costs
        # -----------------------------
        h = model.get_holding_cost()
        b = model.get_shortage_cost()
        cr = model.get_regular_order_cost()
        ce = model.get_expedited_order_cost()

        # -----------------------------
        # Truncation
        # -----------------------------
        support_width = max_demand - min_demand
        base_stock = exp_demand + support_width * np.sqrt(
            np.log(1 + b / h) / 2
        )

        max_I = int(np.ceil(base_stock))
        min_I = int(-max_demand)
        max_order = max_I - min_I

        # -----------------------------
        # State space
        # (I, q1, ..., qL, tau)
        # -----------------------------
        states = List()
        for I in range(min_I, max_I + 1):
            for pipeline in product(range(max_order + 1), repeat=L):
                for tau in range(K):
                    states.append((I,) + pipeline + (tau,))

        logger.info(f"State count: {len(states)}")

        # -----------------------------
        # Initialize value function
        # -----------------------------
        vf = Dict.empty(types.UniTuple(types.int64, L + 2), types.float64)
        for s in states:
            vf[s] = 0.0

        all_values = np.zeros(max_iterations)

        # -----------------------------
        # Value iteration
        # -----------------------------
        for it in tqdm(range(max_iterations)):
            new_vf = Dict.empty(types.UniTuple(types.int64, L + 2), types.float64)

            for state in states:
                v, _ = self._vf_update(
                    demand_prob,
                    min_demand,
                    max_demand,
                    h,
                    b,
                    cr,
                    ce,
                    K,
                    L,
                    state,
                    vf,
                    max_order,
                )
                new_vf[state] = v

            vf = new_vf
            avg_val = np.mean(np.array(list(vf.values())))
            all_values[it] = avg_val / (it + 1)

            if it > 1 and it % validation_freq == 0:
                test_val = abs(all_values[it] - all_values[it - 1])
                print(test_val)
                if abs(all_values[it] - all_values[it - 1]) <= tolerance:
                    break

        # -----------------------------
        # Extract policy
        # -----------------------------
        qf = {}
        for state in states:
            _, q = self._vf_update(
                demand_prob,
                min_demand,
                max_demand,
                h,
                b,
                cr,
                ce,
                K,
                L,
                state,
                vf,
                max_order,
            )
            qf[state] = q

        self.qf = qf
        self.vf = all_values[it]

        logger.info(
            f"DP completed in {datetime.now() - start_time}, "
            f"avg cost ≈ {self.vf:.4f}"
        )

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    def predict(self) -> int:
        if self.qf is None or self.model is None:
            raise RuntimeError("Controller not trained or model not set.")

        I = int(self.model.get_current_inventory().item())
        pipeline = tuple(self.model.get_pipeline())  # length L
        tau = self.model.get_current_cycle_day()

        return self.qf[(I,) + pipeline + (tau,)]

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self) -> None:
        self.qf = None
        self.vf = None
        self.model = None

    def get_average_cost(
            self,
            model,
            periods: int,
            seed: Optional[int] = None,
    ) -> float:
        """
        Evaluate the learned DP policy on the environment
        and return average cost per period.

        Parameters
        ----------
        model : RailRoadInventoryModelLeadTime
            Inventory model with lead time.
        periods : int
            Number of simulation periods.
        seed : int, optional
            Random seed for reproducibility.
        """
        if self.qf is None:
            raise RuntimeError("Controller not trained.")

        if seed is not None:
            torch.manual_seed(seed)

        model.reset(batch_size=1)

        total_cost = 0.0

        for _ in range(periods):
            # -----------------------------
            # Choose action
            # -----------------------------
            I = int(model.get_current_inventory().item())
            pipeline = tuple(model.get_pipeline())
            tau = model.get_current_cycle_day()

            q = self.qf[(I,) + pipeline + (tau,)]
            q_tensor = torch.tensor([[q]])

            # -----------------------------
            # Cost BEFORE transition
            # -----------------------------
            order_cost = (
                             model.get_regular_order_cost()
                             if model.is_regular_day()
                             else model.get_expedited_order_cost()
                         ) * q

            # -----------------------------
            # Advance environment
            # -----------------------------
            model.order(q_tensor)

            # -----------------------------
            # Holding / shortage cost
            # -----------------------------
            I_next = model.get_current_inventory().item()
            holding = model.get_holding_cost() * max(I_next, 0)
            shortage = model.get_shortage_cost() * max(-I_next, 0)

            total_cost += order_cost + holding + shortage

        return total_cost / periods

