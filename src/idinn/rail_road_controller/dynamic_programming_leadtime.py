import logging
from datetime import datetime
from itertools import product
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from numba import njit, types
from numba.typed import Dict as NumbaDict, List
from tqdm import tqdm

from .dynamic_programming import RailRoadDPController
from ..rail_road_model import RailRoadInventoryModelWithLeadTime

logger = logging.getLogger()

LEAD_TIME = 2   # <-- HARD-CODED

class RailRoadDPControllerWithLeadTime(RailRoadDPController):
    """
    Dynamic Programming controller for the rail-road model with lead time.
    State = (inventory, pipeline_1, ..., pipeline_L, cycle_day)
    """

    def __init__(self) -> None:
        self.model: Optional[RailRoadInventoryModelWithLeadTime] = None
        self.qf = None
        self.vf = None

    # ------------------------------------------------------------------
    # Bellman update
    # ------------------------------------------------------------------

    @staticmethod
    @njit
    def _vf_update(
            demand_prob,
            min_demand,
            max_demand,
            h,
            b,
            c,
            cycle_len,
            state,
            vf,
            max_order,
    ):
        """
        State = (I, p1, p2, tau)
        """
        best_cost = 1e12
        best_q = 0

        I = state[0]
        p1 = state[1]
        p2 = state[2]
        tau = state[3]

        next_tau = (tau + 1) % cycle_len

        for q in range(max_order + 1):
            cost = c * q

            for d in range(min_demand, max_demand + 1):
                I_next = I + p1 - d

                # HARD-CODED PIPELINE SHIFT
                next_state = (
                    I_next,
                    p2,  # p1 <- p2
                    q,  # p2 <- q
                    next_tau,
                )

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
    # Fit via value iteration
    # ------------------------------------------------------------------

    def fit(
        self,
        model: RailRoadInventoryModelWithLeadTime,
        max_iterations: int = 200_000,
        tolerance: float = 1e-6,
        validation_freq: int = 100,
        log_freq: int = 500,
    ) -> None:

        self.model = model
        start_time = datetime.now()
        logger.info(f"Starting Rail-Road DP with lead time at {start_time}")

        # --------------------------------------------------
        # Demand distribution
        # --------------------------------------------------

        demand_prob = NumbaDict.empty(types.int64, types.float64)
        support = model.demand_generator.enumerate_support()

        for k, v in support.items():
            demand_prob[k] = v

        min_demand = min(support)
        max_demand = max(support)
        exp_demand = sum(k * v for k, v in support.items())

        # --------------------------------------------------
        # Costs and parameters
        # --------------------------------------------------

        h = model.get_holding_cost()
        b = model.get_shortage_cost()
        cr = model.get_regular_order_cost()
        ce = model.get_expedited_order_cost()
        K = model.cycle_length
        L = model.lead_time

        # --------------------------------------------------
        # Truncation bounds
        # --------------------------------------------------

        support_width = max_demand - min_demand
        base_stock = (L + 1) * exp_demand + support_width * np.sqrt(
            (L + 1) * np.log(1 + b / h) / 2
        )

        I_max = int(np.ceil(base_stock))
        I_min = -max_demand
        max_order = I_max - I_min

        # --------------------------------------------------
        # State space
        # --------------------------------------------------

        states = List()
        for I in range(I_min, I_max + 1):
            for pipeline in product(range(I_max + 1), repeat=L):
                for tau in range(K):
                    states.append((I,) + pipeline + (tau,))

        # --------------------------------------------------
        # Initialize value function
        # --------------------------------------------------

        vf = NumbaDict.empty(
            types.UniTuple(types.int64, L + 2), types.float64
        )

        for s in states:
            vf[s] = 0.0

        all_values = np.zeros(max_iterations)

        # --------------------------------------------------
        # Value iteration
        # --------------------------------------------------

        for it in tqdm(range(max_iterations)):
            new_vf = NumbaDict.empty(
                types.UniTuple(types.int64, L + 2), types.float64
            )

            for state in states:
                tau = state[-1]
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

            if it > 1 and it % log_freq == 0:
                logger.info(
                    f"Iteration {it}/{max_iterations}, avg cost ≈ {all_values[it]:.6f}"
                )

        # --------------------------------------------------
        # Extract policy
        # --------------------------------------------------

        qf = {}
        for state in states:
            tau = state[-1]
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
            f"Rail-Road DP with lead time completed in {datetime.now() - start_time}, "
            f"avg cost ≈ {self.vf:.4f}"
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self) -> int:
        if self.qf is None or self.model is None:
            raise RuntimeError("Controller not trained.")

        I = int(self.model.get_current_inventory().item())
        pipeline = tuple(int(x) for x in self.model.get_pipeline().squeeze().tolist())
        tau = self.model.get_current_cycle_day()

        return self.qf[(I,) + pipeline + (tau,)]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.qf = None
        self.vf = None
        self.model = None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def get_average_cost(
        self,
        model,
        periods: int,
        seed: Optional[int] = None,
    ) -> float:

        if self.qf is None:
            raise RuntimeError("Controller not trained.")

        if seed is not None:
            torch.manual_seed(seed)

        model.reset(batch_size=1)
        total_cost = 0.0

        for _ in range(periods):
            I = int(model.get_current_inventory().item())
            pipeline = tuple(int(x) for x in model.get_pipeline().squeeze().tolist())
            tau = model.get_current_cycle_day()

            q = self.qf[(I,) + pipeline + (tau,)]

            order_cost = (
                model.get_regular_order_cost()
                if tau == 0
                else model.get_expedited_order_cost()
            )
            total_cost += order_cost * q

            model.order(q)

            I_next = model.get_current_inventory().item()
            total_cost += (
                model.get_holding_cost() * max(I_next, 0)
                + model.get_shortage_cost() * max(-I_next, 0)
            )

        return total_cost / periods
