"""
Common simulation utilities for phase controllers (DP and NN).

Runs n_seeds demand realizations through a fitted controller and sourcing model,
recording per-period state and cost. Each sourcing cycle consists of two actual
periods: an even period (regular + expedited order) and an odd period (expedited only).
"""

import os
from typing import Optional

import pandas as pd
import torch

from ..sourcing_model import DualSourcingModel


def run_simulation(
    controller,
    sourcing_model: DualSourcingModel,
    n_seeds: int = 100,
    sourcing_periods: int = 50,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Simulate a fitted controller over n_seeds independent demand realizations.

    Each seed produces 2 * sourcing_periods actual time steps (one 2-period cycle
    per sourcing period). Compatible with both DynamicProgrammingController and
    MultiPeriodNeuralController.

    Parameters
    ----------
    controller : DynamicProgrammingController or MultiPeriodNeuralController
        A fitted/loaded controller with a predict(current_inventory,
        past_regular_orders, past_expedited_orders, output_tensor=True) method.
    sourcing_model : DualSourcingModel
        The sourcing model used for simulation. Reset to its stored init_inventory
        at the start of each seed.
    n_seeds : int
        Number of independent demand realizations to simulate (seeds 0 .. n_seeds-1).
    sourcing_periods : int
        Number of 2-period cycles per seed. Total actual periods = 2 * sourcing_periods.
    output_path : str, optional
        If provided, save the resulting DataFrame as a CSV to this path.
        Parent directories are created automatically.

    Returns
    -------
    pd.DataFrame
        Columns: seed, period, demand, inventory, regular_order, expedited_order, cost
    """
    holding_cost = sourcing_model.get_holding_cost()
    shortage_cost_val = sourcing_model.get_shortage_cost()
    expedited_order_cost = sourcing_model.get_expedited_order_cost()
    regular_order_cost = sourcing_model.get_regular_order_cost()

    records = []

    with torch.no_grad():
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            sourcing_model.reset()

            for t in range(sourcing_periods):
                current_inventory = sourcing_model.get_current_inventory()
                past_regular_orders = sourcing_model.get_past_regular_orders()
                past_expedited_orders = sourcing_model.get_past_expedited_orders()

                regular_q0, expedited_q0, expedited_q1 = controller.predict(
                    current_inventory,
                    past_regular_orders,
                    past_expedited_orders,
                    output_tensor=True,
                )

                # --- Even period: place regular_q0 and expedited_q0 ---
                sourcing_model.order(regular_q0, expedited_q0)
                inv = sourcing_model.get_current_inventory()
                cost = (
                    regular_order_cost * sourcing_model.get_last_regular_order()
                    + expedited_order_cost * sourcing_model.get_last_expedited_order()
                    + holding_cost * torch.relu(inv)
                    + shortage_cost_val * torch.relu(-inv)
                ).item()
                records.append({
                    "seed": seed,
                    "period": 2 * t,
                    "demand": sourcing_model.past_demands[:, -1].item(),
                    "inventory": inv.item(),
                    "regular_order": sourcing_model.get_last_regular_order().item(),
                    "expedited_order": sourcing_model.get_last_expedited_order().item(),
                    "cost": cost,
                })

                # --- Odd period: place only expedited_q1, no regular order ---
                sourcing_model.order(torch.zeros_like(expedited_q1), expedited_q1)
                inv = sourcing_model.get_current_inventory()
                cost = (
                    regular_order_cost * sourcing_model.get_last_regular_order()
                    + expedited_order_cost * sourcing_model.get_last_expedited_order()
                    + holding_cost * torch.relu(inv)
                    + shortage_cost_val * torch.relu(-inv)
                ).item()
                records.append({
                    "seed": seed,
                    "period": 2 * t + 1,
                    "demand": sourcing_model.past_demands[:, -1].item(),
                    "inventory": inv.item(),
                    "regular_order": sourcing_model.get_last_regular_order().item(),
                    "expedited_order": sourcing_model.get_last_expedited_order().item(),
                    "cost": cost,
                })

    df = pd.DataFrame(records)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved simulation results to {output_path}")

    return df
