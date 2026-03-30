
# import logging
# from datetime import datetime
# from itertools import product
# from typing import Dict as TypingDict
# from typing import List as TypingList
# from typing import Optional, Tuple, Union, no_type_check

# import numpy as np
# import torch
# from numba import njit, types  # type: ignore
# from numba.typed import Dict, List
# from tqdm import tqdm

# from ..sourcing_model import DualSourcingModel
# from ..dual_controller.base import BasePeriodicDualController

# # Get root logger
# logger = logging.getLogger()


# class DynamicProgrammingController(BasePeriodicDualController):
#     def __init__(self) -> None:
#         self.sourcing_model = None
#         self.qf = None
#         self.vf = None
#         logger.info("Initialized DynamicProgrammingController")

#     @staticmethod
#     @njit
#     def _vf_update(
#         demand_prob: TypingDict[int, float],
#         min_demand: int,
#         max_demand: int,
#         ce: float,
#         h: float,
#         b: float,
#         state: Tuple[int, ...],
#         vf: TypingDict[Tuple[int, ...], float],
#         actions: TypingList[Tuple[int, int]],
#     ) -> Tuple[float, Optional[Tuple[int, int]]]:
#         """
#         vf_update is a function that calculates a single value iteration update.
#         """
#         best_action = None
#         best_cost = 10e9

#         current_phase = state[-1]
#         current_pipeline = state[1:-1]
#         ip_e = state[0]

#         for qe, qr in actions:

#             if current_phase==0 and qr!=0:
#                 continue # skip the instances where qr is placed in an even time period - this is restricted 
            
#             # Immediate cost of action
#             cost = qe * ce
            
#             # partial inventory update
#             ip_e_partial = ip_e + qe + current_pipeline[0] # Regular order arriving at time t 

#             for dem in range(int(min_demand), int(max_demand) + 1):
#                 ipe_new = int(ip_e_partial) - dem

#                 inventory_on_hand = ipe_new - current_pipeline[0]

#                 inv_cost = h*inventory_on_hand if inventory_on_hand>0 else b*-inventory_on_hand

#                 next_phase = 1-current_phase
#                 next_state = (ipe_new,) + current_pipeline[1:] + (qr,) + (next_phase,)

#                 if (next_state not in vf) or (vf[next_state] > 10e9 - 1.0):
#                     cost = 10e9
#                     break

#                 cost += demand_prob[dem] * (inv_cost + vf[next_state])

#             if cost < best_cost:
#                 best_cost = cost
#                 best_action = (qr, qe)

#         return best_cost, best_action

#     @staticmethod
#     def _get_basestock_ub(
#         exp_demand: float, lead_time: int, support: float, h: float, b: float
#     ) -> float:
#         """
#         Get an upper bound on the single-source basestock level based on
#         Hoeffding's inequality.
#         """
#         n = lead_time + 1
#         base_stock_ub = n * exp_demand + support * np.sqrt(n * np.log(1 + b / h) / 2)
#         return np.ceil(base_stock_ub)

#     @no_type_check
#     def fit(
#         self,
#         sourcing_model: DualSourcingModel,
#         max_iterations: int = 1000000,
#         tolerance: float = 10e-8,
#         validation_freq: int = 100,
#         log_freq: int = 100,
#     ) -> None:
#         """
#         Fit the controller to the given sourcing model.

#         Parameters
#         ----------
#         sourcing_model : DualSourcingModel
#             The sourcing model to fit the controller to.
#         max_iterations : int, default is 1000000
#             Specifies the maximum number of iterations to run.
#         tolerance : float, default is 10e-8
#             Specifies the tolerance to check if the value function has converged.
#         validation_freq : int, default is 100
#             Specifies how many iteration to run before checking the tolerance is reached, e.g. `validation_freq=10` runs validation every 10 epochs.
#         log_freq : int, default is 10
#             Specifies how many training epochs to run before logging the training loss.
#         """
#         self.sourcing_model = sourcing_model

#         # Check demand is uniform distributed
#         # if not isinstance(sourcing_model.demand_generator, UniformDemand):
#         #     raise ValueError(
#         #         "DynamicProgrammingController only supports uniform demand distribution."
#         #     )
#         # Check if the expedited_lead_time is 0
#         if sourcing_model.expedited_lead_time != 0:
#             raise ValueError(
#                 "DynamicProgrammingController only supports expedited_lead_time = 0."
#             )

#         start_time = datetime.now()
#         logger.info(f"Starting dynamic programming at {start_time}")
#         logger.info(
#             f"Sourcing model parameters: batch_size={self.sourcing_model.batch_size}, "
#             f"lead_time={self.sourcing_model.lead_time}, init_inventory={self.sourcing_model.init_inventory.int().item()}, "
#             f"demand_generator={self.sourcing_model.demand_generator.__class__.__name__}"
#         )
#         logger.info(
#             f"Training parameters: max_iterations={max_iterations}, tolerance={tolerance}"
#         )

#         min_demand = int(sourcing_model.demand_generator.get_min_demand())
#         max_demand = int(sourcing_model.demand_generator.get_max_demand())
#         exp_demand = (max_demand + min_demand) / 2.0
#         support = max_demand - min_demand
#         h = sourcing_model.get_holding_cost()
#         b = sourcing_model.get_shortage_cost()
#         ce = sourcing_model.get_expedited_order_cost()
#         le = sourcing_model.get_expedited_lead_time()
#         lr = sourcing_model.get_regular_lead_time()

#         base_e = DynamicProgrammingController._get_basestock_ub(
#             exp_demand=exp_demand, lead_time=le, support=support, h=h, b=b
#         )
#         base_r = DynamicProgrammingController._get_basestock_ub(
#             exp_demand=exp_demand, lead_time=lr, support=support, h=h, b=b
#         )
#         min_ip = int(min(base_r, base_e) - max_demand)
#         max_ip = int(max(base_r, base_e))
#         max_order = max_ip + min_ip
#         dim_pipeline = lr - le - 1

#         demand_prob = Dict.empty(key_type=types.int64, value_type=types.float64)
#         demand_prob_ = sourcing_model.demand_generator.enumerate_support()
#         for k, v in demand_prob_.items():
#             demand_prob[k] = v

#         # states_ = list(
#         #     product(
#         #         range(-min_ip, max_ip + 1),
#         #         *(range(int(max_demand) + 1),) * int(dim_pipeline),
#         #     )
#         # )
#         states = []
#         all_possible_states = []
#         for state in product(
#             range(-min_ip, max_ip + 1),
#             *(range(int(max_demand) + 1),) * dim_pipeline,
#             (0, 1),
#         ):
#             *_, phase = state
#             pipeline = state[1:-1]

#             all_possible_states.append(state)
#             # unreachable: regular order placed in even period
#             if phase == 1 and pipeline[-1] != 0:
#                 continue

#             states.append(state)

#         logger.info(f"All possible states : {len(all_possible_states)}, Pruned {len(all_possible_states)-len(states)} states")
#         logger.info(f"Generated {len(states)} states")

#         actions_ = list(product(range(max_order), range(max_order)))
#         actions = List()
#         for action in actions_:
#             actions.append(action)

#         logger.info(f"Possible actions : {actions}")
#         logger.info(f"Generated {len(actions)} actions")

#         # Values can be initiated arbitrarily
#         vals = np.repeat(1.0, len(states))
#         vf_ = dict(zip(states, vals))
#         vf = Dict.empty(
#             key_type=types.UniTuple(types.int64, lr+1), value_type=types.float64
#         )
#         for k, v in vf_.items():
#             vf[k] = v

#         all_values = np.zeros(max_iterations, dtype=float)
#         these_values = np.zeros(len(states))
#         iteration_arr = []
#         value_arr = []
#         qf = {}
#         val = 0

#         for iteration in tqdm(range(max_iterations)):
#             # We first store each newly updated state
#             for idx, state in enumerate(states):
#                 these_values[idx] = DynamicProgrammingController._vf_update(
#                     demand_prob, min_demand, max_demand, ce, h, b, state, vf, actions
#                 )[0]
#             # After the minimum for each state has been calculated, we update the states
#             # If done in the same loop, converge sucks even more
#             for idx, state in enumerate(states):
#                 vf[state] = these_values[idx]

#             iter_vals = np.array([val for val in vf.values() if val < 10e8])
#             this_average = np.mean(iter_vals)

#             val = this_average / (iteration + 1)
#             all_values[iteration] = val

#             if iteration > 1 and iteration % log_freq == 0:
#                 logger.info(
#                     f"Epoch {iteration}/{max_iterations} - Value: {all_values[iteration]:.4f}"
#                 )

#             # if iteration > 1 and iteration % validation_freq == 0:
#             #     iteration_arr.append(iteration)
#             #     value_arr.append(all_values[iteration])
#             #     delta = all_values[iteration - 1] - all_values[iteration]
#             #     if delta <= tolerance:
#             #         for state in states:
#             #             qa = DynamicProgrammingController._vf_update(
#             #                 demand_prob,
#             #                 min_demand,
#             #                 max_demand,
#             #                 ce,
#             #                 h,
#             #                 b,
#             #                 state,
#             #                 vf,
#             #                 actions,
#             #             )[1]
#             #             if qa is not None:
#             #                 qf[state] = qa
#             #         break
#             if iteration > 1 and iteration % validation_freq == 0:
#                 iteration_arr.append(iteration)
#                 value_arr.append(all_values[iteration])
#                 delta = all_values[iteration - 1] - all_values[iteration]

#                 if delta <= tolerance:

#                     # ================================
#                     # 🔧 NORMALIZE VALUE FUNCTION HERE
#                     # ================================
#                     ref_vals = [
#                         v for k, v in vf.items()
#                         if k[-1] == 0 and v < 1e8   # phase = 0 reference
#                     ]

#                     if len(ref_vals) == 0:
#                         raise RuntimeError(
#                             "No finite reference values found for phase=0. "
#                             "State space or transitions are inconsistent."
#                         )

#                     baseline = np.mean(ref_vals)

#                     for k in vf:
#                         vf[k] -= baseline
#                     # ================================

#                     # 🔽 NOW extract optimal policy
#                     for state in states:
#                         qa = DynamicProgrammingController._vf_update(
#                             demand_prob,
#                             min_demand,
#                             max_demand,
#                             ce,
#                             h,
#                             b,
#                             state,
#                             vf,
#                             actions,
#                         )[1]
#                         if qa is not None:
#                             qf[state] = qa

#                     break


#         self.qf = qf
#         self.vf = val

#         end_time = datetime.now()
#         duration = end_time - start_time
#         logger.info(f"Dynamic programming completed at {end_time}")
#         logger.info(f"Total training duration: {duration}")
#         logger.info(
#             f"Final best cost: {self.get_periodic_average_cost(self.sourcing_model, sourcing_periods=1000, seed=42):.4f}"
#         )

#     def predict(
#         self,
#         current_inventory: Union[int, torch.Tensor],
#         past_regular_orders: Optional[Union[TypingList[int], torch.Tensor]] = None,
#         past_expedited_orders: Optional[Union[TypingList[int], torch.Tensor]] = None,
#         phase: int = 0,
#         output_tensor: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
#         """

#         Parameters
#         ----------
#         current_inventory : int, or torch.Tensor
#             Current inventory.
#         past_regular_orders : list, or torch.Tensor, optional
#             Past regular orders. If the length of `past_regular_orders` is lower than `regular_lead_time`, it will be padded with zeros. If the length of `past_regular_orders` is higher than `regular_lead_time`, only the last `regular_lead_time` orders will be used during inference.
#         past_expedited_orders : list, or torch.Tensor, optional
#             Past expedited orders. Since `expedited_lead_time` is assumed to be 0 for DynamicProgrammingController and the batch size is assumed to be 1, the input of `past_expedited_orders` is optional and will be ignored.
#         output_tensor : bool, default is False
#             If True, the replenishment order quantity will be returned as a torch.Tensor. Otherwise, it will be returned as an integer.

#         """
#         if self.sourcing_model is None:
#             raise AttributeError("The controller is not trained.")

#         regular_lead_time = self.sourcing_model.get_regular_lead_time()

#         current_inventory = self._check_current_inventory(current_inventory)
#         past_regular_orders = self._check_past_orders(
#             past_regular_orders, regular_lead_time
#         )

#         first = (
#             current_inventory.squeeze()
#             + past_regular_orders.squeeze()[-regular_lead_time]
#         )
#         second = past_regular_orders.squeeze()[-regular_lead_time + 1 :]

#         key = tuple([int(first)] + second.int().tolist() + [phase])

#         if output_tensor:
#             return torch.tensor([[self.qf[key][0]]]), torch.tensor([[self.qf[key][1]]])
#         else:
#             return self.qf[key]

#     def reset(self) -> None:
#         self.qf = None
#         self.vf = None
#         self.sourcing_model = None