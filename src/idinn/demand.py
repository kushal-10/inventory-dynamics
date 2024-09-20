import random
import torch
from abc import ABCMeta, abstractmethod


class BaseDemand(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        """
        Reset the state of the demand generator.
        """
        pass

    @abstractmethod
    def sample(self, batch_size) -> torch.Tensor:
        """
        Generate demand for one period.

        Parameters
        ----------
        batch_size: int
            Size of generated demands which should correspond to the batch size or the number of SKUs.
        """
        pass


class UniformDemand(BaseDemand):
    def __init__(self, low, high):
        self.distribution = torch.distributions.Uniform(low=low, high=high + 1)
        # self.demand_prob = 1 / (high - low + 1)
        self.low = low
        self.high = high

    def reset(self):
        pass

    def sample(self, batch_size, batch_width=1) -> torch.Tensor:
        return self.distribution.sample([batch_size, batch_width]).int()
    
    # def enumerate_support(self):
    #     return dict(
    #         zip(
    #             torch.arange(self.low, self.high + 1),
    #             torch.repeat(1.0 / (self.high - self.low + 1.0), int(self.high - self.low + 1)),
    #         )
    #     )
    #     return {x: 1/(self.high - self.low) for x in range(self.high - self.low)}


# class CustomDemand(BaseDemand):
#     def __init__(self, demand_history=None, demdnd_prob=None):
#         """
#         Parameters
#         ----------
#         demand_history: Iterable, torch.Tensor, np.array, or pd.Series
#             A list or array of demand history. Each element in the array will be used as the demand for each time period.
#         """
#         self.demand_history = demand_history
#         self.counter = 0
#         # TODO: Fix this
#         self.demand_prob = {x: 1/len(demand_history) for x in range(len(demand_history))}

#     def reset(self):
#         self.counter = 0

#     def sample(self, batch_size) -> torch.Tensor:
#         """
#         Generate demand for one period.

#         Parameters
#         ----------
#         batch_size: int
#             Size of generated demands which should correspond to the batch size or the number of SKUs. If the size does not match the dimension of the elements from `demand_history`, demand will be upsampled or downsampled to match the size.
#         """
#         current_demand = torch.tensor(self.demand_history[self.counter])
#         # Ensure that current demand is non-negative
#         current_demand = torch.clamp(current_demand, min=0)
#         if current_demand.size() != torch.Size([batch_size, 1]):
#             if current_demand.dim() > 1:
#                 raise ValueError(
#                     f"The element of demand_history at index {self.counter} is not 1D and has a different dimension than the desired size [{batch_size}, 1]."
#                 )
#             elif current_demand.dim() == 0:
#                 current_demand = current_demand.expand(batch_size, 1)
#             elif current_demand.dim() == 1 and len(current_demand) == batch_size:
#                 current_demand = current_demand.unsqueeze(1)
#             else:
#                 idx = random.choices(range(len(current_demand)), k=batch_size)
#                 current_demand = current_demand[idx].unsqueeze(1)
#         self.counter += 1
#         # Reset the counter if it reaches the end of the demand history
#         if self.counter == len(self.demand_history):
#             self.reset()
#         return current_demand
    
#     def enumerate_support(self, x):
#         return self.demand_prob[x]
    
#     def custom_demand_distribution(demand_values: torch.Tensor, demand_probabilities: torch.Tensor, num_samples: int = 1):
#         """
#         Generates a custom demand distribution using the given demand values and their associated probabilities.
        
#         Parameters:
#         demand_values (torch.Tensor): A tensor of demand values.
#         demand_probabilities (torch.Tensor): A tensor of probabilities associated with the demand values. Must sum to 1.
#         num_samples (int): Number of samples to generate from the distribution.
        
#         Returns:
#         torch.Tensor: Tensor containing sampled demand values according to the given probability distribution.
#         """
#         # Ensure probabilities sum to 1
#         if not torch.isclose(demand_probabilities.sum(), torch.tensor(1.0)):
#             raise ValueError("Probabilities must sum to 1.")
        
#         # Sample from the demand values based on the probabilities
#         sampled_indices = torch.multinomial(demand_probabilities, num_samples, replacement=True)
        
#         # Return the corresponding demand values
#         return demand_values[sampled_indices]

#     # Example usage:
#     demand_values = torch.tensor([10, 20, 30, 40, 50])
#     demand_probabilities = torch.tensor([0.1, 0.3, 0.2, 0.25, 0.15])

#     # Generate 5 samples
#     samples = custom_demand_distribution(demand_values, demand_probabilities, num_samples=5)
#     print(samples)