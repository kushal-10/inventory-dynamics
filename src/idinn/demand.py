import random
import torch
from abc import ABCMeta, abstractmethod


class BaseDemand(metaclass=ABCMeta):
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
        self.demand_prob = 1 / (high - low + 1)
        self.min_demand = low
        self.max_demand = high

    def sample(self, batch_size, batch_width=1) -> torch.Tensor:
        return self.distribution.sample([batch_size, batch_width]).int()
    
    def enumerate_support(self):
        return {x: 1/(self.high + 1 - self.low) for x in range(self.high + 1 - self.low)}
    
    def get_min_demand(self):
        return self.low
    
    def get_max_demand(self):
        return self.high


class CustomDemand(BaseDemand):
    def __init__(self, demand_prob=None):
        self.demand_prob = demand_prob

    def sample(self, batch_size, batch_width=1) -> torch.Tensor:
        """
        Generate demand for one period.

        Parameters
        ----------
        batch_size: int
            Size of generated demands which should correspond to the batch size or the number of SKUs. If the size does not match the dimension of the elements from `demand_history`, demand will be upsampled or downsampled to match the size.
        """
        # Draw dictionary keys with corresponding probabilities
        sampled_indices = torch.multinomial(
            torch.tensor(list(self.demand_prob.values())),
            num_samples=batch_size*batch_width,
            replacement=True
        )
        return torch.tensor(list(self.demand_prob.keys()))[sampled_indices].reshape(batch_size, batch_width)
    
    def enumerate_support(self):
        # TODO: check sum of probabilities is 1
        # TODO: check if all keys are int
        return self.demand_prob
    
    def get_min_demand(self):
        return min(self.demand_prob.keys())
    
    def get_max_demand(self):
        return max(self.demand_prob.keys())
        
    
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