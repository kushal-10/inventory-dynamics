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

    def enumerate_support(self):
        pass
    
    def get_min_demand(self):
        pass
    
    def get_max_demand(self):
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
        return {x: 1/(self.max_demand + 1 - self.min_demand) for x in range(
            self.min_demand, self.max_demand + 1)}
    
    def get_min_demand(self):
        return self.min_demand
    
    def get_max_demand(self):
        return self.max_demand


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
