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

    def reset(self):
        pass

    def sample(self, batch_size) -> torch.Tensor:
        return self.distribution.sample([batch_size, 1]).int()


class CustomDemand(BaseDemand):
    def __init__(self, demand_history):
        """
        Parameters
        ----------
        demand_history: Iterable, torch.Tensor, np.array, or pd.Series
            A list or array of demand history. Each element in the array will be used as the demand for each time period.
        """
        self.demand_history = demand_history
        self.counter = 0

    def reset(self):
        self.counter = 0

    def sample(self, batch_size) -> torch.Tensor:
        """
        Generate demand for one period.

        Parameters
        ----------
        batch_size: int
            Size of generated demands which should correspond to the batch size or the number of SKUs. If the size does not match the dimension of the elements from `demand_history`, demand will be upsampled or downsampled to match the size.
        """
        current_demand = torch.tensor(self.demand_history[self.counter])
        # Ensure that current demand is non-negative
        current_demand = torch.clamp(current_demand, min=0)
        if current_demand.size() != torch.Size([batch_size, 1]):
            if current_demand.dim() > 1:
                raise ValueError(
                    f"The element of demand_history at index {self.counter} is not 1D and has a different dimension than the desired size [{batch_size}, 1]."
                )
            elif current_demand.dim() == 0:
                current_demand = current_demand.expand(batch_size, 1)
            elif current_demand.dim() == 1 and len(current_demand) == batch_size:
                current_demand = current_demand.unsqueeze(1)
            else:
                idx = random.choices(range(len(current_demand)), k=batch_size)
                current_demand = current_demand[idx].unsqueeze(1)
        self.counter += 1
        # Reset the counter if it reaches the end of the demand history
        if self.counter == len(self.demand_history):
            self.reset()
        return current_demand
