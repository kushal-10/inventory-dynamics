from abc import abstractmethod

import pandas as pd
import torch


class AbstractDemandGenerator:

    @abstractmethod
    def generate_sample(self, t: int, n_samples: int) -> torch.Tensor:
        """
        Generates a demand sample from the provided generator.

        Parameters
        ----------
        t: int
            The timestep to generate samples for. This will be used for time-dependent demand generators.
        n_samples: int
            The number of samples :math:`N` to generate.

        Returns
        -------
        D: torch.Tensor
            As we assume our dynamics are sampled in mini-batches for training,
            then this method will generate a column  vector of size :math:`N \times 1`.

        """
        pass

    @abstractmethod
    def sample_trajectory(self, t: int, n_samples: int, n_timesteps: int):
        """
        Samples a whole demand trajectory, to speed up training and loading.

        Parameters
        ----------
        t: int
            Start time to sample from, often 0.
        n_samples: int
            Number of samples to generate.
        n_timesteps: int
            Number of timesteps in trajectory.

        Returns
        -------
        D-traj: torch.Tensor
            The sampled trajectories tensor, which has shape :math:`N \times T \times 1`
        """
        pass

class TorchDistDemandGenerator(AbstractDemandGenerator):

    def __init__(self, distribution: torch.distributions.Distribution = torch.distributions.Uniform(low=0, high=4 + 1)):
        """
        Generates demand based on a toch distribution

        Parameters
        ----------
        distribution: torch.distributions.Distribution
        """

        self.distribution = distribution

    def generate_sample(self, t: int, n_samples: int) -> torch.Tensor:
        D = self.distribution.sample([n_samples, 1]).round().int()
        return D

    def sample_trajectory(self, t:int, n_samples: int, n_timesteps: int) -> torch.Tensor:
        D_traj = self.distribution.sample([n_samples, n_timesteps, 1]).round().int()
        return D_traj


class FileBasedDemandGenerator(AbstractDemandGenerator):
    #TODO: load df and test
    def __init__(self, demand_file_path: str):
        """
        Generates demand based on a toch distribution

        Parameters
        ----------
        distribution: torch.distributions.Distribution
        """

        self.demand_file_path = demand_file_path
        self.demand_df = pd.read_csv(demand_file_path)

    def generate_sample(self, t: int, n_samples: int) -> torch.Tensor:
        D = self.distribution([n_samples, 1]).int()
        return D

    def sample_trajectory(self, t:int, n_samples: int, n_timesteps: int) -> torch.Tensor:
        D_traj = self.distribution.sample([n_samples, n_timesteps, 1]).round().int()
        return D_traj