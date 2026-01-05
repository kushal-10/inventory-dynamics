# File: src/idinn/demand_three_subperiods.py
# Purpose: Demand generator with multi-draw sampling for batch_width subperiods.

import torch
from torch.distributions.gamma import Gamma
from typing import Optional, Dict


class DemandGenerator:
    """Base demand generator interface.

    Must implement sample(batch_size, batch_width) -> Tensor shape (batch_size, batch_width)
    """

    def sample(self, batch_size: int, batch_width: int = 1) -> torch.Tensor:
        raise NotImplementedError


class UniformDemand(DemandGenerator):
    def __init__(self, low: int = 0, high: int = 4, device=None, dtype=torch.float32):
        self.low = low
        self.high = high
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    def sample(self, batch_size: int, batch_width: int = 1) -> torch.Tensor:
        # return integers in [low, high] inclusive, shape (batch_size, batch_width)
        # use uniform continuous then floor to integers (consistent with original code style)
        u = torch.rand(batch_size, batch_width, device=self.device, dtype=self.dtype)
        vals = (self.low + (self.high - self.low + 1) * u).floor().clamp(min=self.low, max=self.high)
        return vals

class DiscreteTruncatedGammaDemand(DemandGenerator):
    """
    Discrete, truncated Gamma demand generator.

    Demand takes integer values in {0, 1, ..., d_max}
    with probabilities derived from a Gamma distribution,
    truncated and renormalized.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        d_max: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        assert mean > 0, "Mean must be positive"
        assert std > 0, "Std must be positive"
        assert d_max >= 0, "d_max must be non-negative"

        self.mean = mean
        self.std = std
        self.d_max = d_max
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # ---- convert (mean, std) → Gamma(shape, scale) ----
        self.shape = (mean / std) ** 2
        self.scale = (std ** 2) / mean

        # ---- underlying Gamma distribution ----
        self.gamma = Gamma(
            concentration=torch.tensor(self.shape, device=self.device),
            rate=torch.tensor(1.0 / self.scale, device=self.device),
        )

        # ---- compute probabilities ----
        self.demand_prob = self._compute_demand_prob()
        self.probs_tensor = torch.tensor(
            list(self.demand_prob.values()),
            device=self.device,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------
    # Required DP / CustomDemand-style API
    # ------------------------------------------------------------------

    def enumerate_support(self) -> Dict[int, float]:
        return self.demand_prob

    def get_min_demand(self) -> int:
        return 0

    def get_max_demand(self) -> int:
        return self.d_max

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_demand_prob(self) -> Dict[int, float]:
        """
        Compute P(D = d) for d = 0, ..., d_max
        using Gamma CDF differences and renormalization.
        """
        probs = {}

        for d in range(self.d_max + 1):
            lower = max(d - 0.5, 0.0)
            upper = d + 0.5

            p = (
                self.gamma.cdf(torch.tensor(upper, device=self.device))
                - self.gamma.cdf(torch.tensor(lower, device=self.device))
            )
            probs[d] = float(p.item())

        # ---- renormalize to ensure sum = 1 ----
        total = sum(probs.values())
        for d in probs:
            probs[d] /= total

        return probs

    # ------------------------------------------------------------------
    # Sampling (same behavior as CustomDemand)
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, batch_width: int = 1) -> torch.Tensor:
        """
        Sample integer demand values in {0, ..., d_max}.
        """
        num_samples = batch_size * batch_width

        sampled_indices = torch.multinomial(
            self.probs_tensor,
            num_samples=num_samples,
            replacement=True,
        )

        return sampled_indices.view(batch_size, batch_width).to(self.dtype)


if __name__ == "__main__":
    g = UniformDemand(0, 4)
    print(g.sample(5, batch_width=3))

    gamma = DiscreteTruncatedGammaDemand(mean=6, std=3.4, d_max=4) # 0-4
    print(gamma.demand_prob)
    #{0: 0.004604733012744683, 1: 0.09151984267071936, 2: 0.22915648722810336, 3: 0.3218495298724488, 4: 0.3528694072159838}

