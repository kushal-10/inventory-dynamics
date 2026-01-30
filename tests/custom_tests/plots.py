# Plots for visualizing the breaking points

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma

h = 5.0
b = 495.0   # backorder cost
lam = 5.0
c_e = 8.0
U = 10


# ---------- Expected cost functions ----------

def expected_cost_single_poisson(x, lam, h, b, max_d=50):
    d = np.arange(0, max_d)
    prob = poisson.pmf(d, lam)
    holding = h * np.maximum(x - d, 0)
    backlog = b * np.maximum(d - x, 0)
    return np.sum(prob * (holding + backlog))


def expected_cost_single_uniform(x, U, h, b):
    d = np.arange(0, U + 1)
    prob = np.ones_like(d) / len(d)
    holding = h * np.maximum(x - d, 0)
    backlog = b * np.maximum(d - x, 0)
    return np.sum(prob * (holding + backlog))


def expected_cost_single_gamma(x, shape, scale, h, b, max_d=200):
    # discretized Gamma
    d = np.arange(0, max_d)
    prob = gamma.pdf(d, a=shape, scale=scale)
    prob = prob / prob.sum()  # normalize
    holding = h * np.maximum(x - d, 0)
    backlog = b * np.maximum(d - x, 0)
    return np.sum(prob * (holding + backlog))


# ---------- Plotting ----------

def plot_single_cost_distributions():
    xs = np.arange(-10, 30)

    costs_uniform = [
        expected_cost_single_uniform(x, U=10, h=h, b=b) for x in xs
    ]

    costs_poisson = [
        expected_cost_single_poisson(x, lam=lam, h=h, b=b) for x in xs
    ]

    # Gamma with same mean as Poisson: mean = shape * scale = 5
    costs_gamma = [
        expected_cost_single_gamma(x, shape=2.0, scale=2.5, h=h, b=b)
        for x in xs
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Uniform
    axes[0].plot(xs, costs_uniform)
    axes[0].set_title("Uniform demand")
    axes[0].set_xlabel("Inventory level x")
    axes[0].set_ylabel("Expected one-period cost")

    # Poisson
    axes[1].plot(xs, costs_poisson)
    axes[1].set_title("Poisson demand")
    axes[1].set_xlabel("Inventory level x")

    # Gamma
    axes[2].plot(xs, costs_gamma)
    axes[2].set_title("Gamma demand (heavy tail)")
    axes[2].set_xlabel("Inventory level x")

    plt.suptitle("Single-source base-stock cost under different demand distributions")
    plt.tight_layout()
    plt.show()


def plot_differentials():
    xs = np.arange(-10, 30)

    costs_uniform = [expected_cost_single_uniform(x, U=10, h=h, b=b) for x in xs]
    costs_poisson = [expected_cost_single_poisson(x, lam=lam, h=h, b=b) for x in xs]
    costs_gamma = [
        expected_cost_single_gamma(x, shape=2.0, scale=2.5, h=h, b=b)
        for x in xs
    ]

    diff_uniform = np.diff(costs_uniform)
    diff_poisson = np.diff(costs_poisson)
    diff_gamma = np.diff(costs_gamma)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Uniform
    axes[0].plot(xs[:-1], diff_uniform)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("Uniform demand")
    axes[0].set_xlabel("Inventory level x")
    axes[0].set_ylabel("Δ Cost = C(x+1) − C(x)")

    # Poisson
    axes[1].plot(xs[:-1], diff_poisson)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Poisson demand")
    axes[1].set_xlabel("Inventory level x")

    # Gamma
    axes[2].plot(xs[:-1], diff_gamma)
    axes[2].axhline(0, linestyle="--")
    axes[2].set_title("Gamma demand (heavy tail)")
    axes[2].set_xlabel("Inventory level x")

    plt.suptitle("Marginal cost (finite difference) under different demand distributions")
    plt.tight_layout()
    plt.show()

"""
Dual Sourcing
"""

def expected_cost_Lr2_dynamic(x, q, U, h, b, c_e):
    """
    Two-period expected cost for Lr = 2 dual sourcing
    with optimal expediting in period t+1.
    """
    d_vals = np.arange(0, U + 1)
    prob = np.ones_like(d_vals) / len(d_vals)

    total_cost = 0.0

    for d, p_d in zip(d_vals, prob):
        # After period t demand
        inv_1 = x - d

        # Holding or backlog cost at t
        cost_t = h * max(inv_1, 0) + b * max(-inv_1, 0)

        # At t+1: decide optimal expediting
        # Regular arrives at t+2, so shortage at t+1 is critical
        expedite = max(-inv_1, 0)

        cost_exp = c_e * expedite

        # Inventory entering t+2
        inv_2 = inv_1 + expedite + q

        # Expected holding/backlog at t+2 (mean demand)
        expected_future = h * max(inv_2 - U/2, 0) + b * max(U/2 - inv_2, 0)

        total_cost += p_d * (cost_t + cost_exp + expected_future)

    return total_cost


def plot_dual():
    xs = np.arange(-10, 25)
    pipelines = [0, 5, 10, 15]

    plt.figure(figsize=(8, 6))

    for q in pipelines:
        costs = [expected_cost_Lr2_dynamic(x, q, U, h, b, c_e) for x in xs]
        plt.plot(xs, costs, label=f"pipeline q = {q}")

    plt.xlabel("Current inventory x")
    plt.ylabel("Expected total cost")
    plt.title("Dual sourcing with Lr = 2 (Uniform demand)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_single_cost_distributions()
    # plot_differentials()
    plot_dual()