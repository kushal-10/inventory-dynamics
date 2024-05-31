---
title: 'idinn: A Python Package for Inventory-Dynamics Control with Neural Networks'
tags:
  - Python
  - PyTorch
  - artificial neural networks
  - inventory dynamics
  - optimization
  - control
  - dynamic programming
authors:
  - name: Jiawei Li
    affiliation: 1
    corresponding: true
  - name: Thomas Asikis
    orcid: 0000-0003-0163-4622
    affiliation: 2
  - name: Ioannis Fragkos
    affiliation: 3
  - name: Lucas B\"ottcher
    affiliation: "1,4"
    orcid: 0000-0003-1700-1897
affiliations:
 - name: Department of Computational Science and Philosophy, Frankfurt School of Finance and Management
   index: 1
 - name: Game Theory, University of Zurich
   index: 2
 - name: Department of Technology and Operations Management, Rotterdam School of Management, Erasmus University Rotterdam
   index: 3
 - name: Laboratory for Systems Medicine, Department of Medicine, University of Florida
   index: 4
date: 7 May 2024
bibliography: paper.bib

---

# Summary

Identifying optimal policies for replenishing inventory from multiple suppliers is a key problem in inventory management. Solving such optimization problems means that one must determine the quantities to order from each supplier based on the current inventory and outstanding orders, minimizing the expected backlogging, holding, and sourcing costs. Despite over 60 years of extensive research on inventory management problems, even fundamental dual-sourcing problems [@barankin1961delivery,@fukuda1964optimal]—where orders from an expensive supplier arrive faster than orders from a regular supplier—remain analytically intractable. Additionally, there is a growing interest in optimization algorithms that are capable of handling real-world inventory problems with non-stationary demand [@song2020capacity].

We provide a Python package, `idinn`, implementing inventory dynamics–informed neural networks designed for controlling both single-sourcing and dual-sourcing problems. In single-sourcing problems, a single supplier delivers an ordered quantity to the inventory manager within a known lead time and at a known unit cost. Dual-sourcing problems are similar to single-sourcing problems but tend to be more complex. In a dual-sourcing problem, a company has two potential suppliers for a product, each offering different, but known lead times (the duration for orders to arrive) and unit order costs (the expense of ordering a single item). The company’s decision problem is to determine how much to order from each of the two suppliers at the beginning of each period, given the history of past orders. The objective is to minimize jointly the expected ordering, holding, and stock-out costs over a finite or infinite horizon. In `idinn`, neural network controllers and inventory dynamics are implemented as customizable objects with a PyTorch backend, enabling users to identify near-optimal order policies.

The methods employed in `idinn` leverage advancements in automatic differentiation [@paszke2017automatic,@PaszkeGMLBCKLGA19] and the growing use of neural networks in dynamical system identification [@wang1998runge,@ChenRBD18,@fronk2023interpretable] and control [@asikis2022neural,@bottcher2022ai,@bottcher2022near,@mowlavi2023optimal,@bottcher2023gradient,@bottcher2024control]. 

# Statement of need

Inventory management problems arise in several industries, such as manufacturing, retail, hospitality, fast-fashion, warehousing operations, and energy. A basic and yet analytically intractable problem in inventory management is dual sourcing [@barankin1961delivery,@fukuda1964optimal,@xin2023dual]. `idinn` is a Python package for controlling dual-sourcing inventory dynamics with dynamics-informed neural networks. The classical dual-sourcing problem we consider is usually formulated as an infinite-horizon problem focused on minimizing average costs while accounting for stationary stochastic demand. Employing neural networks, we minimize costs across several demand trajectories. This approach enables us to address not only non-stationary demand but also finite-horizon and infinite-horizon discounted problems. Unlike traditional reinforcement-learning approaches, our optimization approach takes into account how the system being optimized behaves over time, leading to more efficient training and accurate solutions. 

Training neural networks for inventory-dynamics control presents a specific challenge. The adjustment of neural network weights during training relies on propagating real-valued gradients, whereas the neural network outputs—representing replenishment orders—must be integers. To address this challenge in optimizing a discrete problem with real-valued gradient descent learning algorithms, we employ a problem-tailored straight-through estimator [@yang2022injecting,@asikis2023multi,@dyer2023]. This approach enables us to obtain integer-valued neural network outputs while backpropagating real-valued gradients.

`idinn` has been developed for researchers and students working at the intersection of optimization, operations research, and machine learning. It has been made available to students in a machine learning course at Frankfurt School and in a tutorial at California State University, Northridge, demonstrating the effectiveness of artificial neural networks in tackling real-world optimization problems. In a previous publication [@bottcher2023control], a less accessible code base has been used to compute near-optimal solutions of dozens of dual-sourcing instances. 

# Example usage

## Single-sourcing problems

The overall objective in single-sourcing and related inventory management problems is for companies to identify the optimal order quantities to minimize inventory-related costs, given stochastic demand. During periods when inventory remains after demand is met, each unit of excess inventory incurs a holding cost $h$. If the demand exceeds the available inventory in a period, the surplus demand is considered satisfied in subsequent periods, incurring a shortage cost $b$. This problem can be addressed using `idinn`. We first initialize the sourcing model and its associated neural network controller. Subsequently, we train the neural network controller using data generated from the sourcing model. Finally, we can use the trained neural network controller to compute near-optimal order quantities, which depends on the state of the system.

### Initialization

We use the `SingleSourcingModel` class to initialize a single-sourcing model. The single-sourcing model considered in this example has a lead time of 0, i.e., the order arrives immediately after it is placed, and initial inventory of 10. The holding cost, $h$, and the shortage cost, $b$, are 5 and 495, respectively. The demand is generated from a discrete uniform distribution with support $[1,4]$. We use batch size of 32 for training the neural network, i.e. the sourcing model generate 32 samples simmutanously.

In `idinn`, the sourcing model is initialized as follows.

```python
import torch
from idinn.sourcing_model import SingleSourcingModel
from idinn.controller import SingleSourcingNeuralController
from idinn.demand import UniformDemand

single_sourcing_model = SingleSourcingModel(
  lead_time=0,
  holding_cost=5,
  shortage_cost=495,
  batch_size=32,
  init_inventory=10,
  demand_generator=UniformDemand(low=1, high=4),
)
```

The cost at period $t$, $c_t$, is

$$
c_t = h \max(0, I_t) + b \max(0, - I_t)\,,
$$

where $I_t$ is the inventory level at the end of period $t$. The higher the holding cost, the more costly it is to keep the inventory postive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. The joint holding and stockout cost across all periods can be can be calculated using the `get_total_cost()` method of the sourcing model.

```python    
single_sourcing_model.get_total_cost()
```

In this example, this function should return 50 for each sample since the initial inventory is 10 and the holding cost is 5. We have 32 samples in this case, as we specified a batch size of 32.

The function `get_total_cost()`  is used, alongside the state dynamics of the system, to calculate the total cost over a given horizon. If an order takes $l$ periods to arrive, the information that describes the state of the system at the beginning of period $t$ is (i) the current inventory level, $s_t$, and (ii) the history of past orders that have not yet arrived, i.e., the vector $(q_{t-1}, q_{t-2}, \dots, q_{t-l})$. Thus, the state vector is $(s_t, q_{t-1}, q_{t-2}, \dots, q_{t-l})$. This vector is observed again at the beginning of period $t+1$. Until then, three events happen (in this order): (i) the order quantity $q_{t-l}$ arrives, (ii) we decide how much to order, $q_t$, and (iii) demand for the current period, $d_t$, is realized. Thus, the state in period $t+1$ is $(s_{t}+q_{t-l}-d_t, q_{t}, q_{t-1}, \dots, q_{t-l+1})$. This transition is probabilistic and depends on the realization of demand, $d_t$. In the function `get_total_cost()`, we sum over all the end-of-period inventory levels $s_t+q_{t-l}-d_t$. The interested reader is referred to @bottcher2023control for further details.  

For single-sourcing problems, we initialize the neural network controller using the `SingleSourcingNeuralController` class. For illustration purposes, we employ a simple neural network with 1 hidden layer and 2 neurons. The activation function is `torch.nn.CELU(alpha=1)`.

```python
single_controller = SingleSourcingNeuralController(
    hidden_layers=[2],
    activation=torch.nn.CELU(alpha=1)
)
```

### Training

Although the neural network controller has not been trained yet, we can still compute the total cost associated with its order policy. To do so, we integrate it with our previously specified sourcing model and run simulations for 100 periods.

```python    
single_controller.get_total_cost(
    sourcing_model=single_sourcing_model,
    sourcing_periods=100
)
```

Unsurprisingly, the performance is poor because we are only using the untrained neural network in which the weights are just (pseudo) random numbers. We can train the neural network controller using the `train()` method, in which the training data is generated from the given sourcing model. To better monitor the training process, we specify the `tensorboard_writer` parameter to log both the training loss and validation loss. For reproducibility, we also specify the seed of the underlying random number generator using the `seed` parameter.

```python
from torch.utils.tensorboard import SummaryWriter

single_controller.fit(
    sourcing_model=single_sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=5000,
    seed=1,
    tensorboard_writer=SummaryWriter(comment="single")
)
```

After training, we can use the trained neural network controller to calculate the total cost for 100 periods with our previously specified sourcing model. The total cost should be significantly lower than the cost associated with the untrained model.

```python
single_controller.get_total_cost(
  sourcing_model=single_sourcing_model,
  sourcing_periods=100
)
```

### Plotting and order calculation

We can inspect how the controller performs in the specified sourcing environment by plotting the inventory and order histories.

```python
# Simulate and plot the results
single_controller.plot(
  sourcing_model=single_sourcing_model,
  sourcing_periods=100
)
```

Then we can calculate optimal orders using the trained model.

```python
# Calculate the optimal order quantity for applications
single_controller.forward(
  current_inventory=10,
  past_orders=[1, 5]
)
```

## Dual-sourcing problems

Solving dual-sourcing problems with `idinn` is similar to the approached employed for single-sourcing problems described in the previous section.

### Initialization

To address dual-sourcing problems, we employ two main classes: (i) `DualSourcingModel` and (ii) `DualSourcingNeuralController`, responsible for setting up the sourcing model and its corresponding controller. In this example, we examine a dual-sourcing model characterized by the following parameters: the regular order lead time is 2; the expedited order lead time is 0; the regular order cost, $c_r$, is 0; the expedited order cost, $c_e$, is 20; and the initial inventory is 6. Additionally, the holding cost, $h$, and the shortage cost, $b$, are 5 and 495, respectively. Demand is generated from a discrete uniform distribution bounded on $[1, 4]$. In this example, we use a batch size of 256.

```python    
import torch
from idinn.sourcing_model import DualSourcingModel
from idinn.controller import DualSourcingNeuralController
from idinn.demand import UniformDemand

dual_sourcing_model = DualSourcingModel(
    regular_lead_time=2,
    expedited_lead_time=0,
    regular_order_cost=0,
    expedited_order_cost=20,
    holding_cost=5,
    shortage_cost=495,
    batch_size=256,
    init_inventory=6,
    demand_generator=UniformDemand(low=1, high=4),
)
```

The cost at period `t`, `c_t`, is

$$
c_t = c_r q^r_t + c_e q^e_t + h \max(0, I_t) + b \max(0, - I_t)\,,
$$

where $I_t$ is the inventory level at the end of period $t$, $q^r_t$ is the regular order placed at period $t$, and $q^e_t$ is the expedited order placed at period $t$. The higher the holding cost, the more costly it is to keep the inventory positive and high. The higher the shortage cost, the more costly it is to run out of stock when the inventory level is negative. The higher the regular and expedited order costs, the more costly it is to place the respective orders. The cost can be calculated using the `get_total_cost()` method of the sourcing model.

```python    
dual_sourcing_model.get_total_cost(regular_q=0, expedited_q=0)
```

In this example, this function should return 30 for each sample since the initial inventory is 6, the holding cost is 5, and there is neither a regular nor expedited order. We have 256 samples in this case, as we specified a batch size of 256.

Similarly to the single-sourcing case, the state of the system can be described by the inventory level and the history of past orders. However, because now there are two suppliers in the system, we need to include the order history of both suppliers. Therefore, the state of the system can be written as $(s_t, q^r_{t-1}, \dots, q^r_{t-l_r}, q^e_{t-1}, \dots, q^e_{t-l_e})$. The state is updated in a similar way as in single-sourcing models because the events that take place within a period are the same: past orders arrive, new orders are placed, and demand is observed.The new state is therefore $(s_t+q^r_{t-l_r}+q^e_{t-l_e}-d_t, q^r_t, \dots, q^r_{t-l_r+1}, q^e_t, \dots, q^e_{t-l_e+1})$. Then, depending on the inventory level at the end of a period, we pay for holding or stock-out cost, while we also record the ordering cost from each supplier. The interested reader is refered to @bottcher2023control for further details. 

For dual-sourcing problems, we initialize the neural network controller using the `DualSourcingNeuralController` class. We use a simple neural network with 6 hidden layers. The numbers of neurons in each layer are 128, 64, 32, 16, 8, and 4, respectively. The activation function is `torch.nn.CELU(alpha=1)`.

```python
dual_controller = DualSourcingNeuralController(
    hidden_layers=[128, 64, 32, 16, 8, 4],
    activation=torch.nn.CELU(alpha=1)
)
```

### Training

In the same fashion of the previous section, we can evaluate the performance of an untrained dual-sourcing controller by integrating it with our previously specified dual-sourcing model and running simulations for 100 periods.

```python
dual_controller.get_total_cost(
  sourcing_model=dual_sourcing_model,
  sourcing_periods=100
)
```

To improve the initially poor performance, we can train the neural network controller using the `train()` method.

```python
from torch.utils.tensorboard import SummaryWriter

dual_controller.fit(
    sourcing_model=dual_sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=1000,
    tensorboard_writer=SummaryWriter(comment="single"),
    seed=3
)
```

The total cost of the trained controller over 100 periods can be computed as follows.

```python    
dual_controller.get_total_cost(
  sourcing_model=dual_sourcing_model,
  sourcing_periods=100
)
```

### Plotting, and order calculation

We can further examine the controller's performance in the specified sourcing environment by plotting the inventory and order histories.

```python
# Simulate and plot the results
dual_controller.plot(
  sourcing_model=dual_sourcing_model,
  sourcing_periods=100
)
```

Then we can use the trained network to calculate near-optimal orders.

```python
# Calculate the optimal order quantity for applications
regular_q, expedited_q = dual_controller.forward(
    current_inventory=10,
    past_regular_orders=[1, 5],
    past_expedited_orders=[0, 0],
)
```

### Save and load the model

To save and load a given model, one can use the `save()` and `load()` methods, respectively.

```python
# Save the model
dual_controller.save("optimal_dual_sourcing_controller.pt")
# Load the model
dual_controller_loaded = DualSourcingNeuralController(
    hidden_layers=[128, 64, 32, 16, 8, 4],
    activation=torch.nn.CELU(alpha=1)
)
dual_controller_loaded.load("optimal_dual_sourcing_controller.pt")
```

# Acknowledgements

LB acknowledges financial support from hessian.AI and the Army Research Office (grant W911NF-23-1-0129). TA acknowledges financial support from the Schweizerischer Nationalfonds zur Förderung der Wissenschaf­tlichen Forschung (NCCR Automation) (grant P2EZP2 191888).


# References