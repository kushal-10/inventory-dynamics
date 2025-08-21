# idinn: Inventory-Dynamics Control with Neural Networks

[![PyPI Latest Release](https://img.shields.io/pypi/v/idinn.svg)](https://pypi.org/project/idinn/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BAMiveGXmErIp10MK3V_SUJlDAXHAyaI)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://idinn-demo.streamlit.app)
[![status](https://joss.theoj.org/papers/224380be40f3be0b741a4ec711eac83b/status.svg)](https://joss.theoj.org/papers/224380be40f3be0b741a4ec711eac83b)

[<img src="https://gitlab.com/ComputationalScience/idinn/-/raw/main/docs/_static/youtube.png" align="center" width="60%" size="auto" alt="youtube">](https://www.youtube.com/watch?v=hUBfTWV6tWQ)

`idinn` implements **i**nventory **d**ynamics–**i**nformed **n**eural **n**etwork and other related controllers for solving single-sourcing and dual-sourcing problems. Neural network controllers and inventory dynamics are implemented into customizable objects using PyTorch as backend to enable users to find the optimal controllers for the user-specified inventory systems.

## Demo

For a quick demo, you can run our [Streamlit app](https://idinn-demo.streamlit.app/). The app allows you to interactively train and evaluate neural controllers for user-specified dual-sourcing systems. Alternatively, you may use our notebook in [Colab](https://colab.research.google.com/drive/1BAMiveGXmErIp10MK3V_SUJlDAXHAyaI).

## Installation

The package can be installed from PyPI. To do that, run

```
pip install idinn
```

Or, if you want to inspect the source code and edit locally, run

```
git clone https://gitlab.com/ComputationalScience/idinn.git
cd idinn
pip install -e .
```

## Example Usage

```python
import torch
from idinn.sourcing_model import SingleSourcingModel
from idinn.single_controller import SingleSourcingNeuralController
from idinn.demand import UniformDemand

# Initialize the sourcing model and the neural controller
sourcing_model = SingleSourcingModel(
    lead_time=0,
    holding_cost=5,
    shortage_cost=495,
    batch_size=32,
    init_inventory=10,
    demand_generator=UniformDemand(low=1, high=4),
)
controller = SingleSourcingNeuralController(
    hidden_layers=[2],
    activation=torch.nn.CELU(alpha=1)
)
# Train the neural controller
controller.fit(
    sourcing_model=sourcing_model,
    sourcing_periods=50,
    validation_sourcing_periods=1000,
    epochs=5000,
    seed=1,
)
# Simulate and plot the results
controller.plot(sourcing_model=sourcing_model, sourcing_periods=100)
# Calculate the optimal order quantity for applications
controller.predict(current_inventory=10)
```

## Documentation

See the official [documentation](https://inventory-optimization.readthedocs.io/en/latest/) for more information.

## Contribution

We welcome contributions of all kinds! To get started, please see our [contribution guide](CONTRIBUTING.md).

## Papers using `idinn`

* Böttcher, Lucas, Thomas Asikis, and Ioannis Fragkos. "Control of dual-sourcing inventory systems using recurrent neural networks." [INFORMS Journal on Computing](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.2022.0136) 35.6 (2023): 1308-1328.
* Li, Jiawei, Thomas Asikis, Ioannis Fragkos, and Böttcher, Lucas. "idinn: A Python package for inventory-dynamics control with neural networks." [Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.08508#) 10.112 (2025): 8508.


```
@article{bottcher2023control,
  title={Control of dual-sourcing inventory systems using recurrent neural networks},
  author={B{\"o}ttcher, Lucas and Asikis, Thomas and Fragkos, Ioannis},
  journal={INFORMS Journal on Computing},
  volume={35},
  number={6},
  pages={1308--1328},
  year={2023}
}
```

```
@article{li2025, 
title = {idinn: A {P}ython package for inventory-dynamics control with neural networks}, 
author = {Li, Jiawei and Asikis, Thomas and Fragkos, Ioannis and B{\"o}ttcher, Lucas}, 
journal = {Journal of Open Source Software},
volume = {10}, 
number = {112}, 
pages = {8508}, 
year = {2025}
}
```

## Contributors

* [Jiawei Li](https://github.com/iewaij)
* [Thomas Asikis](https://gitlab.com/asikist)
* [Ioannis Fragkos](https://gitlab.com/ioannis.fragkos1)
* [Lucas Böttcher](https://gitlab.com/lucasboettcher)
