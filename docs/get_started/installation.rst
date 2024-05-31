************
Installation
************

Requirements
============

The basic usage of `idinn` requires a working `Python`_ and `PyTorch`_ installation. If plotting simulation result of a controller is needed, `matplotlib`_ should also be installed. We recommend using the following versions for maximum compability:

* Python_     ``>= 3.7``
* PyTorch_    ``>= 2.0``
* matplotlib_ ``>= 3.0``

Install `idinn`
===============

The package can be installed form PyPI. To do that, run

.. code-block::

   pip install idinn

Or, if you want to inspect and locally edit the source code, run

.. code-block::

   git clone https://gitlab.com/ComputationalScience/idinn.git
   cd idinn
   pip install -e .

.. _Python: https://www.python.org/downloads/
.. _PyTorch: https://pytorch.org/get-started/locally/
.. _matplotlib: https://matplotlib.org/stable/users/getting_started/