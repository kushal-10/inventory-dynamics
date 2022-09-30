import numpy as np
from collections import namedtuple


def load_experiment_conf(filename: str):
    """
    Sample file structure:
    100 (c_e)
    0   (c_r)
    0   (le)
    2   (lr)
    5   (h)
    95  (b)
    0   (min demand)
    4   (max demand)
    """
    data = namedtuple('data', 'c_e c_r l_e l_r h b demand')
    demand = namedtuple('demand', 'min max support')

    with open(filename, 'r') as f:
        data.c_e = float(f.readline())
        data.c_r = float(f.readline())
        data.l_e = float(f.readline())
        data.l_r = float(f.readline())
        data.h = float(f.readline())
        data.b = float(f.readline())
        d_min = float(f.readline())
        d_max = float(f.readline())
        # demand can be modeled in a better way (needs to be a class)
        # I will keep it like this for now
        demand.min = d_min
        demand.max = d_max
        support = d_max - d_min
        demand.support = support
        demand.prob = dict(zip(np.arange(d_min, d_max + 1), np.repeat(1 / (support + 1), support + 1)))
        if 'fc' in filename:
            data.f_e = float(f.readline())
            data.f_r = float(f.readline())
    data.demand = demand
    return data