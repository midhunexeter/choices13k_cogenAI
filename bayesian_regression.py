import logging
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

pyro.set_rng_seed(1)
assert pyro.__version__.startswith('1.9.1')



def model(beta, hB, b_rate):
    beta = pyro.sample("beta", dist.Normal(0., 1))
    c = pyro.sample("beta", dist.Normal(0., 1))
    hB = pyro.sample("hB", dist.Normal(0., ))
    brate_mean = beta*hB+c
    
    with pyro.plate("data", len(b_rate)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=b_rate)