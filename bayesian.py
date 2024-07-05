### Pyro Coding ---- Manisha 
# Bayesian Regression model

import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from  sklearn.metrics import mean_squared_error as mse
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from torch import nn

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.9.1')
pyro.set_rng_seed(1)


# # Set matplotlib settings
# %matplotlib inline
# plt.style.use('default')

c13k_fp = "./c13k_selections.csv"
c13k = pd.read_csv(c13k_fp)
minimal = c13k[['Ha', 'pHa', 'Hb', 'pHb', 'Lb', 'bRate', 'bRate_std']]

minimal_array = minimal.to_numpy()
X_train = torch.tensor(minimal_array[:,0:5][1:10000])
y_train = torch.tensor(minimal_array[:,5:][1:10000])
X_test = torch.tensor(minimal_array[:,0:5][10000:12000])
y_test = torch.tensor(minimal_array[:,5:][10000:12000])

print(X_train.dtype)
print(y_train.dtype)
print(X_test.dtype)
print(y_test.dtype)

linear_reg_model = PyroModule[nn.Linear](3, 1)

# Define loss and optimize
loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
num_iterations = 1500 if not smoke_test else 2

def train():
    # run the model forward on the data
    y_pred = linear_reg_model(X_train).squeeze(-1)
    # calculate the mse loss
    loss = loss_fn(y_pred, y_train)
    # initialize gradients to zero
    optim.zero_grad()
    # backpropagate
    loss.backward()
    # take a gradient step
    optim.step()
    return loss

for j in range(num_iterations):
    loss = train()
    if (j + 1) % 50 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))


# Inspect learned parameters
print("Learned parameters:")
for name, param in linear_reg_model.named_parameters():
    print(name, param.data.numpy()) 

