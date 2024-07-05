import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from  sklearn.metrics import mean_squared_error as mse
regr = RandomForestRegressor(max_depth=200, random_state=0)
regr = MLPRegressor((100, 100, 100))


c13k_fp = "./c13k_selections.csv"
c13k = pd.read_csv(c13k_fp)
minimal = c13k[['Ha', 'pHa', 'Hb', 'pHb', 'Lb','LotNumB', 'Amb', 'bRate', 'bRate_std']]

minimal_array = minimal.to_numpy()
X_train = minimal_array[:,0:6][1:10000]
y_train = minimal_array[:,7:9][1:10000]
X_test = minimal_array[:,0:6][10000:12000]
y_test = minimal_array[:,7:9][10000:12000]

regr.fit(X_train, y_train)
plt.scatter(regr.predict(X_test)[:,0], y_test[:,0])
plt.scatter(regr.predict(X_test)[:,1], y_test[:,1])
plt.show()




### Pyro Coding ---- Manisha 
# Regression model
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
