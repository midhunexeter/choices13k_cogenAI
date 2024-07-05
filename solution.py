import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from  sklearn.metrics import mean_squared_error as mse
regr = RandomForestRegressor(max_depth=200, random_state=0)


c13k_fp = "./c13k_selections.csv"
c13k = pd.read_csv(c13k_fp)
minimal = c13k[['Ha', 'pHa', 'Hb', 'pHb', 'Lb','LotNumB', 'Amb', 'bRate', 'bRate_std']]

minimal_array = minimal.to_numpy()
print(minimal_array.shape)


X_train = minimal_array[:,0:6][1:10000]
y_train = minimal_array[:,7:9][1:10000]

X_test = minimal_array[:,0:6][10000:12000]
y_test = minimal_array[:,7:9][10000:12000]



regr.fit(X_train, y_train)
plt.scatter(regr.predict(X_test)[:,0], y_test[:,0])
plt.scatter(regr.predict(X_test)[:,1], y_test[:,1])
plt.show()