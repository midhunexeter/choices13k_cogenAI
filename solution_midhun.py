import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
from  sklearn.metrics import mean_squared_error as mse
# from sklearn.linear_model import ARDRegression, BayesianRidge, LinearRegression

regr = RandomForestRegressor(max_depth=20, random_state=30)
# regr = MLPRegressor((100, 100, 100, 100, 100, 100))
# regr = BayesianRidge(compute_score=True, max_iter=30)

c13k_fp = "./c13k_selections.csv"
c13k = pd.read_csv(c13k_fp)
minimal = c13k[['Ha', 'pHa', 'Hb', 'pHb','La', 'Lb','LotNumB', 'Amb', 'bRate', 'bRate_std']]

minimal_array = minimal.to_numpy()
X_train = minimal_array[:,0:8][1:10000]
y_train = minimal_array[:,8:10][1:10000]
X_test = minimal_array[:,0:8][10000:14000]
y_test = minimal_array[:,8:10][10000:14000]



regr.fit(X_train, y_train)
plt.figure(figsize=(3, 4))
plt.scatter(y_test[:,0], regr.predict(X_test)[:,0])
plt.xlabel('True')
plt.ylabel('Prediction')
plt.savefig('brate.pdf')


plt.figure(figsize=(3, 4))
plt.plot(regr.feature_importances_)
plt.xticks(ticks=range(8), labels=['Ha', 'pHa', 'Hb', 'pHb', 'La', 'Lb', 'LotNumB', 'Amb'], rotation=90)
plt.ylabel('Importance')
plt.savefig('fimport.pdf')



plt.figure(figsize=(3, 4))
plt.scatter(y_test[:,1], regr.predict(X_test)[:,1])
plt.xlabel('True')
plt.savefig('feature_imp.pdf')


plt.show()
# print(mse(regr.predict(X_test)[:,0], y_test[:,0])*100)
