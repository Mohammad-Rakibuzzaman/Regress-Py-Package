from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_iris
from regresspy.regression import Regression
from regresspy.loss import rmse

iris = load_iris()
# We will use sepal length to predict sepal width
X = iris.data[:, 0].reshape(-1, 1)
Y = iris.data[:, 1].reshape(-1, 1)



#TODO Perform a linear regression using sklearn and calculate training rmse.
# Use the SGDRegressor and only select set learning rate and epochs.

stochastic_gradient_descent = SGDRegressor(max_iter= 100, learning_rate= 'constant', eta0= 0.001)
stochastic_gradient_descent.fit(X, Y.reshape(-1))
sto_chas_grad_prediction = stochastic_gradient_descent.predict(X)
sto_chas_grad_rmse = rmse(sto_chas_grad_prediction, Y)
print('Stochastic Gradient Descent Regressor RMSE value:', str(sto_chas_grad_rmse))

# #TODO Perform a linear regression using your code and calculate training rmse.


regression_value = Regression(epochs= 20, learning_rate= 0.0001)
regression_value.fit(X, Y)
reg_pred = regression_value.predict(X)
#print(reg_pred.shape)
regression_rmse = regression_value.score(reg_pred, Y)
print('RMSE value of class: ', str(regression_rmse))