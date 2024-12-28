import numpy as np

# MSE cost function (linear regression)
# Note: This function is convex
def linear_reg_mse(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / m) * np.sum((y - predictions) ** 2)
    return cost

def linear_reg_mse_gradient(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    gradient = -(2 / m) * X.T.dot(y - predictions)
    return gradient

# local minima function and gradient
def local_minima_fun(x):
    return 0.015 * x**4 - 0.05 * x**3 - 0.35 * x**2 + 0.05 * x

def local_minima_gradient(x):
    a = np.power(0.015, 1 / 4)
    b = np.power(0.05, 1 / 3)
    c = np.power(0.35, 1 / 2)
    return 4 * (a**4) * (x**3) - 3 * (b**3) * (x**2) - 2 * (c**2) * x + 0.05

def local_minima_mse(X, y, theta):
    m = len(y)
    residual = y - theta * X
    return 1 / m * np.sum(0.0015 * residual**4 - 0.005 * residual**3 - 0.035 * residual**2 + 0.005 * residual)

def local_minima_mse_gradient(X, y, theta):
    residual = y - theta * X
    gradient = (-0.006 * residual**3 + 0.015 * residual**2 + 0.07 * residual - 0.005) * X
    return np.mean(gradient)


#saddle point function and gradient
def saddle_point_fun(X):
    return X[0]**2 - X[1]**2 + 0.05 * X[1]**4

def saddle_point_gradient(X):
    return np.array([2 * X[0], -2 * X[1] + 0.2 * X[1]**3])

# relu activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
