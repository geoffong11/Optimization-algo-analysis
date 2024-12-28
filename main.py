from CustomPlot import custom_function_gd_comparison
from MSEPlot import dataset_comparison_between_descent
from functions import linear_reg_mse, linear_reg_mse_gradient
from functions import local_minima_fun, local_minima_gradient
from functions import saddle_point_fun, saddle_point_gradient
from graph import plot_3d_graph, plot_graph, plot_regression_function
from NeuralTraining import optimizer_analysis

import numpy as np
from sklearn.datasets import make_regression

def main():
    # 1) MSE
    # Plot the loss function of MSE
    plot_regression_function(linear_reg_mse, "MSE")

    # Plot the loss curve for the gradient descent methods
    n_features = 8
    n_samples = 600
    X, y, coef = make_regression(n_samples=n_samples, n_features=n_features, coef=True, noise=0.1)
    directions = np.random.rand(n_features)
    theta = coef + 2 * n_features * directions / np.linalg.norm(directions, ord=2)
    dataset_comparison_between_descent(linear_reg_mse, linear_reg_mse_gradient, X, y, theta,
                                    lambda x: 0.1 / np.sqrt(x), lambda x: 0.1,
                                    momentum_constant=0.9, rmsprop_constant=0.9,
                                    adam_constant_1=0.9, adam_constant_2=0.9,
                                    batch_size=16, max_t=4)

    # 2) Local Minima
    # Plot the function we want to minmize. Note that there is 1 local and 1 global minima
    plot_graph(local_minima_fun)

    # Plot the loss curve for the gradient descent methods (non-stochastic)
    custom_function_gd_comparison(local_minima_fun, local_minima_gradient, -7, lambda x: 0.01, lambda x: 0.01,
                                  momentum_constant=0.99, rmsprop_constant=0.9,
                                  adam_constant_1=0.99, adam_constant_2=0.9,
                                  max_t=1000)

    # Plot the loss curve for the gradient descent methods (stochastic)
    custom_function_gd_comparison(local_minima_fun, local_minima_gradient, -7, lambda x: 0.1 / np.sqrt(x), lambda x: 0.1,
                                  momentum_constant=0.99, rmsprop_constant=0.9,
                                  adam_constant_1=0.99, adam_constant_2=0.9,
                                  max_t=5000, std=3**2)

    # 3) Saddle point
    # Plot the 3d function we want to minimize. Note that there is a saddle point at (0, 0)
    plot_3d_graph(saddle_point_fun, "saddle point function")

    # Plot the loss curve for the gradient descent methods (non-stochastic, decaying)
    custom_function_gd_comparison(saddle_point_fun, saddle_point_gradient, (5, 0), lambda x: 0.01, lambda x: 0.01,
                                  momentum_constant=0.9, rmsprop_constant=0.9,
                                  adam_constant_1=0.9, adam_constant_2=0.9,
                                  max_t=700)

    # Plot the loss curve for the gradient descent methods (stochastic, decaying)
    custom_function_gd_comparison(saddle_point_fun, saddle_point_gradient, (5, 0), lambda x: 0.05 / np.sqrt(x), lambda x: 0.05,
                                  momentum_constant=0.9, rmsprop_constant=0.9,
                                  adam_constant_1=0.9, adam_constant_2=0.9,
                                  max_t=1500, std=0.25)
    
    # Neural Networks (we work on make_friedman1, a dataset that is non-linear regression)
    nn_features = 25
    nn_samples = 1024
    X_nn, y_nn = make_regression(n_samples=nn_samples, n_features=nn_features, noise=8, random_state=1)
    optimizer_analysis(X_nn, y_nn)
    pass 

if __name__ == "__main__":
    main()