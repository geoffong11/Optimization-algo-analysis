import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
from collections import deque
from mpl_toolkits.mplot3d import Axes3D

def plot_all_loss_graph(loss_dict):
    sns.set_palette("colorblind")
    for algo in loss_dict:
        iter_lst = deque()
        loss_lst = deque()
        for key in loss_dict[algo]:
            iter_lst.append(key)
            loss_lst.append(loss_dict[algo][key])
        plt.plot(iter_lst, loss_lst, label=algo)
    plt.legend()
    plt.show()


def plot_3d_graph(fun, fun_name):
    x1_val = np.linspace(-3, 3, 100)
    x2_val = np.linspace(-4.5, 4.5, 100)
    x1, x2 = np.meshgrid(x1_val, x2_val)
    # Initialize cost matrix
    cost = np.zeros(x1.shape)

    # Calculate the cost for each combination of theta0 and theta1
    for i in range(x1.shape[0]):
        for j in range(x2.shape[1]):
            cost[i, j] = fun(np.array([x1[i, j], x2[i, j]]))
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, cost, cmap='viridis')

    # Labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(X)')
    ax.set_title(fun_name)
    plt.show()

def plot_graph(fun):
    X = np.linspace(-5.5, 7.5, 100)
    y = fun(X)
    plt.plot(X, y)
    plt.legend()
    plt.show()

def plot_regression_function(fun, fun_name):
    X = np.random.uniform(-10, 10, (100, 2))  # 100 samples, 2 features
    y = 3 * X[:, 0] + 2 * X[:, 1] # Linear relation with noise
    # Create a grid of theta values
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-10, 10, 100)
    theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)

    # Initialize cost matrix
    cost = np.zeros(theta0.shape)

    # Calculate the cost for each combination of theta0 and theta1
    for i in range(theta0.shape[0]):
        for j in range(theta0.shape[1]):
            theta = np.array([theta0[i, j], theta1[i, j]])  # Keep theta[2] constant (e.g., 0)
            cost[i, j] = fun(X, y, theta)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0, theta1, cost, cmap='viridis')

    # Labels and title
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Cost')
    ax.set_title(fun_name)

    plt.show()
