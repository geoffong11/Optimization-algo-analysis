import numpy as np
from collections import deque

def adagrad_optimizer(learning_rate, dw, db, t,
                    weights, biases, m_weights,
                    v_weights, m_biases, v_biases,
                    epsilon=10**-8):
    v_weights = v_weights + dw ** 2
    v_biases = v_biases + db**2
    weights -= learning_rate(t) * dw / (np.sqrt(v_weights) + epsilon)
    biases -= learning_rate(t) * db / (np.sqrt(v_biases) + epsilon)
    return weights, biases, m_weights, v_weights, m_biases, v_biases

def custom_fun_gd_adagrad(loss_fun, grad_fun, x, lr_fun, max_t=10**2, epsilon=10**-8, std=0):
    loss_per_epoch = {}
    if (isinstance(x, tuple)):
        x = np.array(x)
    loss_per_epoch[0] = loss_fun(x)
    min_loss = float('inf')
    v = np.zeros_like(x)
    for t in range(1, max_t + 1):
        gradient = grad_fun(x)
        if (std > 0):
            if isinstance(gradient, np.ndarray):
                gradient = gradient.astype(np.float64)
                gradient = gradient + np.random.normal(0, std, size=gradient.shape)
            else:
                gradient = gradient + np.random.normal(0, std)
        v = v + gradient**2
        x = x - lr_fun(t) * gradient / (np.sqrt(v) + epsilon)
        loss = loss_fun(x)
        loss_per_epoch[t] = loss
        min_loss = np.min([min_loss, loss])
    return min_loss, loss_per_epoch
    pass

def gradient_descent_adagrad(loss_fun, grad_fun, X, y, theta, lr_fun, max_t=10**2, epsilon=10**-6, batch_size=32):
    v = 0
    n = X.shape[0]
    num_iter = 0
    loss_per_epoch = {}
    loss_per_epoch[0] = loss_fun(X, y, theta)
    for t in range(max_t):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, n, batch_size):
            num_iter += 1
            X_batch = X_shuffled[i: i + batch_size]
            y_batch = y_shuffled[i: i + batch_size]
            gradient = grad_fun(X_batch, y_batch, theta)
            v = v + gradient**2
            theta = theta - lr_fun(num_iter) * gradient / (np.sqrt(v) + epsilon)
            loss = loss_fun(X_shuffled, y_shuffled, theta)
            loss_per_epoch[t + (i + batch_size) / n] = loss
            
    return theta, loss, loss_per_epoch