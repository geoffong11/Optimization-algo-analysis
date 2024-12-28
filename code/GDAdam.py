import numpy as np
from collections import deque

def adam_optimizer(learning_rate, dw, db, t,
                    weights, biases, m_weights,
                    v_weights, m_biases, v_biases,
                    adam_constant_1=0.9,adam_constant_2=0.9, 
                    epsilon=10**-8):
    m_weights = adam_constant_1 * m_weights + (1 - adam_constant_1) * dw
    v_weights = adam_constant_2 * v_weights + (1 - adam_constant_2) * (dw ** 2)
    m_biases = adam_constant_1 * m_biases + (1 - adam_constant_1) * db
    v_biases = adam_constant_2 * v_biases + (1 - adam_constant_2) * (db ** 2)
    m_hat_weights = m_weights / (1 - adam_constant_1 ** t)
    v_hat_weights = v_weights / (1 - adam_constant_2 ** t)
    m_hat_biases = m_biases / (1 - adam_constant_1 ** t)
    v_hat_biases = v_biases / (1 - adam_constant_2 ** t)
    weights -= learning_rate(t) * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)
    biases -= learning_rate(t) * m_hat_biases / (np.sqrt(v_hat_biases) + epsilon)
    
    return weights, biases, m_weights, v_weights, m_biases, v_biases
    pass

def custom_fun_gd_adam(loss_fun, grad_fun, x, lr_fun, adam_constant_1=0.9, adam_constant_2=0.9, max_t=10**2, epsilon=10**-8, std=0):
    loss_per_epoch = {}
    loss_per_epoch[0] = loss_fun(x)
    v = np.zeros_like(x)
    m = np.zeros_like(x)
    num_iter = 1
    min_loss = float('inf')
    for t in range(1, max_t + 1):
        gradient = grad_fun(x)
        if (std > 0):
            if isinstance(gradient, np.ndarray):
                gradient = gradient.astype(np.float64)
                gradient = gradient + np.random.normal(0, std, size=gradient.shape)
            else:
                gradient = gradient + np.random.normal(0, std)
        v = adam_constant_1 * v + (1 - adam_constant_1) * gradient
        m = adam_constant_2 * m + (1 - adam_constant_2) * gradient**2
        vt = v / (1 - adam_constant_1**(num_iter))
        mt = m / (1 - adam_constant_2**(num_iter))
        x = x - lr_fun(t) * vt / (np.sqrt(mt) + epsilon)
        loss = loss_fun(x)
        num_iter += 1
        loss_per_epoch[t] = loss
        min_loss = np.min([min_loss, loss])
    return min_loss, loss_per_epoch

def gradient_descent_adam(loss_fun, grad_fun, X, y, theta, adam_constant_1, adam_constant_2, lr_fun, max_t=10**2, epsilon=10**-6, batch_size=32):
    v = 0
    m = 0
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
            v = adam_constant_1 * v + (1 - adam_constant_1) * gradient
            m = adam_constant_2 * m + (1 - adam_constant_2) * gradient**2
            vt = v / (1 - adam_constant_1**num_iter)
            mt = m / (1 - adam_constant_2**num_iter)
            theta = theta - lr_fun(num_iter) * vt / (np.sqrt(mt) + epsilon)
            loss = loss_fun(X_shuffled, y_shuffled, theta)
            loss_per_epoch[t + (i + batch_size) / n] = loss
    return theta, loss, loss_per_epoch