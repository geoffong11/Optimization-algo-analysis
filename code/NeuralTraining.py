from NeuralNetwork import NeuralNetwork
from functions import relu, relu_derivative, sigmoid, sigmoid_derivative
from GDAdam import adam_optimizer
from GDAdagrad import adagrad_optimizer
from GDRMSProp import rmsprop_optimizer
from GDMiniBatch import vanilla_optimizer
from GDMomentum import momentum_optimizer
from graph import plot_all_loss_graph

import numpy as np

def optimizer_analysis(X, y):
    layer_sizes = [25, 55, 20, 104, 10, 150, 1]
    init_lr = 0.0001
    # Neural Networks
    adam_nn = NeuralNetwork(layer_sizes, sigmoid, sigmoid_derivative, adam_optimizer, lambda x: init_lr)
    rmsprop_nn = NeuralNetwork(layer_sizes, sigmoid, sigmoid_derivative, rmsprop_optimizer, lambda x: init_lr)
    adagrad_nn = NeuralNetwork(layer_sizes, sigmoid, sigmoid_derivative, adagrad_optimizer, lambda x: init_lr)
    vanilla_nn = NeuralNetwork(layer_sizes, sigmoid, sigmoid_derivative, vanilla_optimizer, lambda x: init_lr / np.sqrt(x))
    momentum_nn = NeuralNetwork(layer_sizes, sigmoid, sigmoid_derivative, momentum_optimizer, lambda x: init_lr / np.sqrt(x))
    batch_size = 32
    epochs = 25
    loss_dict = {}
    nn_dict = {
               "rmsprop": rmsprop_nn,
               "adam": adam_nn,
               "vanilla": vanilla_nn,
               "momentum": momentum_nn,
               "adagrad": adagrad_nn
               }
    for optim_name, nn in nn_dict.items():
        loss_per_epoch = nn.train(X, y, epochs=epochs, batch_size=batch_size)
        loss_dict[optim_name] = loss_per_epoch
    plot_all_loss_graph(loss_dict)

