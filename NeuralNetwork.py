import numpy as np
from WeightLayer import WeightLayer

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_fun, activation_grad, optimizer, learning_rate):
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            init_weights = np.zeros((input_size, output_size))
            init_biases = np.zeros((1, output_size))
            for i in range(input_size):
                if (i % 2 == 0 or (i + 1) % 5 == 0):
                    init_weights[i, :] = np.linspace(0.1 / output_size, np.sqrt(i + 1) / output_size, output_size)
                else:
                    init_weights[i, :] = np.linspace(0.2 / (13 * output_size), np.cbrt(1.5 * i + 1) / output_size, output_size)
            for j in range(output_size):
                if (j % 2 == 0 or (j + 1) % 7 == 0):
                    init_biases[0, j] = 0.1 - (j / output_size)
                else:
                    init_biases[0, j] = 0.1 + (j / output_size)
            # init_weights = np.random.randn(input_size, output_size)
            # init_biases = np.random.randn(1, output_size)
            # init_weights = np.full((input_size, output_size), 0.1)
            #init_biases = np.full((1, output_size), 0.1)
            layer = WeightLayer(
                init_weights, init_biases, activation_fun, activation_grad, learning_rate, optimizer
            )
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y_true, y_pred):
        full_loss_gradient = 2 * (y_pred.reshape(y_pred.shape[0]) - y_true) / y_true.size
        loss_gradient = np.mean(full_loss_gradient)
        # Propagate the gradient back through each layer
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)

    def train(self, X, y, epochs=100, batch_size=16):
        loss_per_epoch = {}
        loss_per_epoch[0] = np.mean((y - self.forward(X)) ** 2)
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                loss = np.mean((y - self.forward(X)) ** 2)
                loss_per_epoch[epoch + (i + batch_size) / len(X)] = loss
        return loss_per_epoch

    def predict(self, X):
        return self.forward(X)
    

