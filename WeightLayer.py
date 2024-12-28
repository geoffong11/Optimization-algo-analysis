import numpy as np

class WeightLayer:
    def __init__(self, init_weight, init_biases, activation_function, activation_derivative, learning_rate, optimizer):
        self.weights = init_weight
        self.biases = init_biases
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.t = 0

    def forward(self, input_data):
        self.input_data = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.output_data = self.activation_function(self.z)
        return self.output_data

    def backward(self, output_gradient):
        dz = output_gradient * self.activation_derivative(self.output_data)
        dw = np.dot(self.input_data.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        self.t += 1
        self.weights, self.biases, self.m_weights, self.v_weights, self.m_biases, self.v_biases = self.optimizer(self.learning_rate, dw, db, self.t,
                                                                                                    self.weights, self.biases, self.m_weights,
                                                                                                    self.v_weights, self.m_biases, self.v_biases)
        return np.dot(dz, self.weights.T)
    