# Layer Class
# Ziping Chen
# March 2020
import numpy as np

class Layer:
    def __init__(self, input_size, output_size, name='layer', activate='relu', regularizer=None):
        # initialize weight and bias with N(0, 0.01)
        self.weight = np.random.normal(0, 0.1, (input_size, output_size))
        self.bias = np.random.normal(0, 0.1, (output_size, ))
        self.name = name
        self.activate = activate
        if regularizer is None:
            self.regularizer = regularizer
        else:
            self.regularizer = regularizer[0]
            self.l = regularizer[1]

    def set_weight(self, w, b):
        self.weight = w[:]
        self.bias = b[:]
        
    def get_weight(self):
        return self.weight, self.bias

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return np.where(x > 0.0, x, 0.0)

    def softmax(self, x):
        # stable softmax by subtracting the maximum value
        x = x - np.max(x, axis=1).reshape(x.shape[0], 1)
        return (np.exp(x) / np.sum(np.exp(x), axis=1).reshape(x.shape[0], 1))

    def forward(self, x):
        self.x = x[:]
        self.s = self.x @ self.weight + self.bias
        if self.activate == 'tanh':
            return self.tanh(self.s)
        elif self.activate == 'relu':
            return self.ReLU(self.s)
        elif self.activate == 'softmax':
            return self.softmax(self.s)

    def backward(self, delta):
        self.delta = delta[:]
        if self.activate == 'tanh':
            self.delta *= (1 - np.square(self.tanh(self.s)))
        elif self.activate == 'relu':
            self.delta *= np.where(self.s >= 0.0, 1.0, 0.0)
        elif self.activate == 'softmax':
            pass
        d = self.delta @ self.weight.T
        return d

    def update(self, eta):
        update_w = (self.x.T @ self.delta) / self.x.shape[0]
        update_b = np.sum(self.delta, axis=0) / self.x.shape[0]
        if self.regularizer == 'l1':
            # L1 regularization
            update_w += self.l * np.where(self.weight >= 0, 1, -1)
            update_b += self.l * np.where(self.bias >= 0, 1, -1)
        elif self.regularizer == 'l2':
            # L2 regularization
            update_w += 2 * self.l * self.weight
            update_b += 2 * self.l * self.bias
        self.weight -= (eta * update_w)
        self.bias -= (eta * update_b)
        assert not np.isnan(np.min(self.weight)) and not np.isnan(np.min(self.bias)), "NaN error during weights update!!!"
