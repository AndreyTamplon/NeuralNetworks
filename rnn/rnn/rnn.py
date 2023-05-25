from math import sqrt
import numpy as np


class RNN:
    def __init__(self, steps, neurons):
        '''
        steps: number of time steps
        neurons: number of neurons in the hidden layer
        U - input weights
        W - hidden weights
        b - bias
        V - output weights
        '''
        self.input = None
        self.output = None
        self.state = None
        self.lr = None
        np.random.seed(0)
        self.U = np.random.uniform(-1, 1, (steps, neurons)) / sqrt(neurons)
        self.W = np.random.uniform(-1, 1, (neurons, neurons)) / sqrt(neurons)
        self.n_b = np.random.uniform(-1, 1, (1, neurons)) / sqrt(neurons)
        self.V = np.random.uniform(-1, 1, (neurons, 1)) / sqrt(neurons)
        self.o_b = np.random.uniform(-1, 1, (1, 1)) / sqrt(neurons)

    def forward(self, input):
        """
        x: input, shape: (window_size, number_of_features)
        state: hidden state, shape: (window_size, number_of_neurons)
        o: output, shape: (window_size, 1)
        """
        state = np.zeros((input.shape[0], self.W.shape[0]))
        output = np.zeros((input.shape[0], 1))
        for i in range(input.shape[0]):
            state[i] = sigmoid(input[i] @ self.U + state[max(i - 1, 0)] @ self.W + self.n_b)
            output[i] = state[i] @ self.V + self.o_b
        self.input = input
        self.state = state
        self.output = output

    def backward(self, y):
        loss_grad = self.loss_grad(self.output, y)
        V_grad = np.zeros(self.V.shape)
        o_b_grad = np.zeros(self.o_b.shape)
        W_grad = np.zeros(self.W.shape)
        n_b_grad = np.zeros(self.n_b.shape)
        U_grad = np.zeros(self.U.shape)
        state_grad = np.zeros(self.state.shape)
        next_state_grad = np.zeros(self.state.shape)

        for i in range(self.input.shape[0] - 1, -1, -1):
            V_grad += np.reshape(self.state[i], (self.state[i].shape[0], 1)) @ np.reshape(loss_grad[i], (1, 1))
            o_b_grad += loss_grad[i]
            state_grad = loss_grad[i] @ self.V.T
            if i != self.input.shape[0] - 1:
                state_grad += next_state_grad @ self.W.T
            state_grad = np.multiply(state_grad, self.state[i] * (1 - self.state[i]))
            if i != 0:
                W_grad += self.state[i - 1] @ state_grad
                n_b_grad += state_grad
            next_state_grad = state_grad
            U_grad += np.reshape(self.input[i], (self.input[i].shape[0], 1)) @ np.reshape(state_grad,
                                                                                          (1, state_grad.shape[0]))

        self.V -= self.lr * V_grad
        self.o_b -= self.lr * o_b_grad
        self.W -= self.lr * W_grad
        self.n_b -= self.lr * n_b_grad
        self.U -= self.lr * U_grad

    def loss(self, y_pred, y_test):
        """
        mse
        """
        return np.mean((y_pred - y_test) ** 2)

    def loss_grad(self, y_pred, y_test):
        return 2 * (y_pred - y_test) / y_pred.shape[0]

    def train(self, x_train, y_train, x_test, y_test, epochs, lr, sequence_length):
        self.lr = lr
        losses = {}
        for epoch in range(epochs):
            train_loss = 0
            for i in range(0, x_train.shape[0] - sequence_length, sequence_length):
                self.forward(x_train[i:i + sequence_length])
                self.backward(y_train[i:i + sequence_length])
                train_loss += self.loss(self.output, y_train[i:i + sequence_length])

            if epoch % 1 == 0:
                test_loss = 0
                for i in range(0, x_test.shape[0] - sequence_length, sequence_length):
                    self.forward(x_test[i:i + sequence_length])
                    test_loss += self.loss(self.output, y_test[i:i + sequence_length])
                train_loss /= (x_train.shape[0] // sequence_length)
                test_loss /= (x_test.shape[0] // sequence_length)
                losses[epoch] = [train_loss, test_loss]
                print('Epoch: {}, Train Loss {}, Test Loss: {}'.format(epoch, train_loss, test_loss))

        return losses

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))