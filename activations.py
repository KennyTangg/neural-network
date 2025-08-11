import numpy as np

class ReLU:
    def forward(self, inputs):
       self.inputs = inputs
       self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values are negative 
        self.dinputs[self.inputs <= 0] = 0

class SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # create dinputs with the same shape as dvalues
        self.dinputs = np.empty_like(dvalues)

        # calculate gradient for each sample
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            # calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # apply chain rule , multiply jacobian matrix by gradient from the next layer
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            