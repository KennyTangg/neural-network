import numpy as np

class Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        # derivative is 1 
        self.dinputs = dvalues.copy()

class Sigmoid:
    '''
    Sigmoid Activation Function

    Formula:
        z(x) = 1 / (1 + exp(-x))

    z(x)           = output of sigmoid for input x
    x              = input to the activation function
    exp            = exponential function (e)
    '''
    def forward(self, inputs):
        self.inputs = inputs
        # between 0 and 1
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

class ReLU:
    '''
    Rectified Linear Unit (ReLU) Activation Function

    Formula:
        ReLU(x) = max(0, x)

    ReLU(x)        = output of ReLU for input x
    x              = input to the activation function

    '''

    def forward(self, inputs):
        self.inputs = inputs
        # between 0 and positive numbers
        self.output = np.maximum(0, inputs)
       
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values are negative 
        self.dinputs[self.inputs <= 0] = 0

class SoftMax:
    '''
    Softmax Activation Function

    Formula:
        softmax(x_i) = exp(x_i) / ∑_j exp(x_j)

    softmax(x_i)   = output probability for class i
    x_i            = input for class i
    ∑_j            = total or sum over all classes j

    '''

    def forward(self, inputs):
        self.inputs = inputs

        # get unnormalized values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # normalize for each sample
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