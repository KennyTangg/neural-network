import numpy as np

class DenseLayer:
    # Layer initialization
    def __init__(self, n_inputs, n_outputs, 
        weight_regularizer_L1=0, weight_regularizer_L2=0,
        bias_regularizer_L1=0, bias_regularizer_L2=0):
        # Random weights and many 0s biases
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.01
        self.biases = np.zeros((1, n_outputs)) 
        # regularization 
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    # forward pass 
    def forward(self, inputs):
        ''' 
        Linear Transformation 
        
        y = XW + b

        X : inputs (n_samples, n_features)
        W : weights (n_features, n_outputs)
        b : biases (1 , n_outputs)
        y : output 
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # gradients on regularization
        # L1 on weights
        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
            
        # L2 on weights
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights
    
        # L1 on biases
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1
        
        # L2 on biases
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases
    
        self.dinputs = np.dot(dvalues, self.weights.T)