import numpy as np

class DenseLayer:
    # Layer initialization
    def __init__(self, n_inputs, n_outputs):
        # Random weights and many 0s biases
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.01
        self.biases = np.zeros((1, n_outputs)) 

    # forward pass 
    def forward(self, inputs):
        ''' 
        Linear Transformation 
        
        Z = XW + b

        X : inputs (n_samples, n_features)
        W : weights (n_features, n_outputs)
        b : biases (1 , n_outputs)
        '''
        
        self.output = np.dot(inputs, self.weights) + self.biases