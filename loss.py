import numpy as np
from activations import SoftMax

class Loss:
    def calculate(self, y_hat, y):
        '''
        calculate the average loss from all sample in the batch 
        '''
        sample_losses = self.forward(y_hat, y)

        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossEntropyLoss(Loss):
    '''
    Categorical Cross-Entropy Loss

    Loss = - ∑ (y_i * log(y_hat_i))

    y : target output, can be 
        - 1D array of class indices (n_samples,)
        - or 2D one-hot encoded array (n_samples, n_classes)

    y_hat : predicted probabilities (n_samples, n_classes)
        - values between 0 and 1
    '''

    def forward(self, y_hat, y):
        # number of samples in current batch       
        samples = len(y_hat)

        # clip data to prevent log(0)
        y_hat_clipped = np.clip(y_hat, 1e-12, 1 - 1e-12)

        # y contains class indices (1D Array)
        if len(y.shape) == 1:
            target_probs = y_hat_clipped[range(samples), y]
        
        # y is one-hot-encoded (2D Array)
        elif len(y.shape) == 2:
            target_probs = np.sum( y_hat_clipped * y , axis=1 )
        
        # Compute per-sample using cross-entropy losses formula
        sample_losses = -np.log(target_probs)
        return sample_losses 

    def backward(self, dy_hat, y):
        ''' 
        Gradient of loss

        (- y / y_hat) * (1 / N)

        y : one-hot encoded
        y_hat : predicted probabilities

        '''

        # number of samples in current batch
        samples = len(dy_hat)

        labels = len(dy_hat[0])
        # If labels are discrete values, turn to one-hot encoded
        if len(y.shape) == 1:
            # create identity matrix of size y
            y = np.eye(labels)[y]

        # calculate gradient
        self.dinputs = - y / dy_hat

        # normalize (calculate the average)
        self.dinputs = self.dinputs / samples

# Softmax activation and cross-entropy loss combined for faster backpropagation
class CategoricalCrossEntropyLoss_SoftMax(Loss):

    def __init__(self):
        # create activation and loss function objects
        self.activation = SoftMax() 
        self.loss = CategoricalCrossEntropyLoss()

    def forward(self, inputs, y):
        # output layer activation function 
        self.activation.forward(inputs) 
        self.output = self.activation.output

        # calculate average and return loss value
        return self.loss.calculate(self.output, y)

    def backward(self, dy_hat, y):
        ''' 
        Gradient of loss 

        (y_hat - y) / N
         
        y : discrete class indices
        y_hat : predicted probabilities 
        '''

        # number of samples in current batch
        samples = len(dy_hat)

        # if labels are one-hot encoded, turn to discrete values
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self.dinputs = dy_hat.copy()

        # calculate gradient
        self.dinputs[range(samples), y] -= 1

        # normalize (calculate the average)
        self.dinputs = self.dinputs / samples