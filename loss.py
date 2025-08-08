import numpy as np

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