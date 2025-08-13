import numpy as np

class Accuracy:
    # calculate an accuracy
    def calculate(self, predictions, y):
        # get comparison results
        comparisons = self.compare(predictions, y)

        # calculate an accuracy
        accuracy = np.mean(comparisons)

        return accuracy

class Categorical(Accuracy):
    
    def init(self, y, reinit=False):
        pass

    # compare predictions 
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Regression(Accuracy):
    def __init__(self):
        # create precision property
        self.precision = None
    
    # calculates precision value
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # compare predictions 
    def compare(self, predictions, y):
        return np.abs(predictions - y) < self.precision