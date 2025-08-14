import numpy as np

class Accuracy:
    # calculate an accuracy
    def calculate(self, predictions, y):
        # get comparison results
        comparisons = self.compare(predictions, y)

        # calculate an accuracy
        accuracy = np.mean(comparisons)

        # add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons) 

        return accuracy

    def calculate_accumulated(self):
        # calculated accumulated accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy
    
    def new_pass(self):
        # reset variables for accumulated accuracy
        self.accumulated_sum = 0
        self.accumulated_count = 0


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