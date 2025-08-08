from activations import *
from layer import *
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)
dense1 = DenseLayer(2, 3)
activation1 = ReLU()

dense2 = DenseLayer(3, 3)
activation2 = SoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

print(dense1.output[:5])

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])