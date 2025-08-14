# Neural Network from Scratch

This project implements a fully connected neural network from scratch using **NumPy**, without using on deep learning frameworks like TensorFlow or PyTorch for the computations.

## Features
- Dense layers, activation functions, loss functions, and optimizers.
- Architecture for building and training models.
- Support for batch training, learning rate decay, and accuracy tracking.
- Visualization utilities for dataset samples and predictions.

## Datasets
This project uses two datasets:

**Fashion MNIST (PNG format)**  
   [https://www.kaggle.com/datasets/andhikawb/fashion-mnist-png/](https://www.kaggle.com/datasets/andhikawb/fashion-mnist-png/)

**Playing Cards Image Dataset**  
   [https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

Make sure to download and place the datasets in this directory name it `fashion_data` and `card_data`

## Installation
```bash
git clone https://github.com/kennytangg/neural-network.git

cd neural-network

python3 -m venv venv # create virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt 
# or just install numpy because that is the only library use to create neural network, the rest are in notebook
pip install numpy
