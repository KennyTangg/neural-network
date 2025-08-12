from layer import *
from loss import *
from activations import *

class Model:
    def __init__(self):
        # create list of objects
        self.layers = []
        # softmax classifier
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
    
    # set loss, optimizer, and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy 

    def finalize(self):
        # create input layer object
        self.input_layer = InputLayer()

        # count all the objects
        layer_count = len(self.layers)

        # Initialize a list contain trainable layers
        self.trainable_layers = []
        
        # iterate the objects 
        for i in range(layer_count):
            # if it is first layer, make the previous layer object the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            # All layers except for the first and the last
            elif i < layer_count - 1: 
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # last layers and make the next object 
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
        
            # if layer has attribute called "weights" means that it is trainable layers, so add it to list of trainable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
            
            # update loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)

        # if output activation is Softmax and loss function is Categorical Cross-Entropy, create an object of combined activation 
        if isinstance(self.layers[-1], SoftMax) and isinstance(self.loss, CategoricalCrossEntropyLoss):
            # create object of combined activation and loss fucntions
            self.softmax_classifier_output = CategoricalCrossEntropyLoss_SoftMax()
    
    # Train the model
    def train(self, X, y, *, epochs = 1, print_every = 1, validation_data = None):
        # initialize accuracy object 
        self.accuracy.init(y)

        # training loop
        for epoch in range(1, epochs + 1):
            # perform forward pass
            output = self.forward(X, training = True)

            # calculate loss 
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization = True)
            loss = data_loss + regularization_loss
            
            # get predictions and calculate
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # perform backward pass
            self.backward(output, y)

            # optimize ( update parameters )
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' + 
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
            
        if validation_data is not None:
            X_val, y_val = validation_data

            # perform forward pass
            output = self.forward(X_val, training = False)

            # calculate the loss
            loss = self.loss.calculate(output, y_val)

            # get predictions and calculate an accuracy 
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # print summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')
    
    def forward(self, X, training):
        # call forward method on input layer
        self.input_layer.forward(X, training)

        # call forward method for every object and pass output of the previous object as a parameter 
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        # if softmax classifier
        if self.softmax_classifier_output is not None:
            # call backward method on combined activation / loss
            self.softmax_classifier_output.backward(output, y)

            # not call the backward method last layer
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # call backward method going through all objects but last in reversed order 
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return

        # first call backward method on the loss
        self.loss.backward(output, y)

        # call backward method going through all objects in reversed order
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)