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
    def train(self, X, y, *, epochs = 1,batch_size = None,  print_every = 1, validation_data = None):
        # initialize accuracy object 
        self.accuracy.init(y)

        # default value if batch size not being set
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X) // batch_size

            # Add 1 to include this is not full batch 
            if train_steps * batch_size < len(X):
                train_steps += 1 
            
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size

                # Add 1 to include this is not full batch 
                if validation_steps * batch_size <  len(X_val):
                    validation_steps += 1

        # training loop
        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')

            # reset accumulated values in loss and accuracy objects 
            self.loss.new_pass()
            self.accuracy.new_pass()

            # iterate over steps
            for step in range(train_steps):

                # if batch size is not set
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # slice batch
                else:
                    batch_X = X[step * batch_size: (step + 1) * batch_size]
                    batch_y = y[step * batch_size: (step + 1) * batch_size]
            
                # forward
                output = self.forward(batch_X, training = True)

                # calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization = True)
                
                loss = data_loss + regularization_loss

                # predictions and accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # backward
                self.backward(output, batch_y)
                
                # optimize or update parameters
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                # Print summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization = True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            print(f'\ntraining : ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')
            
            if validation_data is not None:
                # reset accumulated values in loss
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(validation_steps):
                    # if batch not set
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    
                    # slice a batch
                    else:
                        batch_X = X_val[step * batch_size: (step + 1) * batch_size]
                        batch_y = y_val[step * batch_size: (step + 1) * batch_size]
                
                    output = self.forward(batch_X, training = False)

                    # calculate loss
                    self.loss.calculate(output, batch_y)

                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)

                # get validation loss and accuracy 
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()

                # Print summary
                print(f'validation : ' +
                    f'acc: {validation_accuracy:.3f}, ' +
                    f'loss: {validation_loss:.3f}\n')
    
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
            
    def get_predictions(model, X):
        """
        Perform forward pass through the model to get predicted labels.
        Returns an array of predicted class indices.
        """
        output = model.forward(X, training=False)
        predictions = model.output_layer_activation.predictions(output)
        return predictions