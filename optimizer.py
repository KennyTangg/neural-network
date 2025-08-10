import numpy as np

class SGD:
    # Initialize optimizer
    def __init__(self, learning_rate = 1, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):

        # use momentum update
        if self.momentum:
            # if layer doesn't have momentum arrays, create arrays filled with 0s
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            # weight updates with momentum
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # bias updates with momentum
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        # use normal update
        else:
            weight_updates = - self.current_learning_rate * layer.dweights
            bias_updates = - self.current_learning_rate * layer.dbiases

        # update weights and biases using either normal or momentum update
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    # call once after any parameters updates
    def post_update_params(self):
        self.iterations += 1

class AdaGrad:
    def __init__(self, learning_rate = 1.0, decay = 0.0, epsilon = 1e-10):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon 

    # call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):
        # if layer doesn't have cache arrays, create arrays filled with 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / ( np.sqrt(layer.weight_cache) + self.epsilon )
        layer.biases += -self.current_learning_rate * layer.dbiases / ( np.sqrt(layer.bias_cache) + self.epsilon )

    # call once after any parameters updates
    def post_update_params(self):
        self.iterations += 1

class RMSprop:
    def __init__(self, learning_rate = 0.001 , decay = 0.0, epsilon = 1e-8, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon 
        self.rho = rho

    # call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    # update parameters
    def update_params(self, layer):
        # if layer doesn't have cache arrays, create arrays filled with 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2


        # SGD parameter update + normalization with square rooted cache
        layer.weights += - self.current_learning_rate * layer.dweights / ( np.sqrt(layer.weight_cache) + self.epsilon )
        layer.biases += - self.current_learning_rate * layer.dbiases / ( np.sqrt(layer.bias_cache) + self.epsilon )

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Adam:
    def __init__(self, learning_rate = 0.001 , decay = 0.0, epsilon = 1e-8, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon 
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    # call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    
    # update parameters
    def update_params(self, layer):
        # if layer doesn't have cache arrays, create arrays filled with 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
    
        # update momentum with current gradient
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # update cache with squared current gradient
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / ( np.sqrt(weight_cache_corrected) + self.epsilon )
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / ( np.sqrt(bias_cache_corrected) + self.epsilon )

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1