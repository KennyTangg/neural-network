import numpy as np

class SGD:
    '''
    Stochastic Gradient Descent (SGD)

    Velocity update (momentum term):
        v_t = momentum * v_{t-1} + learning_rate * gradient

    Parameter update:
        W_t = W_{t-1} - v_t

    W_t       = updated parameter
    W_{t-1}   = previous parameter
    v_t       = velocity (weighted average of gradients)
    momentum  = momentum hyperparameter (usually between 0 and 1)
    learning_rate = step size for updates
    gradient  = derivative of loss w.r.t parameters
    '''

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

        # use momentum SGD update
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
            
        # use normal SGD update
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # update weights and biases 
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    # call once after any parameters updates
    def post_update_params(self):
        self.iterations += 1

class AdaGrad:
    '''
    Adaptive Gradient Algorithm (AdaGrad)

    Cache Update (sum of squared gradients): 
        G_t = G_{t-1} + (gradient)^2
    
    Parameter update :
        W_t = W_{t-1} - (learning_rate / (sqrt(G_t) + epsilon)) * gradient

    W_t           = updated parameter
    W_{t-1}       = previous parameter
    G_t           = sum of squares of previous gradients
    learning_rate = step size for updates
    epsilon       = small constant to avoid division by zero
    gradient      = derivative of loss w.r.t parameters
    decay         = optional learning rate decay 
    '''
    def __init__(self, learning_rate = 1.0, decay = 0.0, epsilon = 1e-10):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon 

    # optional explicit decay for the learning rate
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
    '''
    Root Mean Square Propagation (RMSProp)

    Cache Update (exponentially weighted squared gradients):
        G_t = rho * G_{t-1} + (1 - rho) * (gradient)^2

    Parameter update:
        W_t = W_{t-1} - (learning_rate / (sqrt(G_t) + epsilon)) * gradient

    W_t           = updated parameter
    W_{t-1}       = previous parameter
    G_t           = exponentially weighted moving average of squared gradients
    rho           = decay rate for moving average (e.g., 0.9)
    learning_rate = step size for updates
    epsilon       = small constant to avoid division by zero
    gradient      = derivative of loss w.r.t parameters
    decay         = optional learning rate decay 
    '''
    def __init__(self, learning_rate = 0.001 , decay = 0.0, epsilon = 1e-8, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon 
        self.rho = rho

    # optional explicit decay for the learning rate
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
    '''
    Adaptive Moment Estimation (Adam)

    Momentum update (first moment estimate):
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
    Bias-corrected first moment:
        m̂_t = m_t / (1 - β₁^t)

    Cache update (second moment estimate):
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
    Bias-corrected second moment:
        v̂_t = v_t / (1 - β₂^t)

    Parameter update:
        W_t = W_{t-1} - (learning_rate * m̂_t) / (sqrt(v̂_t) + epsilon)

    W_t           = updated parameter
    W_{t-1}       = previous parameter
    g_t           = current gradient
    m_t           = first moment (running average of gradients)
    v_t           = second moment (running average of squared gradients)
    β₁, β₂        = decay rates for moments (default β₁=0.9, β₂=0.999)
    m̂_t, v̂_t     = bias-corrected moments
    learning_rate = step size for updates
    epsilon       = small constant to avoid division by zero
    decay         = optional learning rate decay factor
    '''

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