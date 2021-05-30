import numpy as np


class SGD:
    def __init__(self, params, grads, lr):
        self.params = params
        self.grads = grads
        self.lr = lr
    
    '''update network's paramemters in optimizer's step method
    '''
    def step(self):
        for key in self.params:
            self.params[key] -= self.lr * self.grads[key]


class Momentum:
    def __init__(self, params, grads, lr, momentum):
        self.params = params
        self.grads = grads
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self):
        if self.v == None:
            self.v = {}
            for key, val in self.params.items():
                self.v[key] = val
        
        for key in self.params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * self.grads[key]
            self.params[key] += self.v[key]


class AdaGrad:
    def __init__(self, params, grads, lr):
        self.params = params
        self.grads = grads
        self.lr = lr
        self.h = None

    def step(self):
        if self.h == None:
            self.h = {}
            for key, val in self.params.items():
                self.h[key] = val
        
        for key in self.params.keys():
            self.h[key] += self.grads[key]**2
            self.params[key] -= self.lr * self.grads[key] / (1e-7 + np.sqrt(self.h[key]))  


class Adam:
    def __init__(self):
        pass