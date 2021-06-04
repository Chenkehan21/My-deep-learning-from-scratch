import numpy as np


'''It's better to let params and grads be paramemters of step method, because in the Trainer class we want to design later
things like learning rate or momentum can be decided beforehand however the network's params and loss function's grads can't.
'''


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    '''update network's paramemters in optimizer's step method
    '''
    def step(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = val
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def step(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = val
        
        for key in params.keys():
            self.h[key] += grads[key]**2
            params[key] -= self.lr * grads[key] / (1e-7 + np.sqrt(self.h[key]))  


class Adam:
    def __init__(self):
        pass


class RMSProp:
    def __init__(self):
        pass