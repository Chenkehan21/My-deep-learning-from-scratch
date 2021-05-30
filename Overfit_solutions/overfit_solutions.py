import numpy as np


class Dropout:
    def __init__(self, x, dropout_ratio, to_train=True):
        self.x = x
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.to_train = to_train
    
    def forward(self):
        self.mask = np.random.rand(*self.x.shape) > self.dropout_ratio
        if self.to_train:
            return self.x * self.dropout_ratio
        else:
            # drop trains sub-networks, however we should use the entirty network when testing.
            return self.x
    
    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    def __init__(self):
        pass