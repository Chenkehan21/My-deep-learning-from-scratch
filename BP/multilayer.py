import numpy as np
from Loss_functions.loss_functions import cross_entropy_loss
from Output_layers.output_layers import safe_softmax
from Activate_functions.activate_functions import ReLU


class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1.0 - self.out) 


class AffineLyaer:
    def __init__(self, w, b):
        self.x = None
        self.w = w
        self.b = b
        self.dx = None
        self.dw = None
        self.db = None
    
    '''
    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0) # pay attention to db here, should sum along axis 0
        return self.dx
    '''

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = safe_softmax(x)
        self.t = t
        return cross_entropy_loss(self.y, t)

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        if self.y.ndim != self.t.ndim:
            one_hot_label = np.zeros((self.t.size, 10))
            one_hot_label[np.arange(self.t.size), self.t] = 1
            return (self.y - one_hot_label) / batch_size # remember to divide batch size
        else:
            return (self.y - self.t) / batch_size

    def backward2(self, dout=1):
        batch_size = self.t.shape[0]
        if self.y.size == self.t.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size
        return dx