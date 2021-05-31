import numpy as np
from convertions import img2col, col2img


class CNN:
    def __init__(self, kernel_size: tuple, stride=1, padding=0, bias=False):
        self.kernel = np.random.rand(*kernel_size)
        self.stride = stride
        self.padding = padding
        self.bias = np.random.rand(kernel_size[1]) if bias else np.zeros(kernel_size[1])

        self.x = None
        self.dw = None
        self.db = None
        self.col = None
        self.kernel_col = None

    def forward(self, x):
        N, C, H, W = x.shape
        FN, C, FH, FW = self.kernel.shape
        out_h = (H + 2 * self.padding - FH) // self.stride + 1
        out_w = (W + 2 * self.padding - FW) // self.stride + 1

        col = img2col(x, FH, FW, self.stride, self.padding)
        kernel_col = self.kernel.reshape(FN, -1).T
        res = np.dot(col, kernel_col) + self.bias # res: (N * out_h * out_w, FN)
        res = res.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # res: (N, FN, out_h, out_w)

        self.x = x
        self.kernel_col = kernel_col
        self.col = col

        return res

    def backward(self, dout):
        # dout: (N, FN, out_h, out_w)
        FN, C, FH, FW = self.kernel.shape

        # turn dout to: (N * out_h * out_w, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # CNN is a special FC, so the backward process is similar to FC.
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.kernel_col.T)
        dx = col2img(self.x, dcol, FH, FW, self.stride, self.padding)

        return dx


class Maxpooling:
    def __init__(self, kernel_size: int, stride=1, padding=0):
        self.kernel = np.random.rand(kernel_size, kernel_size) # maxpooling kernel is a square by default
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        FH, FW = self.kernel.shape
        out_h = (H + 2 * self.padding - FH) // self.stride + 1
        out_w = (W + 2 * self.padding - FW) // self.stride + 1

        col = img2col(x, FH, FW, self.stride, self.padding)
        col = col.reshape(-1, FH * FW)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max
        
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        FH, FW = self.kernel.shape
        
        pool_size = FH * FW
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2img(self.x, dcol, FH, FW, self.stride, self.padding)
        
        return dx