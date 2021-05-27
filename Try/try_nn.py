from operator import ne
import sys, os
sys.path.append(os.pardir)
from Activate_functions.activate_functions import Sigmoid, ReLU
from Output_layers.output_layers import identify_function, safe_softmax, softmax
import numpy as np


class Net:
    def __init__(self, input_shape, output_shape, batch_size, hidden_size):
        self.network = {}
        self.network['w1'] = np.random.random((input_shape, hidden_size))
        ''' self.network['b1'] = np.zeros(hidden_size): it will be broadcast 
        to all dimensions.
        Don't use: 
        self.network['b1'] = np.random.random((batch_size, hidden_size)) !
        because it requires all inputs meet batch_size, however the last 
        part of dataset may not meet batch size!!
        '''
        self.network['b1'] = np.zeros(hidden_size)
        self.network['w2'] = np.random.random((hidden_size, hidden_size))
        self.network['b2'] = np.zeros(hidden_size)
        self.network['w3'] = np.random.random((hidden_size, output_shape))
        self.network['b3'] = np.zeros(output_shape)


    def forward(self, x):
        x = np.dot(x, self.network['w1']) + self.network['b1']
        x = ReLU(x)
        x = np.dot(x, self.network['w2']) + self.network['b2']
        x = Sigmoid(x)
        x = np.dot(x, self.network['w3']) + self.network['b3']
        res = safe_softmax(x)
        return res


if __name__ == "__main__":
    input = np.random.random((10, 4))
    input_shape = input.shape[1]
    output_shape = 10
    batch_size = input.shape[0]
    net = Net(input_shape, output_shape, batch_size, 50)
    res = net.forward(input)
    print(res)