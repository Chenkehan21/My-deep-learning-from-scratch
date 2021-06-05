import os, sys
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from BP.multilayer import AffineLyaer, SigmoidLayer, ReLULayer, SoftmaxWithLoss
from Numerical_differentiation.numerical_differentiation import numerical_gradient


class LayerBuilder:
    def __init__(self, input_shape: int, output_shape: int, hidden_size_list: list, batch_size,
                 activation_layer="relu", weight_init_std="relu", weight_decay_lambda=0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size_list = hidden_size_list
        self.hidden_layers_num = len(self.hidden_size_list)
        self.batch_size = batch_size
        self.activation_layer = str(activation_layer).lower()
        self.weight_init_std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        self._init_layers(self.weight_init_std)

        activation_layers = {"sigmoid": SigmoidLayer, "relu":ReLULayer}
        self.layers = {}
        for idx in range(1, self.hidden_layers_num + 1):
            self.layers["affine" + str(idx)] = AffineLyaer(self.params["w" + str(idx)], self.params["b" + str(idx)])
            self.layers["activation" + str(idx)] = activation_layers[self.activation_layer]()

        idx = self.hidden_layers_num + 1
        self.layers["affine" + str(idx)] = AffineLyaer(self.params["w" + str(idx)], self.params["b" + str(idx)])
        self.lastlayer = SoftmaxWithLoss()

    def _init_layers(self, weight_init_std):
        all_layers_list = [self.input_shape] + self.hidden_size_list + [self.output_shape]
        for idx in range(1, len(all_layers_list)):
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_layers_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_layers_list[idx - 1])

            self.params["w" + str(idx)] = np.random.randn(all_layers_list[idx - 1], all_layers_list[idx]) * scale
            self.params["b" + str(idx)] = np.zeros(all_layers_list[idx]) # bias just initialize as all zeros

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layers_num + 2): # consider last affine layer and softmaxwithloss layer
            w = self.params["w" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(w**2)
        loss = self.lastlayer.forward(y, t) + weight_decay
        
        return loss

    def accuracy(self, x, t):
        correct = 0
        for idx in range(0, len(x), self.batch_size):
            y = self.predict(x[idx : self.batch_size + idx])
            y = np.argmax(y, axis=1)
            # print("y: ", y)
            # print("label: ", t[idx : idx + self.batch_size])
            correct += np.sum(y == t[idx : idx + self.batch_size])
        
        return correct / t.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for idx in range(1, self.hidden_layers_num + 2):
            grads["w" + str(idx)] = self.layers["affine" + str(idx)].dw + self.weight_decay_lambda * self.layers["affine" + str(idx)].w
            grads["b" + str(idx)] = self.layers["affine" + str(idx)].db
            
        return grads

    def gradient_numerical(self, x, t):
        loss_func = lambda w: self.loss(x, t)
        grads = {}
        for idx in range(1, self.hidden_layers_num + 2):
            grads["w" + str(idx)] = numerical_gradient(loss_func, self.params["w" + str(idx)])
            grads["b" + str(idx)] = numerical_gradient(loss_func, self.params["b" + str(idx)])
        
        return grads