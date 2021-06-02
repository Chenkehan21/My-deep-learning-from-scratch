import os, sys
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from BP.multilayer import AffineLyaer, SigmoidLayer, ReLULayer, SoftmaxWithLoss


class LayerBuilder:
    def __init__(self, input_shape: int, output_shape: int, hidden_size_list: list, batch_size,
                 activation_layer="relu", weight_init_std="relu", weight_decay_lambda=0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size_list = hidden_size_list
        self.hidden_layers_num = len(self.hidden_size_list)
        self.batch_size = batch_size
        self.activation_layer = activation_layer
        self.weigh_init_std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        self._init_layers()

        activation_layers = {"sigmoid": SigmoidLayer, "relu":ReLULayer}
        self.layers = {}
        for idx in range(1, self.hidden_layers_num):
            self.layers["affine" + str(idx)] = AffineLyaer(self.params["w" + str(idx)], self.params["b" + str(idx)])
            self.layers["activation" + str(idx)] = activation_layers[activation_layer]()

        idx = self.hidden_layers_num + 1
        self.layers["affine" + str(idx)] = AffineLyaer(self.params["w" + str(idx)], self.params["b" + str(idx)])
        self.lastlayer = SoftmaxWithLoss()

    def _init_layers(self, weight_init_std):
        all_layers_list = [self.input_shape] + len(self.hidden_size) + [self.output_shape]
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
            weight_decay += 0.5 * self.weight_decay_lambda * w**2
        
        loss = self.lastlayer.forward(y, t) + weight_decay
        return loss

    def accuracy(self, x, t):
        correct = 0
        for idx in range(0, len(x), self.batch_size):
            y = self.predict(x[idx : self.batch_size + idx])
            y = np.argmax(y, axis=1)
            if y == t[idx : self.batch_size + idx]:
                correct += 1
        return correct / t.shape[0]

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = self.layers.values()
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        for idx in range(1, self.hidden_layers_num + 2):
            self.params["w" + str(idx)] = self.layers["w" + str(idx)].dw + self.weight_decay_lambda * self.params["w" + str(idx)].w
            self.params["b" + str(idx)] = self.params["b" + str(idx)].db
            
        return grads