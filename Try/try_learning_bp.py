import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from MNIST_Dataset.load_data import load_mnist
from try_nn import Net
from try_learning import Network
from Activate_functions.activate_functions import Sigmoid, ReLU
from Output_layers.output_layers import safe_softmax
from Loss_functions.loss_functions import cross_entropy_loss
from Numerical_differentiation.numerical_differentiation import gradient_batch
from BP.multilayer import MultiLayer, AddLayer, AffineLyaer, ReLULayer, SigmoidLayer, SoftmaxWithLoss


BATCH_SIZE = 128
LEARNING_RATE = 0.15
TOTAL_ITERATION_TIMES = 10000

(train_data, train_labels), (test_data, test_labels) =\
     load_mnist(normalize=True, flatten=True, one_hot_label=False)

CHECK_PER_EPOCH = max(1, train_data.shape[0] / BATCH_SIZE)


class Network_BP(Network):
    def __init__(self, input_shape, output_shape, hidden_size, weight_init_std):
        super(Network_BP, self).__init__(input_shape, output_shape, hidden_size, weight_init_std)
        self.layers = OrderedDict()
        self.layers["Affine1"] = AffineLyaer(self.network["w1"], self.network["b1"])
        self.layers["ReLU1"] = ReLULayer()
        self.layers["Affine2"] = AffineLyaer(self.network["w2"], self.network["b2"])
        self.layers["ReLU2"] = ReLULayer()
        self.layers["Affine3"] = AffineLyaer(self.network["w3"], self.network["b3"])
        
        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)

    def gradient_bp(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        layers = list(self.layers.values())
        layers.reverse()
        dout = 1
        dout = self.lastlayer.backward(dout)
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["w1"] = self.layers["Affine1"].dw
        grads["b1"] = self.layers["Affine1"].db
        grads["w2"] = self.layers["Affine2"].dw
        grads["b2"] = self.layers["Affine2"].db
        grads["w3"] = self.layers["Affine3"].dw
        grads["b3"] = self.layers["Affine3"].db
        return grads


def train(net):
    print("start training")
    best_test_acc = -10.0
    epoch = 0
    loss_list = []
    train_acc_list, test_acc_list = [], []
    for i in range(TOTAL_ITERATION_TIMES):
        batch_mask = np.random.choice(train_data.shape[0], BATCH_SIZE)
        train_data_batch = train_data[batch_mask] # batch_mask is an numpy array and can be used as mask
        train_labels_batch = train_labels[batch_mask]
        grads = net.gradient_bp(train_data_batch, train_labels_batch)
        for key in net.network:
            net.network[key] -= LEARNING_RATE * grads[key]
        loss = net.loss(train_data_batch, train_labels_batch)
        loss_list.append(loss)
        print("iteration: %d|  loss: %.6f" % (i, loss))

        if i % CHECK_PER_EPOCH == 0:
            epoch += 1
            train_acc = net.accuracy(train_data, train_labels)
            test_acc = net.accuracy(test_data, test_labels)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("epoch: %d|  train accuracy: %.6f%%" % (epoch, train_acc * 100))
            print("epoch: %d|  test accuracy: %.6f%%" % (epoch, test_acc * 100))
            if test_acc > best_test_acc:
                print("update network %.6f%% -> %.6f%%" % (best_test_acc * 100, test_acc * 100))
                with open("./try_learning_bp_networks/network_%.3f.pickle" % (test_acc), "wb") as f:
                    pickle.dump([net.network, net.grads], f)
    
            plt.figure()
            x = np.arange(len(train_acc_list))
            plt.plot(x, train_acc_list, label='train acc', color='b')
            plt.plot(x, test_acc_list, label='test acc', color='r')
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.legend()
            plt.savefig("./try_learning_bp_figures/try_learning_bp_acc_%d.png" % epoch)

            plt.close()
            plt.figure()
            x = np.arange(len(loss_list))
            plt.plot(x, loss_list, label='train loss')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig("./try_learning_bp_figures/try_learning_bp_loss_%d.png" % epoch)
            print("save figures done!")

    print("training done!")


def gradient_check(net):
    data = train_data[:3]
    label = train_labels[:3]
    gradient_numerical = net.numerical_gradient(data, label)
    gradient_bp = net.gradient_bp(data, label)
    for key in net.network:
        diff = np.average(np.abs(gradient_numerical[key] - gradient_bp[key]))
        print(str(key) + " difference: ",  diff)


def main(to_train=False, check_gradient=False):
    input_shape = 784
    output_shape = 10
    hidden_size = 50
    weight_init_std = 0.01
    net = Network_BP(input_shape, output_shape, hidden_size, weight_init_std)
    if check_gradient:
        gradient_check(net)
    if to_train:
        train(net)


if __name__ == "__main__":
    main(to_train=False, check_gradient=True)