import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
import matplotlib.pyplot as plt
from MNIST_Dataset.load_data import load_mnist
from try_nn import Net
from Activate_functions.activate_functions import Sigmoid, ReLU
from Output_layers.output_layers import safe_softmax
from Loss_functions.loss_functions import cross_entropy_loss
from Numerical_differentiation.numerical_differentiation import gradient_batch


BATCH_SIZE = 128
LEARNING_RATE = 0.01
EPOCH = 10000

(train_data, train_labels), (test_data, test_labels) =\
     load_mnist(normalize=True, flatten=True, one_hot_label=False)

CHECK_PER_ITER = max(1, train_data.shape[0] / BATCH_SIZE)


class Network(Net):
    def __init__(self, input_shape, output_shape, batch_size, hidden_size):
        super(Network, self).__init__(input_shape, output_shape, batch_size, hidden_size)
        self.grads = {}

    def predict(self, x):
        x = np.dot(x, self.network['w1']) + self.network['b1']
        x = ReLU(x)
        x = np.dot(x, self.network['w2']) + self.network['b2']
        x = Sigmoid(x)
        x = np.dot(x, self.network['w3']) + self.network['b3']
        res = safe_softmax(x)
        return res

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_loss(y, t)

    def numerical_gradient(self, x, t):
        loss_func = lambda w: self.loss(x, t)
        for key, val in self.network.items():
            self.grads[key] = gradient_batch(loss_func, val)
    
    def accuracy(self, x, t):
        correct = 0
        for i in range(0, len(x), BATCH_SIZE):
            res = self.predict(x[i : BATCH_SIZE + i])
            res = np.argmax(res, axis=1)
            correct += np.sum(res == t[i : BATCH_SIZE + i])
        return float(correct) / t.shape[0]


def train(net):
    print("start training")
    best_test_acc = -10.0
    loss_list = []
    train_acc_list, test_acc_list = [], []
    for epoch in range(EPOCH):
        batch_mask = np.random.choice(train_data.shape[0], BATCH_SIZE)
        train_data_batch = train_data[batch_mask] # batch_mask is an numpy array and can be used as mask
        train_labels_batch = train_labels[batch_mask]
        net.numerical_gradient(train_data_batch, train_labels_batch)
        for key in net.network:
            net.network[key] -= LEARNING_RATE * net.grads[key]
        loss = net.loss(train_data_batch, train_labels_batch)
        loss_list.append(loss)
        print("epoch: %d|  loss: %.6f" % (epoch, loss))

        if epoch % CHECK_PER_ITER == 0:
            train_acc = net.accuracy(train_data, train_labels)
            test_acc = net.accuracy(test_data, test_labels)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("epoch: %d|  train accuracy: %.6f%%" % (epoch, train_acc * 100))
            print("epoch: %d|  test accuracy: %.6f%%" % (epoch, test_acc * 100))
            if test_acc > best_test_acc:
                print("update network %.6f%% -> %.6f%%" % (best_test_acc * 100, test_acc * 100))
                with open("./try_learning_networks/network_%.3f.pickle" % (test_acc), "wb") as f:
                    pickle.dump(f)
    
            plt.figure()
            x = np.arange(len(train_acc_list))
            plt.plot(x, train_acc_list, label='train acc', color='b')
            plt.plot(x, test_acc_list, label='test acc', color='r')
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.legend()
            plt.savefig("./try_learning_figures/try_learning_acc_%d.png" % epoch)

            plt.close()
            plt.figure()
            x = np.arange(len(loss_list))
            plt.plot(x, loss_list, label='train loss')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig("./try_learning_figures/try_learning_loss_%d.png" % epoch)
            print("save figures done!")

    print("training done!")


def main():
    input_shape = 784
    output_shape = 10
    hidden_size = 50
    net = Network(input_shape, output_shape, BATCH_SIZE, hidden_size)
    train(net)


if __name__ == "__main__":
    main()