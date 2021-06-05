import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from Optimizers.optimizers import *


class Trainer:
    def __init__(self, train_data, train_labels, test_data, test_labels,
                 network, batch_size=128, epochs=50, optimizer="SGD", optimizer_params={"lr": 1e-3}):
        t = time.time() % 100
        self.figure_path = "./figures_%.4f/" % t
        self.network_path = "./network_files_%.4f/" % t
        os.makedirs(self.figure_path, exist_ok=True)
        os.makedirs(self.network_path, exist_ok=True)
        print("t: %.4f" % t)
        
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.network = network
        self.batch_size = batch_size
        self.epochs = epochs
        self.check_per_epoch = max(1, self.train_data.shape[0] / self.batch_size)

        # optimizer
        optimizer_dict = {"sgd": SGD, "momentum": Momentum, "adagrad": AdaGrad, "adam": Adam, "rmsprop": RMSProp}
        self.optimizer = optimizer_dict[str(optimizer).lower()](**optimizer_params)

        self.loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        print("start training")
        best_test_acc = -10.0
        max_iter = int(self.epochs * self.check_per_epoch)
        epoch = 0
        for i in range(max_iter):
            batch_mask = np.random.choice(self.train_data.shape[0], self.batch_size)
            train_data_batch = self.train_data[batch_mask] # batch_mask is an numpy array and can be used as mask
            train_labels_batch = self.train_labels[batch_mask]
            grads = self.network.gradient(train_data_batch, train_labels_batch)

            # for key in self.net.network.params:
            #     self.network.params[key] -= self.optimizer_params["lr"] * grads[key]

            self.optimizer.step(self.network.params, grads)

            loss = self.network.loss(train_data_batch, train_labels_batch)
            self.loss_list.append(loss)

            if i % self.check_per_epoch == 0:
                print("iteration: %d|  loss: %.6f" % (i, loss))
                epoch += 1
                train_acc = self.network.accuracy(self.train_data, self.train_labels)
                test_acc = self.network.accuracy(self.test_data, self.test_labels)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                print("epoch: %d|  train accuracy: %.6f%%" % (epoch, train_acc * 100))
                print("epoch: %d|  test accuracy: %.6f%%" % (epoch, test_acc * 100))
                if test_acc > best_test_acc:
                    print("update network %.6f%% -> %.6f%%" % (best_test_acc * 100, test_acc * 100))
                    best_test_acc = test_acc
                    with open(self.network_path + "network_%.3f.pickle" % (test_acc), "wb") as f:
                        pickle.dump([self.network.params, grads], f)
        
                plt.figure()
                x = np.arange(len(self.train_acc_list))
                plt.plot(x, self.train_acc_list, label='train acc', color='b')
                plt.plot(x, self.test_acc_list, label='test acc', color='r')
                plt.xlabel("epoch")
                plt.ylabel("accuracy")
                plt.legend()
                plt.savefig(self.figure_path + "/try_learning_bp_acc_%d.png" % epoch)

                plt.close()
                plt.figure()
                x = np.arange(len(self.loss_list))
                plt.plot(x, self.loss_list, label='train loss')
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend()
                plt.savefig(self.figure_path + "try_learning_bp_loss_%d.png" % epoch)
                print("save figures done!")

        print("training done!")

    def gradient_check(self):
        data = self.train_data[:3]
        label = self.train_labels[:3]
        # print(data.shape)
        gradient_numerical = self.network.gradient_numerical(data, label)
        gradient_bp = self.network.gradient(data, label)
        for key in self.network.params.keys():
            diff = np.average(np.abs(gradient_numerical[key] - gradient_bp[key]))
            print(str(key) + " difference: ",  diff)