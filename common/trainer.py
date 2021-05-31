import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
import matplotlib.pyplot as plt


os.mkdir('./figures/', exist_ok=True)
os.mkdir('./network_files/', exist_ok=True)


class Trainer:
    def __init__(self, train_data, train_labels, test_data, test_labels,
                 network, batch_size=128, epochs=50, optimizer="SGD", optimizer_params={"lr": 1e-3}):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.network = network
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimzer = optimizer
        self.optimizer_params = optimizer_params
        self.check_per_epoch = max(1, self.train_data.shape[0] / self.batch_size)
        self.loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        print("start training")
        best_test_acc = -10.0
        max_iter = self.epochs * self.check_per_epoch
        epoch = 0
        for i in range(max_iter):
            batch_mask = np.random.choice(self.train_data.shape[0], self.batch_size)
            train_data_batch = self.train_data[batch_mask] # batch_mask is an numpy array and can be used as mask
            train_labels_batch = self.train_labels[batch_mask]
            grads = self.net.gradient_bp(train_data_batch, train_labels_batch)
            for key in self.net.network:
                self.net.network[key] -= self.optimizer_params["lr"] * grads[key]
            loss = self.net.loss(train_data_batch, train_labels_batch)
            self.loss_list.append(loss)
            print("iteration: %d|  loss: %.6f" % (i, loss))

            if i % self.check_per_epoch == 0:
                epoch += 1
                train_acc = self.net.accuracy(self.train_data, self.train_labels)
                test_acc = self.net.accuracy(self.test_data, self.test_labels)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                print("epoch: %d|  train accuracy: %.6f%%" % (epoch, train_acc * 100))
                print("epoch: %d|  test accuracy: %.6f%%" % (epoch, test_acc * 100))
                if test_acc > best_test_acc:
                    print("update network %.6f%% -> %.6f%%" % (best_test_acc * 100, test_acc * 100))
                    with open("./network_files/network_%.3f.pickle" % (test_acc), "wb") as f:
                        pickle.dump([self.net.network, self.net.grads], f)
        
                plt.figure()
                x = np.arange(len(self.train_acc_list))
                plt.plot(x, self.train_acc_list, label='train acc', color='b')
                plt.plot(x, self.test_acc_list, label='test acc', color='r')
                plt.xlabel("epoch")
                plt.ylabel("accuracy")
                plt.legend()
                plt.savefig("./figures/try_learning_bp_acc_%d.png" % epoch)

                plt.close()
                plt.figure()
                x = np.arange(len(self.loss_list))
                plt.plot(x, self.loss_list, label='train loss')
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend()
                plt.savefig("./figures/try_learning_bp_loss_%d.png" % epoch)
                print("save figures done!")

        print("training done!")

    def gradient_check(self):
        data = self.train_data[:3]
        label = self.train_labels[:3]
        gradient_numerical = self.net.numerical_gradient(data, label)
        gradient_bp = self.net.gradient_bp(data, label)
        for key in self.net.network:
            diff = np.average(np.abs(gradient_numerical[key] - gradient_bp[key]))
            print(str(key) + " difference: ",  diff)