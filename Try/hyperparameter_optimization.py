import sys
import os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from Common.layerBuilder import LayerBuilder
from Common.trainer import Trainer
from MNIST_Dataset.load_data import load_mnist


HIDDEN_SIZE_LIST = [100] * 6
BATCH_SIZE = 128
VALIDATION_RATE = 0.2
EPOCHS = 50
HYPERPARAMETER_SEARCH_TIMES = 20


def shuffle(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def generate_data():
    (train_data, train_labels), (test_data, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

    train_data = train_data[:500]
    train_labels = train_labels[:500]
    train_data, train_labels = shuffle(train_data, train_labels)

    val_size = int(VALIDATION_RATE * train_data.shape[0])
    train_data = train_data[val_size:]
    train_labels = train_labels[val_size:]
    val_data = train_data[:val_size]
    val_labels = train_labels[:val_size]
    return (train_data, train_labels), (val_data, val_labels)



def _train(train_data, train_labels, val_data, val_labels, learning_rate_range, weight_decay_range, to_save=False):
    learning_rate = 10 **np.random.uniform(*learning_rate_range)
    weight_decay_lambda = 10 **np.random.uniform(*weight_decay_range)

    network = LayerBuilder(input_shape=784, output_shape=10, hidden_size_list=HIDDEN_SIZE_LIST, 
                        batch_size=BATCH_SIZE, activation_layer="relu", 
                        weight_init_std="relu", weight_decay_lambda=weight_decay_lambda)
    trainer = Trainer(train_data, train_labels, val_data, val_labels, 
                    network, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    optimizer="SGD", optimizer_params={"lr": learning_rate}, to_save=to_save)
    trainer.train()

    return (trainer.train_acc_list, trainer.test_acc_list), (learning_rate, weight_decay_lambda)


def visualize(train_acc_dict, val_acc_dict):
    cols = 5
    rows = int(np.ceil(HYPERPARAMETER_SEARCH_TIMES / cols))
    plt.figure(figsize=(16, 9))
    i = 0
    for key, value in sorted(train_acc_dict.items(), key=lambda x: x[1][-1], reverse=True):
        plt.subplot(rows, cols, i + 1)
        # plt.subplots_adjust(wspace=0.5, hspace=1.5)
        plt.title("idx: %d" % (i + 1))
        plt.ylim(0.0, 1.0)
        plt.yticks(np.arange(0.0, 1.2, 0.2))
        if i % cols:
            plt.yticks([])
        plt.xticks([])
        x = np.arange(len(value))
        plt.plot(x, value, label="train acc", color="b")
        plt.plot(x, val_acc_dict[key], label="test acc", color="r")
        plt.legend()
        i += 1
        print("idx: %d | validation accuracy: %.6f " % (i, val_acc_dict[key][-1]) + key)
        if i >= HYPERPARAMETER_SEARCH_TIMES:
            break
    plt.savefig("./hyperparameter_optimization_figures/hyperparameter_optimization4.png")


def main():
    (train_data, train_labels), (val_data, val_labels) = generate_data()
    learning_rate_range = (-6, 0) 
    weight_decay_range = (-8, 0)
    train_acc_dict = {}
    val_acc_dict = {}

    for i in range(HYPERPARAMETER_SEARCH_TIMES):
        (train_acc_list, val_acc_list), (learning_rate, weight_decay_lambda) = _train(train_data, train_labels, 
                                               val_data, val_labels, 
                                               learning_rate_range, weight_decay_range)
        key = "lr: " + str(learning_rate) + " | " + "weight_decay_lambda: " + str(weight_decay_lambda)
        train_acc_dict[key] = train_acc_list
        val_acc_dict[key] = val_acc_list
        
    visualize(train_acc_dict, val_acc_dict)


if __name__ == "__main__":
    main()