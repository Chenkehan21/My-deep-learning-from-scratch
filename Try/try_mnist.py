import sys, os
sys.path.append(os.pardir)
from MNIST_Dataset.load_data import load_mnist
import matplotlib.pyplot as plt
import numpy as np
from try_nn import Net

BATCH_SIZE = 100

def get_acc(test_data, test_labels, net):
    correct = 0
    for i, data in enumerate(test_data):
        output = net.forward(data)
        output = np.argmax(output)
        if output == test_labels[i]:
            correct += 1
    return correct / len(test_data)


def get_batch_acc(test_data, test_labels, net):
    correct = 0
    for i in range(0, len(test_data), BATCH_SIZE):
        output = net.forward(test_data[i : BATCH_SIZE + i])
        output = np.argmax(output, axis=1)
        correct += np.sum(output == test_labels[i : BATCH_SIZE + i])
    return correct / len(test_data)


if __name__ == "__main__":
    (train_img, train_label), (test_img, test_label) = load_mnist(flatten=True, normalize=True)
    # random_net = Net(784, 10, 1, 50)
    random_net2 = Net(784, 10, BATCH_SIZE, 50)
    acc = get_acc(test_img, test_label, random_net2)
    print("acc: %.6f%%" % (acc * 100))
    # image = train_img[10].reshape((28, 28))
    # print(image)
    # plt.imshow(image)
    # plt.show()