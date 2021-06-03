import sys
import os
sys.path.append(os.pardir)

from common.layerBuilder import LayerBuilder
from common.trainer import Trainer
from MNIST_Dataset.load_data import load_mnist


BATCH_SIZE = 128
HIDDEN_SIZE = 50
EPOCH = 50
OPTIMIZER = "SGD"
LEARNING_RATE = 1e-3
ACTIVATION_LAYER = "ReLU"
WEIGHT_DECAY_LAMBDA = 0.9
WEIGHT_INIT_STD = "ReLU"


if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    input_shape = 784
    output_shape = 10
    network = LayerBuilder(input_shape, output_shape, hidden_size_list=[HIDDEN_SIZE] * 3, 
                           batch_size=BATCH_SIZE, activation_layer=ACTIVATION_LAYER,
                           weight_init_std=WEIGHT_INIT_STD, weight_decay_lambda=WEIGHT_DECAY_LAMBDA)
    optimizer_params = {"lr": LEARNING_RATE}
    trainer = Trainer(train_data, train_labels, test_data, test_labels, network, 
                      batch_size=BATCH_SIZE, epochs=EPOCH, optimizer=OPTIMIZER, optimizer_params=optimizer_params)

    trainer.gradient_check()
    # data = train_data[:3]
    # label = train_labels[:3]
    # print(data.shape)
    # print("numerical_gradient: ", network.gradient_numerical(data, label))
    # print("gradient_bp: ", network.gradient(data, label))