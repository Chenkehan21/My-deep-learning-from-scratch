import sys, os
sys.path.append(os.pardir)
import numpy as np
from Try.try_nn import Net
from Activate_functions.activate_functions import ReLU, Sigmoid
from Loss_functions.loss_functions import mse_loss, cross_entropy_loss, cross_entropy_loss_one_hot
from Numerical_differentiation.numerical_differentiation import gradient_batch


def main():
    input = np.random.random((3, 4))
    input_shape = input.shape[1]
    output_shape = 10
    batch_size = input.shape[0]
    net = Net(input_shape, output_shape, batch_size, 8)
    res = net.forward(input)
    labels = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    loss = cross_entropy_loss_one_hot(res, labels)
    print("loss: ", loss)

    """Pay attention: 
    1. in loss_func the parameter x is useless
    2. we should pass in cross_entropy_loss_one_hot() net.forward(input) rather than res
       because in function gradient(f, x) we will give x which would be network['w1'] here
       a small change, if use res we've calculated before the small change on w1 won't be used
       then the gradient will all be 0.0 since f(tmp + delta) = f(tmp - delta) 
    """
    loss_func = lambda x: cross_entropy_loss_one_hot(net.forward(input), labels)
    for key, val in net.network.items():
        grad = gradient_batch(loss_func, net.network['w1'])
        print("dL/d" + str(key) + ":\n", grad)
        print()


if __name__ == "__main__":
    main()