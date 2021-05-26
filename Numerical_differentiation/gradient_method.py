import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
from numerical_differentiation import gradient, gradient_batch, func3


LEARNING_RATE = 0.1
EPOCH = 50


def main():
    x0 = np.arange(-2.5, 2.5, 0.25)
    x1 = np.arange(-2.5, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    X = X.flatten()
    Y = Y.flatten()
    value = func3(np.array([x0, x1]))
    print(np.array([x0, x1]))
    print(value)

    plt.figure()


    x, y = np.random.uniform(-10, 10, 2)
    for i in range(EPOCH):
        res = func3(np.array([x, y]))
        grad = gradient_batch(func3, np.array([x, y]))
        x -= LEARNING_RATE * grad[0]
        y -= LEARNING_RATE * grad[1]
        print(res)


if __name__ == "__main__":
    main()
