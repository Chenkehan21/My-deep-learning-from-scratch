import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, os

from numpy.lib.histograms import histogram
sys.path.append(os.pardir)
from numerical_differentiation import gradient, gradient_batch, func3


LEARNING_RATE = 0.1
EPOCH = 50


def main():
    fig = plt.figure()
    x0 = np.arange(-12.0, 12.0, 0.25)
    x1 = np.arange(-12.0, 12.0, 0.25)
    X0, X1 = np.meshgrid(x0, x1)
    ax1 = Axes3D(fig)
    ax1.plot_surface(X0, X1, X0**2 + X1**2, rstride=1, cstride=1, alpha=0.5, shade=False)
    ax1.set_xlabel("$x_{0}$")
    ax1.set_ylabel("$x_{1}$")
    ax1.set_title("$f(x_{0}, x_{1}) = x_{0}^{2} + x_{1}^2$")

    X = np.array([10., 10.])
    histroy = []
    for i in range(EPOCH):
        histroy.append(X.copy())
        res = func3(X)
        grad = gradient_batch(func3, X)
        X -= LEARNING_RATE * grad
        print(res)
    histroy = np.array(histroy)
    print(histroy)
    value = np.array([np.sum(x**2) + 5 for x in histroy])
    ax1.scatter(histroy[:, 0], histroy[:, 1], value, alpha=1.0, c='r')
    plt.savefig("./gradient_descent.png")

if __name__ == "__main__":
    main()
