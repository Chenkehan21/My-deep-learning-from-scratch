import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def numerical_diff(f, x):
    delta = 1e-4 # can't be too small, otherwise will cause rounding error
    return np.abs(f(x - delta) - f(x + delta)) / (2 * delta)


def gradient(f, x):
    delta = 1e-5
    grad = np.zeros_like(x)
    # print("grad in gradient: ", grad)
    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = float(tmp) + delta # use array to calculate directly
        f1 = f(x)
        x[idx] = tmp - delta
        f2 = f(x)
        grad[idx] = (f1 - f2) / (2 * delta)
        x[idx] = tmp

    return grad


def gradient_batch(f, X):
    if X.ndim == 1:
        return gradient(f, X)
    else:
        grad = np.zeros_like(X)
        # print(grad)
        for idx, x in enumerate(X):
            grad[idx] = gradient(f, x)
    
        return grad


def  numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad


def func1(x):
    return 0.01 * x**2 + 0.1 * x


def func2(x):
    return x[0]**2 + x[1]**2


def func3(x):
    if x.ndim == 1:
        return np.sum(x**2) # remember to sum!
    else:
        return np.sum(x**2, axis=1)


def line(x1, y1, k, x):
    return y1 + k * (x - x1)


def diff_func1():
    x = np.linspace(0, 20, 1000)
    y = func1(x)
    diff_5 = numerical_diff(func1, 5)
    diff_10 = numerical_diff(func1, 10)
    print("diff_5: ", diff_5)
    print("diff_10: ", diff_10)
    y2 = line(5, func1(5), diff_5, x)
    y3 = line(10, func1(10), diff_10, x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    ax1.plot(x, y)
    ax2.plot(x, y)
    ax1.plot(x, y2)
    ax2.plot(x, y3)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    plt.savefig('./diff.png')


def func2_gradient():
    tmp_func2 = lambda x1: x1**2 + 4**2
    diff_x0_3 = numerical_diff(tmp_func2, 3)
    print("func2 partial deriative, dy/dx1(x1 = 3, x2 = 4)", diff_x0_3)
    x1 = np.array([3.0, 4.0])
    grad = gradient(func2, x1)
    print("func2 gradient (3, 4): ", grad)


def bowl_function():
    fig = plt.figure()
    x0 = np.arange(-2.0, 2.5, 0.25)
    x1 = np.arange(-2.0, 2.5, 0.25)
    X0, X1 = np.meshgrid(x0, x1)
    ax1 = Axes3D(fig)
    ax1.plot_surface(X0, X1, X0**2 + X1**2)
    ax1.set_xlabel("$x_{0}$")
    ax1.set_ylabel("$x_{1}$")
    ax1.set_title("$f(x_{0}, x_{1}) = x_{0}^{2} + x_{1}^2$")
    plt.savefig("./bowl_function.png")

    X0 = X0.flatten()
    X1 = X1.flatten()
    plt.close()
    plt.figure()
    grad = gradient_batch(func3, np.array([X0, X1]))
    plt.title("$f(x_{0}, x_{1}) = x_{0}^{2} + x_{1}^{2}$", fontsize=15)
    plt.xlabel("$x_{0}$", fontsize=13)
    plt.ylabel("$x_{1}$", fontsize=13)
    plt.quiver(X0, X1, -grad[0], -grad[1], angles='xy', color="#123456")
    plt.grid()
    plt.draw()
    plt.savefig("./gradient.png")


def main():
    diff_func1()
    func2_gradient()
    bowl_function()


if __name__ == "__main__":
    main()