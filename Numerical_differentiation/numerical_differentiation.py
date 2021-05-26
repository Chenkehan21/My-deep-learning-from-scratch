import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    delta = 1e-4 # can't be too small, otherwise will cause rounding error
    return np.abs(f(x - delta) - f(x + delta)) / (2 * delta)


def gradient(f, x):
    delta = 1e-5
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp + delta # use array to calculate directly
        f1 = f(x)

        x[idx] = tmp - delta
        f2 = f(x)
        grad[idx] = np.abs(f1 - f2) / (2 * delta)
    return grad


def func1(x):
    return 0.01 * x**2 + 0.1 * x


def func2(x):
    return x[0]**2 + x[1]**2


def line(x1, y1, k, x):
    return y1 + k * (x - x1)


if __name__ == "__main__":
    x = np.linspace(0, 20, 1000)
    y = func1(x)

    # func1
    diff_5 = numerical_diff(func1, 5)
    diff_10 = numerical_diff(func1, 10)
    print("diff_5: ", diff_5)
    print("diff_10: ", diff_10)
    y2 = line(5, func1(5), diff_5, x)
    y3 = line(10, func1(10), diff_10, x)

    # func2, x1 = 3, x2 = 4, dy/dx1
    def tmp_func2(x1):
        return x1**2 + 4**2
    y4 = tmp_func2(x)
    diff_x0_3 = numerical_diff(tmp_func2, 3)
    print("func2, dy/dx1(x1 = 3, x2 = 4)", diff_x0_3)

    # try gradient
    x1 = np.array([3.0, 4.0])
    y5 = func2(x1)
    grad = gradient(func2, x1)
    print("func2 gradient (3, 4): ", grad)

    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    ax1.plot(x, y)
    ax2.plot(x, y)
    ax1.plot(x, y2)
    ax2.plot(x, y3)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("f(x)")
    plt.show()
    '''