import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    if x > 0:
        return 1
    if x < 0:
        return 0


# important skill !
def step_function_numpy(x):
    return np.array(x > 0, dtype=np.int16)


# numpy array can be input, broadcast
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# broadcast
def ReLU(x):
    return np.maximum(0, x)


if __name__ == "__main__":
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(60, 10))
    x = np.linspace(-10, 10, 1000)
    y1 = step_function_numpy(x)
    y2 = sigmoid(x)
    y3 = ReLU(x)
    ax1.plot(x, y1)
    ax1.set_title('step function')
    ax2.plot(x, y2)
    ax2.set_title('sigmoid function')
    ax3.plot(x, y3)
    ax3.set_title('ReLU function')
    plt.savefig('./activate_functions.png')