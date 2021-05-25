import numpy as np

def identify_function(x):
    return x

def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(exp)
    return exp / exp_sum

def safe_softmax(x):
    c = np.max(x)
    exp = np.exp(x - c)
    exp_sum = np.sum(exp)
    return exp / exp_sum


if __name__ == "__main__":
    x = np.array([0.3, 2.9, 4.0])
    # x = np.array([1010, 1000, 990])
    y1 = softmax(x)
    y2 = safe_softmax(x)
    print("y1: ", y1)
    print("sum of y1: ", np.sum(y1))
    print("y2: ", y2)
    print("sum of y2: ", np.sum(y2))