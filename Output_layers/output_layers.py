import numpy as np

def identify_function(x):
    return x

def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(exp)
    return exp / exp_sum

def safe_softmax(x):
    '''if x is a batch of data, we need to subtract the max value by batch
    however in numpy it's only allowed to broadcast beween (a, b) and (b,)
    so we need to transpose input!
    '''
    if x.ndim == 2:
        x = x.T
        x -= np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == "__main__":
    x = np.array([0.3, 2.9, 4.0])
    # x = np.array([1010, 1000, 990])
    y1 = softmax(x)
    y2 = safe_softmax(x)
    print("y1: ", y1)
    print("sum of y1: ", np.sum(y1))
    print("y2: ", y2)
    print("sum of y2: ", np.sum(y2))