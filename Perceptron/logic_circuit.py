import numpy as np


# input is 0 or 1

def andgate(a, b):
    # print("AndGate")
    w1, w2, thereshold = 1.0, 1.0, 1.0
    res = a * w1 + b * w2
    if res > thereshold:
        return 1
    else:
        return 0


def orgate(a, b):
    # print("OrGate: ")
    w1, w2, threshold = 1.0, 1.0, 0.5
    res = a * w1 + b * w2
    if res < threshold:
        return 0
    else:
        return 1


def nandgate(a, b):
    # print("NAndeGate: ")
    w1, w2, threshold = 1.0, 1.0, 1.5
    res = a * w1 + b * w2
    if res > threshold:
        return 0
    else:
        return 1

def perceptron_andgate(a, b):
    # print("Perceptron_AndGate: ")
    input = np.array([[a, b, 1]])
    
    # add bias into w directly
    w = np.array([[1.0, 1.0, -1.0]]).T
    res = np.dot(input, w)
    if res.item() <= 0:
        return 0
    else:
        return 1


def perceptron_orgate(a, b):
    # print("Perceptron OrGate: ")
    input = np.array([[a, b, 1]])

    # create weight matrix and bias matrix seperately then concatenate them together.
    weight = np.array([[1.0, 1.0]])
    bias = np.array([[-0.5]])
    w = np.concatenate((weight, bias), axis=1).T
    res = np.dot(input, w)
    if res.item() <= 0:
        return 0
    else:
        return 1


def perceptron_nandgate(a, b):
    # print("Perceptron_NAndGate: ")
    input = np.array([[a, b, 1]])
    w = np.array([[1.0, 1.0, -1.5]]).T
    res = np.dot(input, w)
    if res.item() <= 0:
        return 1
    else:
        return 0


def execuate(method):
    datas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a, b in datas:
        print(method(a, b))


def main():
    methods = [andgate, perceptron_andgate, 
               orgate, perceptron_orgate,
               nandgate, perceptron_nandgate
            ]
    for method in methods:
        execuate(method)
        print()


if __name__ == "__main__":
    main()