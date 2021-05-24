import numpy as np
from logic_circuit import perceptron_orgate, perceptron_andgate, perceptron_nandgate, execuate

'''
x1  x2  a(x1 or x2)  b(x1 nand x2)  y(a and b) i.e. "xor"   
0    0       0            1              0
1    0       1            1              1
0    1       1            1              1
1    1       1            0              0
'''

def xor(x1, x2):
    a = perceptron_orgate(x1, x2)
    b = perceptron_nandgate(x1, x2)
    y = perceptron_andgate(a, b)
    if y <= 0:
        return 0
    else:
        return 1


def main():
    datas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a, b in datas:
        print(xor(a, b))


if __name__ == "__main__":
    main()