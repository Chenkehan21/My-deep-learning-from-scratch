import sys, os
sys.path.append(os.pardir)
from MNIST_Dataset.load_data import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
print("x_train's type: ", type(x_train))
print("x_train's shape: ", x_train.shape)
print("x_test's type: ", type(x_test))
print("x_test's shape: ", x_test.shape)
# print(sys.path)
# print(os.pardir)