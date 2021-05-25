import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    delta = 1e-4 # can't be too small, otherwise will cause rounding error
    return np.abs(f(x - delta) - f(x + delta)) / (2 * delta)

def func1(x):
    return 0.01 * x**2 + 0.1 * x

def line(x1, y1, k, x):
    return y1 + k * (x - x1)

x = np.linspace(0, 20, 1000)
y = func1(x)
diff_5 = numerical_diff(func1, 5)
diff_10 = numerical_diff(func1, 10)
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
plt.show()