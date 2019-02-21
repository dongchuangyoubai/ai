import numpy as np
import random
import matplotlib.pyplot as plt

x = [round(10 * random.random(), 2) for _ in range(100)]  # m * n
y = [[round(2 * i + 10 + random.uniform(-1, 1), 2)] for i in x]  # m * 1
y = np.array(y)
x_mat = np.array([[1, i] for i in x])  # m * (n + 1)
w = np.array([[1], [1]])  # m * 1


epoches = 100000
learning_rate = 0.001
for i in range(epoches):
    h = np.dot(x_mat, w)
    w = w - (1 / 100) * learning_rate * np.dot(np.transpose(x_mat), (h - y))

plt.scatter(x, y)
plt.plot(x, [w[1][0] * i + w[0][0] for i in x])
plt.show()
print(w)