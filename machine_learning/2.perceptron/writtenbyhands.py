import numpy as np
import matplotlib.pyplot as plt
X = np.array([[3, 3], [4, 3], [1, 1]])
Y = np.array([[1], [1], [-1]])

# init
w = np.transpose([0, 0])
b = 0
print(X.shape, Y.shape, w.shape)
print(X[0])
lr = 1

while True:
    wrong_count = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        np.inner(x, w)
        tmp = y * (np.inner(x, w) + b)
        if tmp > 0:
            continue
        w = w + lr * y * x
        b = b + lr * y
        wrong_count += 1
        print("w:", w, "b:", b)
    if wrong_count == 0:
        print("train done")
        break


plt.scatter([3, 4, 1], [3, 3, 1])
plt.plot([3, 4, 1], [-i + 3 for i in [3, 4, 1]])
plt.show()