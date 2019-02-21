from sklearn.linear_model import LinearRegression
import numpy as np
import random

x = [round(10 * random.random(), 2) for _ in range(100)]
y = [[round(2 * i + 10 + random.uniform(-1, 1), 2)] for i in x]
y = np.array(y)
x_mat = np.array([[i] for i in x])
w = np.array([[1]])

reg = LinearRegression().fit(x_mat, y)

print(reg.coef_)
print(reg.intercept_)