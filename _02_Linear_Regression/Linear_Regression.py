# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    X, y = read_data()
    alpha = 0.1
    X_T = np.transpose(X)
    n = X.shape[1]
    beta = np.linalg.inv(X_T @ X + alpha * np.identity(n)) @ X_T @ y
    return data @ beta
    
def lasso(data):
    X, y = read_data()
    learning_rate = 0.01
    max_iter = 1000
    alpha = 0.5
    tol = 1e-4
    n, m = X.shape
    X = np.c_[np.ones(n), X]
    w = np.zeros(m + 1)
    for i in range(max_iter):
        w_old = w.copy()
        for j in range(m + 1):
            if j == 0:
                w[j] = w[j] - (learning_rate / n) * sum(X @ w - y)
            else:
                a = 2 * (X[:, j] ** 2).sum()
                b = 2 * (X[:, j] * (X @ w - y)).sum()
                if b < -alpha:
                    w[j] = (b + alpha) / a
                elif b > alpha:
                    w[j] = (b - alpha) / a
                else:
                    w[j] = 0

        if np.linalg.norm(w - w_old) < tol:
            break
    X1 = np.append(1, data)
    return X1 @ w

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y