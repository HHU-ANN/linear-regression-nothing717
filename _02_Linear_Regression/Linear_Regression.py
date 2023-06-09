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
    learning_rate = 1e-10
    max_iter = 10000
    alpha = 0.1
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    # 梯度下降迭代
    for i in range(max_iter):
        y_pred = np.dot(X, w) + b
        dw_reg = alpha * np.sign(w)
        residuals = y - y_pred
        dw = (-2/n_samples) * np.dot(X.T, residuals) + dw_reg
        db = (-2/n_samples) * np.sum(residuals)
        w -= learning_rate * dw
        b -= learning_rate * db
    return np.dot(data, w) + b

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y