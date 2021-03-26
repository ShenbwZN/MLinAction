"""
线性回归
"""
import numpy as np
import pandas as pd


# 加载数据
def load_dataset(filename):
    df = pd.read_csv(filename, sep="\t", header=None)
    index = df.keys().values  # 得到每列的标签名
    dataMat = df[index[:-1]].values
    labelMat = df[index[-1]].values
    return dataMat, labelMat


# 标准法公式计算，(XTX)^-1XTy
def stand_regression(x_arr, y_arr):
    xMat = np.mat(x_arr)
    yMat = np.mat(y_arr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# 局部加权线性回归-返回预测值
def lwlr(test_point, x_arr, y_arr, k=1.0):
    xMat = np.mat(x_arr)
    yMat = np.mat(y_arr).T
    m = xMat.shape[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = test_point - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0:
        print("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return test_point * ws


# 局部加权回归测试(test_arr测试集)
def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = test_arr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return yHat


# 误差平方和
def rss_error(y_arr, y_hat):
    return ((y_hat - y_arr) ** 2).sum()


# 岭回归
def ridge_regression(x_mat, y_mat, lam=0.2):
    xTx = x_mat.T * x_mat
    denom = xTx + np.eye(x_mat.shape[1]) * lam
    if np.linalg.det(denom) == 0:
        print("矩阵不可逆")
        return
    ws = denom.I * (x_mat.T * y_mat)
    return ws


# 岭回归测试
def ridge_test(x_arr, y_arr):
    xMat = np.mat(x_arr)
    yMat = np.mat(y_arr).T
    yMean = np.mean(yMat, axis=0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, axis=0)
    xVar = np.var(xMat, axis=0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridge_regression(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat
