"""
(XTX)^-1XTy
"""
import regression
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
filename = "../Data/ex0.txt"
xArr, yArr = regression.load_dataset(filename)
print(xArr[:2])

# 计算回归系数
ws = regression.stand_regression(xArr, yArr)
print("ws = ", ws)

xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHat = xMat * ws

# 计算相关系数
corr = np.corrcoef(yHat.T, yMat)
print("corr = ", corr)

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111)
# 原始数据
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

# 绘制回归直线(首先将点进行排序在进行预测计算)
# 其实已经得到ws了，还可根据y = wsX图画，其中X取最值带入计算（只用两个点）(仅使用直线时)
xCopy = xMat.copy()
xCopy.sort(axis=0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()
