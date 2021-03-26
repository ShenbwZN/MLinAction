"""
ws = (XTWX)^-1XTWy
"""
import regression
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
filename = "../Data/ex0.txt"
xArr, yArr = regression.load_dataset(filename)
# print(yArr[0])
k = 0.03
# 得到所有点的点估计
yHat = regression.lwlr_test(xArr, xArr, yArr, k)
# print(yHat)

# 排序
xMat = np.mat(xArr)
sortInd = xMat[:, 1].argsort(axis=0)
xSort = xMat[sortInd][:, 0, :]

# 绘图
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[sortInd], label="k = " + str(k))
ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=10, c="red")

plt.legend()
plt.show()
