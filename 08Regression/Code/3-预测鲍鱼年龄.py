import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regression

filename = "../Data/abalone.txt"
abX, abY = regression.load_dataset(filename)
#
# yHat01 = regression.lwlr_test(abX[0:99], abX[0:99], abY[0:99], 0.1)
# yHat1 = regression.lwlr_test(abX[0:99], abX[0:99], abY[0:99], 1)
# yHat10 = regression.lwlr_test(abX[0:99], abX[0:99], abY[0:99], 10)
#
# print(regression.rss_error(yHat01.T, abY[0:99]))
# print(regression.rss_error(yHat1.T, abY[0:99]))
# print(regression.rss_error(yHat10.T, abY[0:99]))
#
# # 新数据
# newYHat01 = regression.lwlr_test(abX[100:199], abX[0:99], abY[0:99], 0.1)
# print(regression.rss_error(newYHat01.T, abY[100:199]))
#
# newYHat1 = regression.lwlr_test(abX[100:199], abX[0:99], abY[0:99], 1)
# print(regression.rss_error(newYHat1.T, abY[100:199]))
#
# newYHat10 = regression.lwlr_test(abX[100:199], abX[0:99], abY[0:99], 10)
# print(regression.rss_error(newYHat10.T, abY[100:199]))
#
# # 简单线性回归
# ws = regression.stand_regression(abX[0:99], abY[0:99])
# yHat = np.mat(abX[100:199]) * ws
# print(regression.rss_error(abY[100:199], yHat.T.A))


ridgeWeights = regression.ridge_test(abX, abY)
print(ridgeWeights)

plt.plot(ridgeWeights)
plt.show()


