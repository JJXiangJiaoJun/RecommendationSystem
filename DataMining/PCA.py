import numpy as np
from sklearn import datasets


class PACTest(object):
    def __init__(self):
        pass

    def loadIris(self):
        """加载鸢尾花数据集中的特征作为PCA的原始数据集并进行标准化
        """
        data = datasets.load_iris()["data"]
        return data

    # 标准化数据
    def Standard(self, data):
        # axis = 0 按列取均值
        mean_vector = np.mean(data, axis=0)
        return mean_vector, data - mean_vector

    # 计算协方差矩阵
    def getCovMatrix(self, newData):
        # rowvar = 0 表示数据的每一列代表一个 feature

        return np.cov(newData, rowvar=0)

    def getFValueAndFVector(self, covMatrix):
        fValue, fVector = np.linalg.eig(covMatrix)
        return fValue, fVector

    # 得到特征向量矩阵
    def getVectorMatrix(self, fValue, fVector, k):
        fValueSort = np.argsort(fValue)
        fValueTopN = fValueSort[:-(k + 1):-1]
        return fVector[:, fValueTopN]

    def getResult(self, data, vectorMatrix):
        return np.dot(data, vectorMatrix)


if __name__ == "__main__":
    pcatest = PACTest()
    data = pcatest.loadIris()

    mean_vector, newData = pcatest.Standard(data)
    covMatrix = pcatest.getCovMatrix(newData)
    print("协方差矩阵为：\n{}".format(covMatrix))

    # 得到特征值和特征向量
    fValue, fVector = pcatest.getFValueAndFVector(covMatrix)
    print("特征值为:{}".format(fValue))
    print("特征向量为：\n{}".format(fVector))

    # 得到要降到k维的特征向量矩阵
    vectorMatrix = pcatest.getVectorMatrix(fValue, fVector, k=2)
    print("k 维特征向量矩阵为：\n{}".format(vectorMatrix))

    # 计算结果
    result = pcatest.getResult(newData, vectorMatrix)
    print("最终降维结果为：\n{}".format(result))

    # 得到重构数据为
    print("最终重构结果为:\n{}".format(np.mat(result) * vectorMatrix.T + mean_vector))


