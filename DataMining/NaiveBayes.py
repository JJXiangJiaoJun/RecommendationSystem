# coding=utf-8

import numpy as np


class NaiveBayesian(object):
    def __init__(self, alpha):
        self.classP = dict()
        self.classP_feature = dict()
        self.alpha = alpha  # 平滑系数

    def createData(self):
        data = np.array(
            [
                [320, 204, 198, 265],
                [253, 53, 15, 2243],
                [53, 32, 5, 325],
                [63, 50, 42, 98],
                [1302, 523, 202, 5430],
                [32, 22, 5, 143],
                [105, 85, 70, 322],
                [872, 730, 840, 2762],
                [16, 15, 13, 52],
                [92, 70, 21, 693],
            ]
        )

        labels = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        return data, labels

    def gaussian(self, mu, sigma, x):
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def calMuAndSigma(self, feature):
        mu = np.mean(feature)
        sigma = np.std(feature)
        return mu, sigma

    def train(self, data, labels):
        numData = len(labels)
        numFeatures = len(data[0])

        # 是异常用户的概率
        self.classP[1] = (
            (sum(labels) + self.alpha) * 1.0 / (numData + self.alpha * len(set(labels)))
        )
        # 不是异常用户的概率
        self.classP[0] = 1 - self.classP[1]

        # 用来存放每个label下每个特征标签下对应的高斯分布中的均值和方差
        self.classP_feature = dict()
        # 遍历每个特征标签
        for c in set(labels):
            self.classP_feature[c] = {}
            for i in range(numFeatures):
                # numpy 索引方式
                feature = data[np.equal(labels, c), i]
                self.classP_feature[c][i] = self.calMuAndSigma(feature)

    def predict(self, x):
        # 预测新用户是否为异常用户
        label = -1
        maxP = 0

        for key in self.classP.keys():
            # 计算后验概率
            label_p = self.classP[key]
            currentP = 1.0
            feature_p = self.classP_feature[key]
            j = 0
            for fp in feature_p.keys():
                currentP *= self.gaussian(feature_p[fp][0], feature_p[fp][1], x[j])
                j += 1

            if currentP * label_p > maxP:
                maxP = currentP * label_p
                label = key

        return label


if __name__ == "__main__":
    nb = NaiveBayesian(1.0)
    data, labels = nb.createData()
    nb.train(data, labels)
    labels = nb.predict(np.array([134, 84, 235, 349]))
    print("未知类型的用户行为数据为：[134,84,235,349],该用户的可能类型为；{}".format(labels))
