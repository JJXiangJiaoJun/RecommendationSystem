import numpy as np
import math
import collections
from collections import Counter, OrderedDict, namedtuple


class DiscreteByEntropy(object):
    """使用信息熵来对数据进行离散化操作
    """

    def __init__(self, group, threshold):
        """
        Parameters
        ----------
        group : int
            最大分组数
        threshold : float
            停止划分的最小熵
        """

        self.maxGroup = group
        self.minInfoThreshold = threshold
        self.result = dict()

    def loadData(self):
        data = np.array(
            [
                [56, 1], [87, 1], [129, 0], [23, 0], [342, 1],
                [641, 1], [63, 0], [2764, 1], [2323, 0], [453, 1],
                [10, 1], [9, 0], [88, 1], [222, 0], [97, 0],
                [2398, 1], [592, 1], [561, 1], [764, 0], [121, 1],
            ]
        )

        return data

    def calEntropy(self, data):
        numData = len(data)
        labelCounts = Counter(data[:, 1])
        # for feature in data:
        #     # 获得标签
        #     oneLabel = feature[-1]
        #     # 如果标签不在新定义的字典里则创建该标签
        #     labelCounts.setdefault(oneLabel, 0)
        #     labelCounts[oneLabel] += 1

        Ent = 0.0
        for key in labelCounts.keys():
            prob = float(labelCounts[key]) / numData
            Ent -= prob * math.log(prob, 2)

        return Ent

    def split(self, data):
        """对一组数据寻找最佳切分点：遍历所有属性值，数据按照该属性进行划分，使得对应的熵最小
        """
        minEnt = np.inf
        # 记录最终分割索引
        index = -1
        sortData = data[np.argsort(data[:, 0])]
        # 初始化最终分割数据后的熵
        lastE1, lastE2 = -1, -1
        # 返回的数据结构，包含数据和对应的熵
        S1 = dict()
        S2 = dict()

        for i in range(len(sortData)):
            # 分割数据集
            splitData1, splitData2 = sortData[:i + 1], sortData[i + 1:]
            entropy1, entropy2 = (
                self.calEntropy(splitData1),
                self.calEntropy(splitData2)
            )  # 计算信息熵

            entropy = entropy1 * len(splitData1) / len(sortData) + \
                      entropy2 * len(splitData2) / len(sortData)
            # 如果调和平均熵小于最小值

            if entropy < minEnt:
                minEnt = entropy
                index = i
                lastE1 = entropy1
                lastE2 = entropy2

        S1["entropy"] = lastE1
        S1["data"] = sortData[:index + 1]
        S2["entropy"] = lastE2
        S2["data"] = sortData[index + 1:]

        return S1, S2, minEnt

    def train(self, data):
        # 需要遍历的key
        needSplitKey = [0]

        self.result.setdefault(0, {})
        self.result[0]["entropy"] = np.inf
        self.result[0]["data"] = data
        group = 1
        for key in needSplitKey:
            S1, S2, entropy = self.split(self.result[key]["data"])
            # 如果满足条件
            if entropy > self.minInfoThreshold and group < self.maxGroup:
                self.result[key] = S1
                newKey = max(self.result.keys()) + 1
                self.result[newKey] = S2
                needSplitKey.extend([key, newKey])
                group += 1
            else:
                break


if __name__ == "__main__":
    dbe = DiscreteByEntropy(group=6, threshold=0.5)
    data = dbe.loadData()
    dbe.train(data)

    print("result is {}".format(dbe.result))
