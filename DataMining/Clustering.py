import numpy as np
import pandas as pd
import random


class KMeans(object):
    def __init__(self):
        pass

    # 加载数据集
    def loadData(self, file):
        return pd.read_csv(file, header=0, sep=',')

    def filterAnomalyValue(self, data):
        # 数据清洗，去除异常值
        upper = np.mean(data["price"] + 3 * np.std(data["price"]))
        lower = np.mean(data["price"] - 3 * np.std(data["price"]))

        upper_limit = upper if upper > 5000 else 5000
        lower_limit = lower if lower > 5000 else 1

        print("最大异常值为；{}，最小异常数值为：{}".format(upper_limit, lower_limit))

        # 过滤掉大于最大异常值和小于最小异常值的
        newData = data[(data["price"] < upper_limit) & (data["price"] > lower_limit)]

        return newData, upper_limit, lower_limit

    def initCenters(self, values, K, Cluster):
        random.seed(100)
        oldCenters = list()
        for i in range(K):
            index = random.randint(0, len(values))
            Cluster.setdefault(i, {})
            Cluster[i]["center"] = values[index]
            Cluster[i]["value"] = []

            oldCenters.append(values[index])
        return oldCenters, Cluster

    def distance(self, price1, price2):
        return np.sqrt(pow((price1 - price2), 2))

    def train(self, data, K, maxIters):
        # 停止聚类条件为：更新后簇类中心不变，或者达到最大迭代次数
        Cluster = dict()  # 最终聚类中心
        oldCenters, Cluster = self.initCenters(data, K, Cluster)
        print("初始聚类中心为：\n{}".format(oldCenters))
        # 标志标量，若为True，则继续迭代
        clusterChanged = True
        i = 0
        while clusterChanged:
            for price in data:
                # 每条数据与最近簇类中心距离，初始化为正无穷
                minDistance = np.inf
                # 每条数据对应索引,初始化为-1
                minIndex = -1
                for key in Cluster.keys():
                    # 计算每条数据到簇类中心的距离
                    dis = self.distance(price, Cluster[key]["center"])
                    if dis < minDistance:
                        minDistance = dis
                        minIndex = key
                Cluster[minIndex]["value"].append(price)

            newCenters = list()
            for key in Cluster.keys():
                newCenter = np.mean(Cluster[key]["value"])
                Cluster[key]["center"] = newCenter
                newCenters.append(newCenter)
            print('第{}次迭代后的簇类中心为:{}'.format(i, newCenters))
            if oldCenters == newCenters or i > maxIters:
                clusterChanged = False
            else:
                oldCenters = newCenters
                i += 1
                # 删除Cluster 中记录的簇类之
                for key in Cluster.keys(): Cluster[key]["value"] = []
        return Cluster

    def SSE(self, data, mean):
        newData = np.mat(data) - mean
        return (newData * newData.T).tolist()[0][0]

    def bisectKMeans(self, data, K=7):
        clusterSSEResult = dict()
        clusterSSEResult.setdefault(0, {})
        clusterSSEResult[0]["value"] = data
        clusterSSEResult[0]['sse'] = np.inf
        clusterSSEResult[0]['center'] = np.mean(data)

        while len(clusterSSEResult) < K:
            maxSSE = np.inf
            maxSSEKey = 0
            # 找到最大SSE值对应数据，进行kmeans聚类
            for key in clusterSSEResult.keys():
                if clusterSSEResult[key]["sse"] > maxSSE:
                    maxSSE = clusterSSEResult[key]["sse"]
                    maxSSEkey = key

            clusterResult = self.train(clusterSSEResult[maxSSEkey]['value'], K=2, maxIters=200)
            # 删除clusterSSE的minKey对应的值
            del clusterSSEResult[maxSSEKey]

            # 将经过聚类后的结果赋值给clusterSSEResult
            clusterSSEResult.setdefault(maxSSEKey, {})
            clusterSSEResult[maxSSEKey]['center'] = clusterResult[0]["center"]
            clusterSSEResult[maxSSEKey]['value'] = clusterResult[0]['value']
            clusterSSEResult[maxSSEKey]['sse'] = self.SSE(clusterResult[0]['value'],
                                                          clusterResult[0]['center'])

            maxKey = max(clusterSSEResult.keys()) + 1
            clusterSSEResult.setdefault(maxKey, {})
            clusterSSEResult[maxKey]['center'] = clusterResult[1]['center']
            clusterSSEResult[maxKey]['value'] = clusterResult[1]['value']
            clusterSSEResult[maxKey]['sse'] = self.SSE(clusterResult[1]['value'], clusterResult[1]['center'])

        return clusterSSEResult



if __name__ == "__main__":
    file = '../data/sku-price/skuid_price.csv'
    km = KMeans()
    data = km.loadData(file)
    newData, upper_limit, lower_limit = km.filterAnomalyValue(data)
    # Cluster = km.train(newData["price"].values, K=7, maxIters=200)
    clusterSSE = km.bisectKMeans(newData['price'].values,K = 7)
    print(clusterSSE)
