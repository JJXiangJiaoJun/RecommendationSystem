import random
import math
import os
import json


class ItemCFRec(object):
    def __init__(self, datafile, ratio):
        # 原始数据路径文件
        self.datafile = datafile
        self.ratio = ratio
        self.data = self.loadData()
        self.trainData, self.testData = self.splitData(3, 47)
        self.items_sim = self.ItemSimilarityBest()

    def loadData(self):
        # 加载评分数据到data
        print("加载数据....")
        data = []
        for line in open(self.datafile):
            userid, itemid, record, _ = line.split("::")
            data.append((userid, itemid, int(record)))
        return data

    def splitData(self, k, seed, M=9):
        print("训练数据集与测试数据集切分....")
        train, test = {}, {}
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, M) == k:
                test.setdefault(user, {})
                test[user][item] = record
            else:
                train.setdefault(user, {})
                train[user][item] = record
        return train, test

    def ItemSimilarityBest(self):
        print("计算开始物品之间的相似度")
        if os.path.exists('../data/ml-1m/use/item_sim.json'):
            print("物品相似度从文件加载.....")
            itemSim = json.load(open("../data/ml-1m/use/item_sim.json", "r"))
        else:
            itemSim = dict()
            item_user_count = dict()  # 每个物品有多少个用户产生过行为
            count = dict()
            for user, item in self.trainData.items():
                print("user is {}".format(user))
                for i in item.keys():
                    item_user_count.setdefault(i, 0)
                    if self.trainData[str(user)][i] > 0.0:
                        item_user_count[i] += 1
                    for j in item.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if self.trainData[str(user)][i] > 0.0 and self.trainData[str(user)][j] > 0.0 and i != j:
                            count[i][j] += 1

            # 共现矩阵 -> 相似度矩阵
            for i, related_items in count.items():
                itemSim.setdefault(i, dict())
                for j, cuv in related_items.items():
                    itemSim[i].setdefault(j, 0)
                    itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])
            json.dump(itemSim, open("../data/ml-1m/use/item_sim.json", "w"))

        return itemSim

    def recommend(self, user, k=8, nitems=40):
        result = dict()
        u_items = self.trainData.get(user, {})
        for i, pi in u_items.items():
            for j, wj in sorted(self.items_sim[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if j in u_items:
                    continue
                result.setdefault(j, 0)
                result[j] += pi * wj

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    def precision(self, k=8, nitems=10):
        print("开始计算准确率...")
        hit = 0
        precision = 0
        for user in self.testData.keys():
            u_items = self.testData.get(user, {})
            result = self.recommend(user, k=k, nitems=nitems)
            for item, rate in result.items():
                if item in u_items:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)


if __name__ == "__main__":
    ib = ItemCFRec("../data/ml-1m/ratings.dat", [1, 9])
    print("用户1进行推荐的结果如下：{}".format(ib.recommend("1")))
    print("准确率为：{}".format(ib.precision()))
