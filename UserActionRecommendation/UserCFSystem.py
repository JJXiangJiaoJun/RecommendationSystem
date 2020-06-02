import random
import math
import json
import os


class UserCFRec(object):
    def __init__(self, datafile):
        self.datafile = datafile
        self.data = self.loadData()
        self.trainData, self.testData = self.splitData(3, 47)
        self.users_sim = self.UserSimilarityBest()

    # 加载评分数据到data
    def loadData(self):
        print("加载数据...")
        data = []
        for line in open(self.datafile):
            userid, itemid, record, _ = line.split("::")
            data.append((userid, itemid, int(record)))
        return data

    def splitData(self, k, seed, M=8):
        """拆分为训练集和测试集"""
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

    def UserSimilarityBest(self):
        """计算用户之间的相似度，采用惩罚热门商品和优化算法复杂度的方法"""
        print("开始计算用户之间的相似度 .....")
        if os.path.exists('../data/ml-1m/use/user_sim.json'):
            print("用户相似度文件加载 ...")
            userSim = json.load(open('../data/ml-1m/use/user_sim.json', "r"))
        else:
            item_users = dict()
            for u, items in self.trainData.items():
                for i in items.keys():
                    item_users.setdefault(i, set())
                    if self.trainData[u][i] > 0:
                        item_users[i].add(u)

            # 构建倒排表
            count = dict()
            user_item_count = dict()
            for i, users in item_users.items():
                for u in users:
                    user_item_count.setdefault(u, 0)
                    user_item_count[u] += 1
                    count.setdefault(u, {})
                    for v in users:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        count[u][v] += 1 / math.log(1 + len(users))

            userSim = dict()
            for u, related_users in count.items():
                userSim.setdefault(u, {})
                for v, cuv in related_users.items():
                    if u == v:
                        continue
                    userSim[u].setdefault(v, 0.0)
                    userSim[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v])
            json.dump(userSim, open('../data/ml-1m/use/user_sim.json', "w"))
        return userSim

    def recommend(self, user, k=8, nitems=40):
        result = dict()
        have_score_items = self.trainData.get(user, {})
        for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in self.trainData[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] += wuv * rvi
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    def precision(self, k=8, nitems=10):
        print("开始计算准确率")
        hit = 0
        precision = 0
        for user in self.trainData.keys():
            tu = self.testData.get(user, {})
            rank = self.recommend(user, k=k, nitems=nitems)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)


if __name__ == "__main__":
    cf = UserCFRec("../data/ml-1m/ratings.dat")
    result = cf.recommend("1")
    print("user '1' recommend result is {}".format(result))

    precision = cf.precision()
    print("precison is {}".format(precision))
