# coding=utf-8

import math
import json
import os
import random
from sklearn.model_selection import train_test_split


class NewUserCF(object):
    def __init__(self, datafile):
        self.alpha = 0.5
        self.beta = 0.8
        self.datafile = datafile
        self.train, self.test, self.max_data = self.loadData()
        self.users_sim = self.UserSimilarityBest()

    def loadData(self):
        """加载数据集，并分拆成训练集和测试集"""
        print("Start laod Data and Split data ....")
        data = list()
        max_data = 0
        for line in open(self.datafile):
            userid, itemid, record, timestamp = line.split("::")
            data.append((userid, itemid, int(record), int(timestamp)))
            if int(timestamp) > max_data:
                max_data = int(timestamp)

        train_list, test_list = train_test_split(data, test_size=0.1, random_state=40)

        train_dict = self.transform(train_list)
        test_dict = self.transform(test_list)
        return train_dict, test_dict, max_data

    def transform(self, data):
        data_dict = dict()
        for user, item, record, timestamp in data:
            data_dict.setdefault(user, {}).setdefault(item, {})
            data_dict[user][item]["rate"] = record
            data_dict[user][item]["time"] = timestamp
        return data_dict

    def UserSimilarityBest(self):
        print("Start calculation user's similarity.......")
        if os.path.exists("data/user_sim.json"):
            print("从文件加载......")
            userSim = json.load(open("data/user_sim.json", "r"))
        else:
            item_eval_by_users = dict()
            # 倒排
            for u, items in self.train.items():
                for i in items.keys():
                    item_eval_by_users.setdefault(i, set())
                    if self.train[u][i]["rate"] > 0:
                        item_eval_by_users[i].add(u)
            # 建立倒排表
            count = dict()
            # 用户评价过多少个item
            user_eval_item_count = dict()
            for i, users in item_eval_by_users.items():
                for u in users:
                    user_eval_item_count.setdefault(u, 0)
                    user_eval_item_count[u] += 1
                    count.setdefault(u, {})
                    for v in users:
                        count[u].setdefault(v, 0)
                        if u == v:
                            continue
                        count[u][v] += 1 / (
                            1 + self.alpha * abs(self.train[u][i]['time'] - self.train[v][i]['time']) / (
                                24 * 60 * 60) * 1 / math.log(1 + len(users)))
            userSim = dict()
            for u, related_users in count.items():
                userSim.setdefault(u, {})
                for v, cuv in related_users.items():
                    if u == v:
                        continue
                    userSim[u].setdefault(v, 0.0)
                    userSim[u][v] = cuv / math.sqrt(user_eval_item_count[u] * user_eval_item_count[v])
            json.dump(userSim, open('data/user_sim.json', "w+"))
        return userSim

    def recommend(self, user, k=8, nitems=40):
        rank = dict()
        interacted_items = self.train.get(user, {})
        for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            # 对于相似用户的每个评价过的物品
            for i, rv in self.train[v].items():
                if i in interacted_items:
                    continue
                rank.setdefault(i, 0)
                # 累加相似度 i 为物品id
                rank[i] += wuv * rv["rate"] * 1 / (1 + self.beta * (self.max_data - abs(rv["time"])))

        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    def precision(self, k=8, nitems=10):
        hit = 0
        precision = 0
        for user in random.sample(self.train.keys(), 10):
            tu = self.test.get(user, {})
            rank = self.recommend(user, k=k, nitems=nitems)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += nitems
        return hit / ((precision + 1) * 1.0)


if __name__ == "__main__":
    cf = NewUserCF("../data/ml-1m/ratings.dat")
    result = cf.recommend("1")
    print("user '1' recommend result is {} ".format(result))

    precision = cf.precision()
    print("precision is {} ".format(precision))
