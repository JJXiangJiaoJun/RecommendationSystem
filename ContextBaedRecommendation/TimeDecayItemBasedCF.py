import math
import json
import os
import random
from sklearn.model_selection import train_test_split


class ItemBasedCF(object):
    def __init__(self, datafile):
        self.alpha = 0.5
        self.beta = 0.8
        self.datafile = datafile

        self.train, self.test, self.max_data = self.loadData()

        self.items_sim = self.ItemSimilarityBest()

    def loadData(self):
        print("Start load Data and Split data ....")
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

    def ItemSimilarityBest(self):
        print("Start calculate item's similarity")
        if os.path.exists("data/item_sim.json"):
            print("load from file ...")
            itemSim = json.load(open("data/item_sim.json", "r"))
        else:
            itemSim = dict()
            item_eval_by_user_count = dict()
            count = dict()
            for user, items in self.train.items():
                for i in items.keys():
                    item_eval_by_user_count.setdefault(i, 0)
                    if self.train[str(user)][i]["rate"] > 0.0:
                        item_eval_by_user_count[i] += 1
                    for j in items.keys():
                        count.setdefault(i, {}).setdefault(j, 0)
                        if self.train[str(user)][i]["rate"] > 0.0 and self.train[str(user)][j]["rate"] > 0.0 and i != j:
                            count[i][j] += 1 * 1 / (
                                1 + self.alpha * abs(self.train[user][i]["time"] - self.train[user][i]["time"]) / (
                                    24 * 60 * 60))

            for i, relatead_items in count.items():
                itemSim.setdefault(i, {})
                for j, num in relatead_items.items():
                    itemSim[i].setdefault(j, 0.0)
                    itemSim[i][j] = num / math.sqrt(item_eval_by_user_count[i] * item_eval_by_user_count[j])
            json.dump(itemSim, open("data/item_sim.json", "w+"))
        return itemSim

    def recommend(self, user, k=8, nitems=40):
        result = dict()
        u_items = self.train.get(user, {})
        for i, rate_time in u_items.items():
            for j, wj in sorted(self.items_sim[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if j in u_items:
                    continue
                result.setdefault(j, 0)
                result[j] += rate_time["rate"] * wj * 1 / (1 + self.beta * (self.max_data - abs(rate_time["time"])))

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])


    def precision(self,k=8,nitems=10):
        hit = 0
        precision = 0
        print(len(self.test.keys()))
        for user in random.sample( self.test.keys(), 10):
            print(user)
            u_items = self.test.get(user, {})
            result = self.recommend(user, k=k, nitems=nitems)
            for item, rate in result.items():
                if item in u_items:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)

if __name__ == "__main__":
    ib = ItemBasedCF("../data/ml-1m/ratings.dat")
    result = ib.recommend("1")
    print("user '1' recommend result is {} ".format(result))

    precision = ib.precision()
    print("precision is {}".format(precision))