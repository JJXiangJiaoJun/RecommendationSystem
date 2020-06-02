import json

import pandas  as pd
from sklearn import preprocessing
import numpy as np
import math
import random


# 对 iPhone的离散属性建模（颜色和内存）进行one-hot编码
def test():
    onehot = preprocessing.OneHotEncoder()
    onehot.fit([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
    print(onehot.transform([[0, 1]]).toarray())

    def MaxMinNormalization(x, Max, Min):
        x = (x - Min) / (Max - Min)
        return x

    sizes = [4, 4.7, 5.5]

    # prices
    prices = [1358, 2788, 3656]
    size_min, size_max = min(sizes), max(sizes)
    price_min, price_max = min(prices), max(prices)

    nor_size = []
    for size in sizes:
        nor_size.append(round(MaxMinNormalization(size, size_max, size_min), 4))
    print('尺寸归一化为：%s' % nor_size)

    nor_price = []
    for price in prices:
        nor_price.append(MaxMinNormalization(price, price_max, price_min))

    print("价格归一化为:%s" % nor_price)


class DataProcessing(object):
    def __init__(self):
        pass

    def process(self):
        print("开始转换用户数据(users.dat)...")
        self.process_user_data()
        print("开始转换电影数据(movies.dat)...")
        self.process_movies_data()
        print("开始转换用户对电影评分数据(rating.dat)...")
        self.process_rating_data()
        print('Over!')

    def process_user_data(self, file='../data/ml-1m/users.dat'):
        fp = pd.read_table(file, sep='::', engine='python',
                           names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        fp.to_csv('../data/ml-1m/use/users.csv', index=False)

    def process_rating_data(self, file='../data/ml-1m/ratings.dat'):
        fp = pd.read_table(file, sep='::', engine='python',
                           names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        fp.to_csv('../data/ml-1m/use/rating.csv', index=False)

    def process_movies_data(self, file='../data/ml-1m/movies.dat'):
        fp = pd.read_table(file, sep='::', engine='python',
                           names=['MovieID', 'Title', 'Genres'])
        fp.to_csv('../data/ml-1m/use/movies.csv', index=False)

    # 计算电影的特征信息矩阵
    def prepare_item_profile(self, file='../data/ml-1m/use/movies.csv'):
        items = pd.read_csv(file)
        item_ids = set(items['MovieID'].values)
        self.item_dict = {}
        genres_all = list()
        # 将每个电影的类型放在item_dict中
        for item in item_ids:
            genres = items[items["MovieID"] == item]['Genres'].values[0].split("|")
            self.item_dict.setdefault(item, []).extend(genres)
            genres_all.extend(genres)
        self.genres_all = set(genres_all)

        # 将每个电影的特征信息矩阵存放在 self.item_matrix中
        # 保存dict时 ,key只能为str,所以这里对item id做str()转换
        self.item_matrix = {}
        for item in self.item_dict.keys():
            self.item_matrix[str(item)] = [0] * len(set(self.genres_all))
            for genre in self.item_dict[item]:
                index = list(set(genres_all)).index(genre)
                self.item_matrix[str(item)][index] = 1
        json.dump(self.item_matrix,
                  open('../data/ml-1m/use/item_profile.json', 'w'))
        print('item 信息计算完成,保存路径为:{}'
              .format('../data/ml-1m/use/user_profile.json'))

    # 计算用户的偏好矩阵
    def prepare_user_profile(self, file='../data/ml-1m/use/rating.csv'):
        users = pd.read_csv(file)
        user_ids = set(users['UserID'].values)
        # 将user信息转换为dict
        users_rating_dict = {}
        for user in user_ids:
            users_rating_dict.setdefault(str(user), {})
        with open(file, 'r') as fr:
            for line in fr.readlines():
                if not line.startswith("UserID"):
                    (user, item, rate) = line.split(",")[:3]
                    users_rating_dict[user][item] = int(rate)
        # 获取用户对每个类型下都有哪些电影进行了评分
        self.user_matrix = {}
        # 遍历每个用户
        for user in users_rating_dict.keys():
            print('user is {}'.format(user))
            score_list = users_rating_dict[user].values()
            # 用户的平均打分
            avg = sum(score_list) / len(score_list)
            self.user_matrix[user] = []

            # 遍历每个类型（保证item_profile和user_profile信息中矩阵每列表示的类型一致）
            for genre in self.genres_all:
                score_all = 0.0
                score_len = 0
                # 遍历每个item
                for item in users_rating_dict[user].keys():
                    if genre in self.item_dict[int(item)]:
                        score_all += (users_rating_dict[user][item] - avg)
                        score_len += 1
                if score_len == 0:
                    self.user_matrix[user].append(0.0)
                else:
                    self.user_matrix[user].append(score_all / score_len)

        json.dump(self.user_matrix,
                  open("../data/ml-1m/use/user_profile.json", "w"))
        print("user 信息计算完成：保存路径为:{}"
              .format("../data/ml-1m/use/user_profile.json"))


class CBRecoomend(object):
    def __init__(self, K):
        self.K = K
        self.item_profile = json.load(open("../data/ml-1m/use/item_profile.json", "r"))
        self.user_profile = json.load(open("../data/ml-1m/use/user_profile.json", "r"))

    def get_none_score_item(self, user):
        """获取用户未进行评分的item列表"""
        items = pd.read_csv("../data/ml-1m/use/movies.csv")["MovieID"].values
        data = pd.read_csv("../data/ml-1m/use/rating.csv")
        have_score_items = data[data["UserID"] == user]["MovieID"].values
        none_score_items = set(items) - set(have_score_items)
        return none_score_items

    def cosUI(self, user, item):
        """cosine similarity"""
        Uia = sum(
            np.array(self.user_profile[str(user)])
            *
            np.array(self.item_profile[str(item)])
        )
        Ua = math.sqrt(sum([math.pow(one, 2) for one in self.user_profile[str(user)]]))
        Ia = math.sqrt(sum([math.pow(one, 2) for one in self.item_profile[str(item)]]))

        return Uia / (Ua * Ia)

    def recommend(self, user):
        """为用户进行电影推荐"""
        user_result = {}
        item_list = self.get_none_score_item(user)
        for item in item_list:
            user_result[item] = self.cosUI(user, item)
        if self.K is None:
            result = sorted(
                user_result.items(), key=lambda k: k[k], reverse=True
            )
        else:
            result = sorted(
                user_result.items(), key=lambda k: k[1], reverse=True
            )[:self.K]
        print(result)

    def evaluate(self):
        """推荐系统效果评估"""
        evas = []
        data = pd.read_csv("../data/ml-1m/use/rating.csv")
        # 随机选取20个用户进行效果评估
        for user in random.sample([one for one in range(1, 6040)], 20):
            have_score_items = data[data["UserID"] == user]["MovieID"].values
            items = pd.read_csv("../data/ml-1m/use/movies.csv")["MovieID"].values

            user_result = {}
            for item in items:
                user_result[item] = self.cosUI(user, item)
            results = sorted(
                user_result.items(), key=lambda k: k[1], reverse=True
            )[:len(have_score_items)]
            rec_items = []
            for one in results:
                rec_items.append(one[0])
            eva = len(set(rec_items) & set(have_score_items)) / len(have_score_items)
            evas.append(eva)

        return sum(evas) / len(evas)


if __name__ == "__main__":
    # dp = DataProcessing()
    # dp.process()
    # dp.prepare_item_profile()
    # dp.prepare_user_profile()
    cb = CBRecoomend(K=10)
    cb.recommend(1)
    print(cb.evaluate())
