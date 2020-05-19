# -*- coding:utf-8 -*-
import os
import random
import json
import math
import shutil


class FirstRec(object):
    """从Netflix电影数据集中随机选择1000个用户进行推荐系统的相似度计算
    """

    def __init__(self, file_path, seed, k, n_items):
        """初始化函数
        Parameters
        ----------
        file_path : str
            源文件的路径
        seed : int
            随机数粽子
        k : int
            KNN 中的超参数k
        nitems : int
            为每个用户推荐top_nitems 个电影
        """
        self.file_path = file_path
        self.seed = seed
        self.users_1000 = self.__select_1000_users()
        self.k = k
        self.n_items = n_items
        self.train, self.test = self._load_and_split_data()

    def __select_1000_users(self):
        """获取所有用户并且随机选择1000个
        """
        print("随机选取1000个用户！")
        if os.path.exists("data/train.json") and os.path.exists("data/test.json"):
            print("已存在 train.json 和 test.json")
            return list()
        else:
            users = set()
            for file in os.listdir(self.file_path):
                # one_path = "{}/{}".format(self.file_path,file)
                one_path = os.path.join(self.file_path, file)
                print("{}".format(one_path))
                with open(one_path, 'r') as f:
                    for line in f.readlines():
                        if line.strip().endswith(":"):
                            continue
                        userID, _, _ = line.split(",")
                        users.add(userID)

            users_1000 = random.sample(list(users), 1000)

            print(users_1000)
            return users_1000

    def _load_and_split_data(self):
        """加载数据并且拆分为训练集和测试集 
        """
        train = dict()
        test = dict()
        if os.path.exists("data/train.json") and os.path.exists("data/test.json"):
            print("从文件中加载训练集和测试集")
            train = json.load(open("data/train.json"))
            test = json.load(open('data/test.json'))
            print("从文件中加载数据完成")
        else:
            # 设置产生随机数的种子，保证每次实验产生的随机结果一致
            random.seed(self.seed)
            for file in os.listdir(self.file_path):
                one_path = os.path.join(self.file_path, file)
                print("{}".format(one_path))
                with open(one_path, 'r') as f:
                    movieID = f.readline().split(":")[0]
                    for line in f.readlines():
                        if line.endswith(":"):
                            continue
                        userID, rate, _ = line.split(",")

                        if userID in self.users_1000:
                            if random.randint(1, 50) == 1:
                                test.setdefault(userID, {})[movieID] = int(rate)
                            else:
                                train.setdefault(userID, {})[movieID] = int(rate)
            print("导出数据到 data/train.json 和 data/test.json")
            json.dump(train, open("data/train.json", "w+"))
            json.dump(test, open("data/test.json", "w+"))
            print("导出数据完成")
        return train, test

    def pearson(self, rating1, rating2):
        """计算皮尔逊相关系数
        Parameters
        ----------
        rating1 : dict
            第一个用户对电影打分的字典
        rating2 : dict
            第二个用户对电影打分的字典
        """
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        num = 0
        for key in rating1.keys():
            if key in rating2.keys():
                num += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += math.pow(x, 2)
                sum_y2 += math.pow(y, 2)
        if num == 0:
            return 0
        # pearson 相关系数分母
        denominator = math.sqrt(sum_x2 - math.pow(sum_x, 2) / num) * math.sqrt(sum_y2 - math.pow(sum_y, 2) / num)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / num) / denominator

    def recommend(self, userID):
        """根据用户ID进行电影推荐
        Parameters
        ----------
        userID : str
            用户ID
        """
        neighborUser = dict()
        for user in self.train.keys():
            if userID != user:
                distance = self.pearson(self.train[userID], self.train[user])
                neighborUser[user] = distance
        newNU = sorted(neighborUser.items(), key=lambda k: k[1], reverse=True)

        movies = dict()
        for (sim_user, sim) in newNU:
            for movieID in self.train[sim_user].keys():
                movies.setdefault(movieID, [0, 0])
                movies[movieID][1] += 1
                movies[movieID][0] += sim * self.train[sim_user][movieID]
        newMovies = sorted(movies.items(), key=lambda k: k[1][0], reverse=True)
        return newMovies

    def evaluate(self, num=30):
        """
        Parameters
        ----------
        num : int. default 30
            随机选取num个人进行测试
        """
        print("评估推荐系统模型")
        precisions = list()
        random.seed(10)
        for userID in random.sample(self.test.keys(), num):
            hit = 0
            result = self.recommend(userID)
            for movie, _ in result[:self.n_items]:
                if movie in self.test[userID].keys():
                    hit += 1
            precisions.append(hit / self.n_items)
        return sum(precisions) / num


if __name__ == "__main__":
    print('this is netflix prize script')
    file_path = "../data/netflix/training_set"
    seed = 30
    k = 5
    n_items = 50
    f_rec = FirstRec(file_path, seed, k, n_items)

    print("{} 和 {} 的相关系数为 {}".format(195100, 1547579, f_rec.pearson(f_rec.train["195100"], f_rec.train["1547579"])))

    print("给用户{}推荐电影".format(195100))
    for moviesID, cnt_and_score in f_rec.recommend("195100"):
        print("电影名称:{}  观看总次数:{} 推荐分数:{} 平均评分为:{}".format(moviesID, cnt_and_score[1], cnt_and_score[0],
                                                          cnt_and_score[0] / cnt_and_score[1]))

    print("推荐准确率为:{:.4f}".format(f_rec.evaluate()))
