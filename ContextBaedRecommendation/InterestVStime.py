import os
import json
import matplotlib.pyplot as plt
import numpy as np


class Demo(object):
    def __init__(self, filePath):
        self.dataPath = filePath
        self.users = ["1086360"]
        self.items = ["80"]

    def showPersonal(self):
        """查看self.user中用户个人兴趣度变化趋势"""
        userItemRate = dict()
        if os.path.exists("data/userItemRate.json"):
            userItemRate = json.load(open("data/userItemRate.json", "r"))
            print("userItemRate Load OK !")
        else:
            for file in os.listdir(self.dataPath):
                onePath = os.path.join(self.dataPath, file)
                print(onePath)
                for line in open(onePath, "r").readlines():
                    if not line.endswith(":") and line.strip().split(",")[0] in self.users:
                        userID, rate, date = line.strip().split(",")
                        userItemRate.setdefault(userID, {})
                        newDate = "".join(date.split("-")[:2])
                        userItemRate[userID].setdefault(newDate, []).append(int(rate))

            # 计算每个月份对应平均分
            for uid in userItemRate.keys():
                for date in userItemRate[uid].keys():
                    userItemRate[uid][date] = round(sum(userItemRate[uid][date]) / len(userItemRate[uid][date]), 2)
            json.dump(userItemRate, open("data/userItemRate.json", "w+"))
            print("userItemRate Message Saved OK !")
        return userItemRate

    def showItem(self):
        """查看self.items中物品流行度趋势"""
        itemUserRate = dict()
        if os.path.exists("data/itemUserRate.json"):
            itemUserRate = json.load(open("data/itemUserRate.json", "r"))
            print("itemUserRate Load OK !")
        else:
            # 遍历文件夹下的每一个文件
            for file in os.listdir(self.dataPath):
                onePath = os.path.join(self.dataPath, file)
                print(onePath)
                flag = False
                for line in open(onePath, "r").readlines():
                    if line.strip().endswith(":") and line.strip().split(":")[0] in self.items:
                        itemID = line.split(":")[0]
                        flag = True
                        continue
                    elif line.strip().endswith(":"):
                        flag = False
                        continue
                    if flag:
                        _, rate, date = line.strip().split(",")
                        itemUserRate.setdefault(itemID, {})
                        newDate = "".join(date.split("-")[:2])
                        itemUserRate[itemID].setdefault(newDate, []).append(int(rate))
                        # 计算每个月份对应的平均分
            for itemId in itemUserRate.keys():
                for data in itemUserRate[itemId].keys():
                    itemUserRate[itemId][data] = round(
                        sum(itemUserRate[itemId][data]) / len(itemUserRate[itemId][data]), 2)
            json.dump(itemUserRate, open("data/itemUserRate.json", "w+"))
            print("itemUserRate Message Saved OK !")

        return itemUserRate

    def showGroup(self):
        groupRate = dict()
        if os.path.exists("data/groupRate.json"):
            groupRate = json.load(open("data/groupRate.json", "r"))
            print("groupRate Load OK !")
        else:
            # 遍历文件夹下的每一个文件
            for file in os.listdir(self.dataPath):
                onePath = os.path.join(self.dataPath, file)
                print(onePath)
                for line in open(onePath, "r").readlines():
                    if not line.strip().endswith(":"):
                        _, rate, date = line.strip().split(",")
                        newDate = "".join(date.split("-")[:2])
                        groupRate.setdefault(newDate, []).append(int(rate))

            # 计算每个月份对应的平均分
            for date in groupRate.keys():
                groupRate[date] = round(sum(groupRate[date]) / len(groupRate[date]), 2)
            json.dump(groupRate, open("data/groupRate.json", "w+"))
            print("groupRate Message Saved OK !")
        return groupRate

    def showPicture(self, _dict, label):
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False
        new_dict = sorted(_dict.items(), key=lambda x: x[0], reverse=False)
        x = [one[0] for one in new_dict]
        y = [one[1] for one in new_dict]
        plt.plot(x, y, marker='o', label=label)
        plt.xticks(np.arange(len(x), step=2), rotation=90)
        plt.xlabel(u"时间-单位/月")
        plt.ylabel(u'平均打分/月')
        plt.title(u'平均评分随时间的变化')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    filePath = "F:\\推荐系统开发实战源码\\数据\\data\\netflix\\training_set"
    demo = Demo(filePath)

    userItemRate = demo.showPersonal()
    print(userItemRate)
    demo.showPicture(userItemRate[demo.users[0]], "uid=1086360")

    itemUserRate = demo.showItem()
    print(itemUserRate)
    demo.showPicture(itemUserRate[demo.items[0]], "itemID=2")

    groupRate = demo.showGroup()
    print(groupRate)
    demo.showPicture(groupRate, None)
