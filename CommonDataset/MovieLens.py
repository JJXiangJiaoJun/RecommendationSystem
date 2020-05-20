import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = {"SimHei"}
plt.rcParams["axes.unicode_minus"] = False


def getRatings(file_path):
    rates = pd.read_table(
        file_path,
        header=None,
        sep="::",
        names=["userID", "movieID", "rate", "timestamp"]
    )
    print("userID的范围是:<{},{}>".format(min(rates["userID"]), max(rates["userID"])))
    print("movieID的范围是:<{},{}>".format(min(rates["movieID"]), max(rates["movieID"])))
    print("评分值的范围为:<{},{}>".format(min(rates["rate"]), max(rates["rate"])))

    print("数据总条数为:\n{}".format(rates.count()))

    print("数据前5条记录为:\n{}".format(rates.head(5)))

    df = rates["userID"].groupby(rates["userID"])
    print("用户评分记录最少条数为:{}".format(df.count().min()))

    scores = rates["rate"].groupby(rates["rate"]).count()
    # scores.plot("bar")
    # plt.figure()
    for x, y in zip(scores.keys(), scores.values):
        plt.text(x, y + 2, "%.0f" % y, ha="center", va="bottom", fontsize=12)

    plt.bar(scores.keys(), scores.values, tick_label=scores.keys(), color=['r', 'g', 'b', 'c', 'm', 'y'])
    plt.xlabel("评分分数")
    plt.ylabel("对应的人数")
    plt.title("评分分数对应的人数统计")
    plt.show()


def getMovies(file_path):
    movies = pd.read_table(file_path,
                           header=None,
                           sep="::",
                           names=["movieID", "title", "genres"])
    print("movieID的范围为:<{},{}>".format(min(movies["movieID"]), max(movies["movieID"])))

    print("数据总条数为:\n{}".format(movies.count()))
    moviesDict = dict()
    for line in movies["genres"].values:
        for one in line.split("|"):
            moviesDict.setdefault(one, 0)
            moviesDict[one] += 1

    print("电影的类型总数为:{}".format(len(moviesDict.keys())))
    print("电影类型分别为:{}".format(moviesDict.keys()))
    print(moviesDict)

    newMD = sorted(moviesDict.items(), key=lambda k: k[1], reverse=True)
    # 设置标签
    labels = [newMD[i][0] for i in range(len(newMD))]
    values = [newMD[i][1] for i in range(len(newMD))]

    # 与label对应，数值越大离中心区越远
    explode = [x * 0.01 for x in range(len(newMD))]

    # 设置X轴与Y轴比例
    plt.axes(aspect=1)

    # labeldistance表示标签离中心距离，pctdistance表示百分百数据离中心区距离
    # autopct 表示百分比格式，shadow表示阴影

    plt.pie(
        x=values,
        labels=labels,
        explode=explode,
        autopct='%3.1f %%',
        shadow=False,
        labeldistance=1.1,
        startangle=0,
        pctdistance=0.8,
        center=(-1, 0)
    )

    # 控制位置：在bbox_to_anchor 数组中，前者控制左右移动，后者控制上下移动
    # ncol控制图例所列的列数，默认为1
    plt.legend(loc=7, bbox_to_anchor=(1.3, 1.0), ncol=3, fancybox=True, shadow=True, fontsize=6)
    plt.show()


def getUsers(file_path):
    users = pd.read_table(
        file_path,
        header=None,
        sep="::",
        names=["userID", "gender", "age", "Occupation", "zip-code"]
    )

    print("userID的范围为:<{}，{}>".format(min(users["userID"]), max(users["userID"])))

    print("数据总条数为:\n{}".format(users.count()))

    usersGender = users["gender"].groupby((users["gender"])).count()
    print("usersGender 类型", type(usersGender))
    print(usersGender)
    plt.axes(aspect=1)
    plt.pie(x=usersGender.values, labels=usersGender.keys(), autopct="%3.1f  %%")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()

    usersAge = users["age"].groupby((users["age"])).count()

    usersGenAge = users.groupby(["gender","age"])['gender'].count()
    print(usersAge)

    print("-------------------------")
    #print(dict(list(usersGenAge)))
    print(usersGenAge)
    plt.plot(
        usersAge.keys(),
        usersAge.values,
        label="用户年龄信息展示",
        linewidth=3,
        color="r",
        marker="o",
        markerfacecolor="blue",
        markersize=12,
    )

    for x, y in zip(usersAge.keys(), usersAge.values):
        plt.text(x, y + 10, "%.0f" % y, ha="center", va="bottom", fontsize=12)
    plt.xlabel("用户年龄")
    plt.ylabel("年龄段对应的人数")
    plt.title("用户年龄段人数统计")
    plt.show()


if __name__ == "__main__":
    # getRatings("F:\\推荐系统开发实战源码\\数据\\data\\ml-1m\\ratings.dat")
    # getMovies("F:\\推荐系统开发实战源码\\数据\\data\\ml-1m\\movies.dat")
    getUsers("F:\\推荐系统开发实战源码\\数据\\data\\ml-1m\\users.dat")
