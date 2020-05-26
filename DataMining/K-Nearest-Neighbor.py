import numpy as np
import collections


class KNN(object):
    def __init__(self, k):
        self.K = k

    # 准备数据
    def createData(self):
        features = np.array([[180, 76], [158, 43], [176, 78], [161, 49]])
        labels = ["男", "女", "男", "女"]
        return features, labels

    def Normalization(self, data):
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        new_data = (data - mins) / (maxs - mins)
        return new_data, maxs, mins

    def classify(self, one, data, labels):
        differenceData = data - one
        squareData = (differenceData ** 2).sum(axis=1)
        distance = np.sqrt(squareData)
        sortDistanceIndex = np.argsort(distance, axis=-1)

        # print(sortDistanceIndex)
        sorted_labels = labels[sortDistanceIndex]
        topK_labels = sorted_labels[:self.K]
        predict_label = collections.Counter(topK_labels).most_common(1)

        return predict_label[0][0]


if __name__ == "__main__":
    knn = KNN(3)

    features, labels = knn.createData()

    new_data, maxs, mins = knn.Normalization(features)

    one = np.array([176, 76])

    new_one = (one - mins) / (maxs - mins)

    result = knn.classify(new_one, new_data, np.array(labels))
    print("数据 {} 预测的性别为 :{}".format(one, result))
