from numpy import *


def EuclideanDistance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


print("a,b 二维欧氏距离为：", EuclideanDistance((1, 1), (2, 2)))


def ManhattanDistance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


print("a,b二维曼哈顿距离为：", ManhattanDistance((1, 1), (2, 2)))


def ChebyshevDistance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


print("a,b二维切比雪夫距离：", ChebyshevDistance((1, 2), (3, 4)))
