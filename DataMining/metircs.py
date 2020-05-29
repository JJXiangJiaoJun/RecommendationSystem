from sklearn.metrics import classification_report
from sklearn import metrics

if __name__ == "__main__":
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    target_names = ['class0', 'class1', 'class2']
    print(classification_report(y_true, y_pred, target_names=target_names))

    labels_true = [0, 0, 0, 1, 1, 1]
    labels_pred = [0, 0, 1, 1, 2, 2]

    # 兰德指数
    print(metrics.adjusted_rand_score(labels_true, labels_pred))

    # 互信息
    print(metrics.adjusted_mutual_info_score(labels_true, labels_pred))

    # 同质性
    print(metrics.homogeneity_score(labels_true, labels_pred))

    # 完整性
    print(metrics.completeness_score(labels_true, labels_pred))

    # 同质性与完整性的调和平均
    print(metrics.v_measure_score(labels_true, labels_pred))

    # FMI
    print(metrics.fowlkes_mallows_score(labels_true, labels_pred))
