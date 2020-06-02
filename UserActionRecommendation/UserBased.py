import math


class UserCF(object):
    def __init__(self):
        self.user_score_dict = self.initUserScore()
        self.users_sim = self.userSimilarityBest()

    def initUserScore(self):
        """初始化用户评分数据"""
        user_score_dict = {"A": {"a": 3.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 0.0},
                           "B": {"a": 4.0, "b": 0.0, "c": 4.5, "d": 0.0, "e": 3.5},
                           "C": {"a": 0.0, "b": 3.5, "c": 0.0, "d": 0., "e": 3.0},
                           "D": {"a": 0.0, "b": 4.0, "c": 0.0, "d": 3.50, "e": 3.0}}
        return user_score_dict

    def userSimilarity(self):
        W = dict()
        for u in self.user_score_dict.keys():
            W.setdefault(u, {})
            for v in self.user_score_dict.keys():
                if u == v:
                    continue
                u_set = set([key for key in self.user_score_dict[u].keys() if
                             self.user_score_dict[u][key] > 0])
                v_set = set([key for key in self.user_score_dict[v].keys() if
                             self.user_score_dict[v][key] > 0])
                W[u][v] = float(len(u_set & v_set)) / math.sqrt(len(u_set) * len(v_set))

        return W

    def preUserItemScore(self, userA, item):
        """预测用户对item评分"""
        score = 0.0
        for user in self.users_sim[userA].keys():
            if user != userA:
                score += self.users_sim[userA][user] * self.user_score_dict[user][item]

        return score

    def recommend(self, userA):
        """为用户推荐物品"""
        user_item_score_dict = dict()
        for item in self.user_score_dict[userA].keys():
            if self.user_score_dict[userA][item] <= 0:
                user_item_score_dict[item] = self.preUserItemScore(userA, item)

        return user_item_score_dict

    def userSimilarityBetter(self):
        item_users = dict()
        for u, items in self.user_score_dict.items():
            for i in items.keys():
                item_users.setdefault(i, set())
                if self.user_score_dict[u][i] > 0:
                    item_users[i].add(u)

        # 建立倒排表
        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    C[u].setdefault(v, 0)
                    if u == v:
                        continue
                    C[u][v] += 1
        print(C)
        print(N)

        # 构建相似度矩阵
        W = dict()
        for u, related_users in C.items():
            W.setdefault(u, {})
            for v, cuv in related_users.items():
                if u == v:
                    continue
                W[u].setdefault(v, 0.0)
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W

    def userSimilarityBest(self):
        """加上惩罚"""
        item_users = dict()
        for u, items in self.user_score_dict.items():
            for i in items.keys():
                item_users.setdefault(i, set())
                if self.user_score_dict[u][i] > 0:
                    item_users[i].add(u)

        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                N.setdefault(u, 0)
                N[u] += 1
                C.setdefault(u, {})
                for v in users:
                    C[u].setdefault(v, 0)
                    if u == v:
                        continue
                    C[u][v] += 1 / math.sqrt(1 + len(users))
        W = dict()
        for u, related_users in C.items():
            W.setdefault(u, {})
            for v, cuv in related_users.items():
                if u == v:
                    continue
                W[u].setdefault(v, 0.0)
                W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return W


if __name__ == "__main__":
    ub = UserCF()
    print(ub.recommend("C"))
