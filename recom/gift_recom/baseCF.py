import random
import math
import operator

def ReadData():
    data = dict()
    with open('./dataset/ratings.dat') as file_object:
        for line in file_object:
            line = line.strip('\n')
            user_data = line.split('::')
            user = user_data.pop(0)
            item = user_data.pop(0)
            if user not in data:
                data[user] = []
            data[user].append(item)
    return data

# 将数据集分为 训练集 与测试集，相当于把用户的喜好点击分为trian、test
# train中记录的是用户的部分喜好，test中记录的也是用户的部分喜好
def SplitData(data, M,):
    test = dict()
    train = dict()
    random.seed(2000)
    for user, items in data.items():
        for item in items:
            if random.randint(0, M) == 4:
                if user not in test:
                    test[user] = []
                test[user].append(item)
            else:
                if user not in train:
                    train[user] = []
                train[user].append(item)
    return train, test

# 基础版
def UserSimilarity1(train):
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            a, b = [int(i) for i in train[u]], [int(i) for i in train[v]]
            W.setdefault(u, {})
            W[u][v] = len([i for i in a if i in b])
            W[u][v] /= math.sqrt(len(a) * len(b) * 1.0)
    return W

# -- 进阶版 余弦相似度的计算
def UserSimilarity2(train):
    item_uesrs = dict()
    for u, items in train.items():
        for i in items:
            if i not in item_uesrs:
                item_uesrs[i] = set()
            item_uesrs[i].add(u)
    # 计算用户间 的相同物品
    C, N = dict(), dict()
    for item, users in item_uesrs.items():
        for u in users:
            if u not in N:
                N[u] = 0
            N[u] += 1
            if u not in C:
                C[u] = dict()
            for v in users:
                if u == v:
                    continue
                else:
                    if v not in C[u]:
                        C[u][v] = 0
                    C[u][v] += 1
    # 计算余弦相似度 矩阵 W
    W = dict()
    for u, related_users in C.items():
        W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def Recommend(user, train, W ,K):
    rank = dict()
    interacted_items = train[user]
    # 找到K个相似的人
    for v, wuv in sorted(W[user].items(), key=operator.itemgetter(1),reverse=True)[0:K]:
        # 查看这K个人喜好的书和用户v喜好的书重合度
        for v_item in train[v]:
            #喜好的的书和v重合，跳过
            if v_item in interacted_items:
                continue
            # 喜好的的书和v不重合，计入推荐，且是前k的不同相似度的叠加
            if v_item not in rank:
                rank[v_item] = 0
            rank[v_item] += wuv * 1.0
    return rank

def GetRecommendation(user, K, N):
    rank = Recommend(user, train, W, K)
    rank_N = dict(sorted(rank.items(), key=operator.itemgetter(1),reverse=True)[0:N])
    return rank_N

def evaluate(train, test, K, N):
    hit = 0
    pcount, rcount = 0, 0
    for user in train.keys():
        if user in list(test.keys()):
            tu = test[user]
            rank = GetRecommendation(user, K, N)
            for item, pui in rank.items():
                if item in tu:
                    hit += 1
            pcount += N
            rcount += len(tu)
    return hit / (pcount * 1.0), hit / (rcount * 1.0)


if __name__ == '__main__':

    split_redio, topK, topN = 5, 80, 5
    data = ReadData()
    train, test = SplitData(data, split_redio)

    W = UserSimilarity2(train)

    p, r = evaluate(train, test, topK, topN)
    f1 = 2 * p * r / (p + r)

    print('准确率为：', p)
    print('召回率为：', r)
    print('f1:', f1)


