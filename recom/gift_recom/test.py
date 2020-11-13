from sklearn import model_selection as cv
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from operator import itemgetter

def get_data():
    header = ['uid', 'item', 'count']
    df = pd.read_csv('./dataset/ratings.dat', sep='::', names=header)

    user = df['uid'].unique().tolist()
    item = df['item'].unique().tolist()

    n_users,n_items,num = len(user),len(item), df.shape[0]
    print(n_users,n_items,df.shape[0])
    df['uid'] = df['uid'].apply(lambda x : user.index(x))
    df['item'] = df['item'].apply(lambda x : item.index(x))

    train_data_matrix = np.zeros((n_users,n_items))
    trainset, testset = {}, {}
    for i,line in enumerate(df.values):
        # print(line)
        if i<int(num*0.2):
            train_data_matrix[int(line[0]),int(line[1])] = line[2]
            trainset.setdefault(str(line[0]),{})
            trainset[str(line[0])][str(line[1])]=line[2]
        else:
            testset.setdefault(str(line[0]),{})
            testset[str(line[0])][str(line[1])]=line[2]

    return train_data_matrix,trainset,testset,user,item

def predict(train, similarity, type='user'):
    if type == 'user':
        mean_user_rating = train.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (train - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = train.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def recommend(S, user, train, K, N):
    rank = dict()
    interacted_items = train[user]
    # 找到K个相似的人
    for v, wuv in sorted(S[user].items(), key=itemgetter(1),reverse=True)[0:K]:
        # 查看这K个人喜好的书和用户v喜好的书重合度
        for v_item in train[v]:
            #喜好的的书和v重合，跳过
            if v_item in interacted_items:
                continue
            # 喜好的的书和v不重合，计入推荐，且是前k的不同相似度的叠加
            if v_item not in rank:
                rank[v_item] = 0
            rank[v_item] += wuv * 1.0
    rank_N = dict(sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N])
    return rank_N

def evaluate(S, train, test, K, N):
    hit = 0
    pcount, rcount = 0, 0
    for user in train.keys():
        if user in list(test.keys()):
            tu = test[user]
            rank = recommend(S, user, train, K, N)
            for item, pui in rank.items():
                if item in tu:
                    hit += 1
            pcount += N
            rcount += len(tu)
    return hit / (pcount * 1.0), hit / (rcount * 1.0)

def transform(similar):
    simi={}
    for i,s in enumerate(similar):
        for j,v in enumerate(s):
            if v == 0:
                continue
            simi.setdefault(i,{})
            simi[i][j]=v
    return simi


if __name__ == '__main__':

    train_data_matrix, trainset, testset, user, item = get_data()

    user_similar = pairwise_distances(train_data_matrix, metric='cosine')
    # item_similar = pairwise_distances(train_data_matrix.T, metric='cosine')

    # user_prediction = predict(train, user_similar, type='user')
    # item_prediction = predict(train, item_similar, type='item')

    simi = transform(user_similar)
    evaluate(simi,trainset,testset,10,5)



