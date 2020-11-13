from sklearn.metrics import mean_squared_error
from sklearn import model_selection as cv
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating']
df = pd.read_csv('./dataset/ratings.dat', sep='::', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

train_data, test_data = cv.train_test_split(df, test_size=0.2)
train_data_matrix = np.zeros((n_users, n_items))

print(train_data_matrix.shape)
for line in train_data.itertuples():
    print(line)
    train_data_matrix[int(line[0])-1, int(line[1])-1] = float(line[2])
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[int(line[0])-1, int(line[1])-1] = float(line[2])

# 你可以使用 sklearn 的pairwise_distances函数来计算余弦相似性。注意，因为评价都为正值输出取值应为0到1.
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
#矩阵的转置实现主题的相似度
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

#有许多的评价指标，但是用于评估预测精度最流行的指标之一是Root Mean Squared Error (RMSE)。

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()#nonzero(a)返回数组a中值不为零的元素的下标,相当于对稀疏矩阵进行提取
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

