# coding = utf-8

# 基于用户的协同过滤推荐算法实现
import random
import math
from operator import itemgetter


class UserBasedCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 80
        self.n_rec_movie = 5

        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommneded movie number = %d' % self.n_rec_movie)

    # 读文件得到“用户-电影”数据
    def get_dataset(self, filename):
        trainSet_len = 0
        testSet_len = 0
        random.seed(2000)
        for line in open(filename,'r',encoding='utf-8'):
            user, movie, rating = line.strip().split('::')
            if random.randint(0, 5) == 4:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                testSet_len += 1
            else:
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                trainSet_len += 1
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % trainSet_len)
        print('TestSet = %s' % testSet_len)

    # 计算用户之间的相似度
    def calc_user_sim(self):
        # 构建“电影-用户”倒排索引
        # key = movieID, value = list of userIDs who have seen this movie
        print('Building movie-user table ...')
        movie_user = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                if movie not in movie_user:
                    movie_user[movie] = set()
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        self.movie_count = len(movie_user)
        print('Total movie number = %d' % self.movie_count)

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    # User - CF
                    # self.user_sim_matrix[u][v] += 1
                    # User - IIF
                    self.user_sim_matrix[u][v] += 1/math.log(1+len(users))
        print('Build user co-rated movies matrix success!')

        # 计算相似性
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')

    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]

        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            for v_item in self.trainSet[v]:
                if v_item in watched_movies:
                    continue
                if v_item not in rank:
                    rank[v_item] = 0
                rank[v_item] += wuv * 1.0

        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]


    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        N = self.n_rec_movie
        # 准确率和召回率
        hit = 0
        rec_count = 0
        pre_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user, in enumerate(self.trainSet):
            if user in self.testSet:
                test_movies = self.testSet.get(user, {})
                rec_movies = self.recommend(user)
                for movie, w in rec_movies:
                    if movie in test_movies:
                        hit += 1
                    all_rec_movies.add(movie)
                rec_count += len(test_movies)
                pre_count += N
            else:
                continue
        precision = hit / (1.0 * pre_count)
        recall = hit / (1.0 * rec_count)
        f1 = 2*(precision * recall)/(precision + recall)

        print('precisioin=%.4f\trecall=%.4f\tf1=%.4f' % (precision, recall, f1))


if __name__ == '__main__':
    rating_file = './dataset/ratings.dat'
    userCF = UserBasedCF()
    userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    userCF.evaluate()
