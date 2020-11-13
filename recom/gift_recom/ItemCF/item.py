from utility import data_process
from itemCF.tool import item_similarity, user_item_score
from utility.common import topk

if __name__ == '__main__':

    file_path = '../dataset/ratings.dat'
    n_user, n_item, train_data, test_data, topk_data = data_process.pack_data(data_process.read_ml(file_path))

    W = item_similarity(train_data, n_user, n_item)
    for N in (10, 20, 80, 160):
        scores = user_item_score(train_data, n_user, n_item, W, N)
        score_fn = lambda ui: [scores[u][i] for u, i in zip(ui['user_id'], ui['item_id'])]
        topk(topk_data, score_fn)
