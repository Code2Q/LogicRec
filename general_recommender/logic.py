# import numpy as np
# import scipy.sparse as sp
# import torch

# from recbole.model.abstract_recommender import GeneralRecommender
# from recbole.utils import InputType, ModelType
# from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels

# class Logic(GeneralRecommender):
#     r"""ItemKNN is a basic model that compute item similarity with the interaction matrix."""

#     input_type = InputType.POINTWISE
#     type = ModelType.TRADITIONAL

#     def __init__(self, config, dataset):
#         super(Logic, self).__init__(config, dataset)

#         # load parameters info
#         self.max_branch = config["max_branch"]
#         self.logic_step = config['logic_step']
#         self.shrink = config["shrink"] if "shrink" in config else 0.0
#         self.item_tag_id = 'item_tag_id'
#         self.user_tag_id = 'user_tag_id'
#         self.item_tag_id_list = 'item_tag_id_list'
#         self.user_tag_id_list = 'user_tag_id_list'
#         self.item_feat = dataset.get_item_feature().to(self.device)
#         print(f"====item feat {self.item_feat}")


#         self.utid2itid = dataset.utid2itid_feat
#         # self.iid2itid = dataset.iid2itid_feat
#         # self.iid2utid = dataset.iid2utid_feat
#         self.itid2utid = dataset.itid2utid_feat
#         self.item_tag_num = self.itid2utid['item_tag_id'].shape #item tag num: torch.Size([490132]
#         self.max_item_tag_id = self.itid2utid['item_tag_id'].max().item()
#         self.min_item_tag_id = self.itid2utid['item_tag_id'].min().item()
#         self.max_item_tag = self.max_item_tag_id - self.min_item_tag_id + 1

#         self.history_item_matrix, _, _ = dataset.history_item_matrix()  # 用户历史交互物品


#         # 构建稀疏矩阵
#         self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
#         self.user_history_sparse = self.interaction_matrix

#         self.item_tag_sparse = self._build_sparse_item_tag()


#         self.fake_loss = torch.nn.Parameter(torch.zeros(1))
#         self.other_parameter_name = ["utid2itid", 'itid2utid']


#     def _build_sparse_item_tag(self):
#         rows = []
#         cols = []
#         data = []

#         for item_id, tags in enumerate(self.item_feat['item_tag_id_list']):
#             for tag in tags:
#                 rows.append(self.item_feat['item_id'][item_id].cpu().item())
#                 cols.append(tag.cpu().item())
#                 data.append(1)
#         print(self.n_items, self.max_item_tag)

#         return sp.csr_matrix((data, (rows, cols)), shape=(self.n_items, self.max_item_tag))

#     def forward(self, user, item):
#         pass

#     def calculate_loss(self, interaction):
#         return torch.nn.Parameter(torch.zeros(1))

#     def _calculate_item_score(self, user_id):
#         user_histories = self.user_history_sparse[user_id]
#         propagated_tags = user_histories @ self.item_tag_sparse
#         propagated_tag_scores = propagated_tags.multiply(1 / (propagated_tags.sum(axis=1) + 1e-9))

#         row_sums_inv = 1 / (self.item_tag_sparse.sum(axis=1) + 1e-9)
#         item_tag_norm = self.item_tag_sparse.multiply(row_sums_inv)

#         # 将稀疏矩阵转换为密集矩阵
#         # propagated_tag_scores_dense = propagated_tag_scores.toarray()
#         # item_tag_norm_dense = item_tag_norm.toarray()
#         # 确保稀疏矩阵是 csr 格式
#         propagated_tag_scores = propagated_tag_scores.tocsr()
#         item_tag_norm = item_tag_norm.tocsr()
#         # print(f"====== {type(propagated_tag_scores), type(item_tag_norm)}")

#         # 计算相似度矩阵
#         all_scores = pairwise_kernels(propagated_tag_scores, item_tag_norm, metric='cosine', n_jobs=-1)
#         # all_scores = cosine_similarity(propagated_tag_scores_dense, item_tag_norm_dense)

#         return all_scores

#     def predict(self, interaction):
#         user = interaction[self.USER_ID].cpu().numpy().astype(int)
#         item = interaction[self.ITEM_ID].cpu().numpy().astype(int)

#         all_scores = self._calculate_item_score(user)

#         scores = all_scores[np.arange(len(user)), item]
#         scores = torch.tensor(scores).to(self.device)

#         return scores

#     def full_sort_predict(self, interaction):
#         user = interaction[self.USER_ID].cpu().numpy().astype(int)

#         all_scores = self._calculate_item_score(user)
#         all_scores = torch.tensor(all_scores).to(self.device)

#         return all_scores

import numpy as np
import scipy.sparse as sp
import torch

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType

class Logic(GeneralRecommender):
    r"""ItemKNN is a basic model that compute item similarity with the interaction matrix."""

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(Logic, self).__init__(config, dataset)

        # load parameters info
        self.max_branch = config["max_branch"]
        self.logic_step = config['logic_step']
        self.shrink = config["shrink"] if "shrink" in config else 0.0
        self.item_tag_id = 'item_tag_id'
        self.user_tag_id = 'user_tag_id'
        self.item_tag_id_list = 'item_tag_id_list'
        self.user_tag_id_list = 'user_tag_id_list'
        self.item_feat = dataset.get_item_feature().to(self.device)
        print(f"====item feat {self.item_feat}")


        self.utid2itid = dataset.utid2itid_feat
        # self.iid2itid = dataset.iid2itid_feat
        # self.iid2utid = dataset.iid2utid_feat
        self.itid2utid = dataset.itid2utid_feat
        self.item_tag_num = self.itid2utid['item_tag_id'].shape #item tag num: torch.Size([490132]
        self.max_item_tag_id = self.itid2utid['item_tag_id'].max().item()
        self.min_item_tag_id = self.itid2utid['item_tag_id'].min().item()
        self.max_item_tag = self.max_item_tag_id - self.min_item_tag_id + 1

        self.history_item_matrix, _, _ = dataset.history_item_matrix()  # 用户历史交互物品
        # 预处理用户历史记录
        self.user_history_dict = {}
        for user_id in range(self.n_users):
            user_history = self.history_item_matrix[user_id].cpu().numpy().astype(int)
            self.user_history_dict[user_id] = user_history

        # 构建稀疏矩阵
        self.user_history_sparse = self._build_sparse_user_history()
        self.item_tag_sparse = self._build_sparse_item_tag()


        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["utid2itid", 'itid2utid']

    def _build_sparse_user_history(self):
        rows = []
        cols = []
        data = []

        for user_id, history in self.user_history_dict.items():
            for item_id in history:
                rows.append(user_id)
                cols.append(item_id)
                data.append(1)

        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_users, self.n_items))

    def _build_sparse_item_tag(self):
        rows = []
        cols = []
        data = []

        for item_id, tags in enumerate(self.item_feat['item_tag_id_list']):
            for tag in tags:
                rows.append(self.item_feat['item_id'][item_id].cpu().item())
                cols.append(tag.cpu().item())
                data.append(1)

        print(max(rows))
        print(max(cols))
        print(self.n_items, self.max_item_tag)

        return sp.csr_matrix((data, (rows, cols)), shape=(self.n_items, self.max_item_tag))

    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    # def _calculate_item_score(self, user_id):
    #     user_histories = self.user_history_sparse[user_id].toarray()
    #     propagated_tags = user_histories @ self.item_tag_sparse.toarray()

    #     propagated_tag_scores = propagated_tags / (propagated_tags.sum(axis=1, keepdims=True) + 1e-9)
    #     all_scores = propagated_tag_scores @ self.item_tag_sparse.T.toarray()

    #     return all_scores

    def _calculate_item_score(self, user_id):
        user_histories = self.user_history_sparse[user_id]
        propagated_tags = user_histories @ self.item_tag_sparse

        propagated_tag_scores = propagated_tags.multiply(1 / (propagated_tags.sum(axis=1) + 1e-9))
        all_scores = propagated_tag_scores @ self.item_tag_sparse.T

        return all_scores.toarray()

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy().astype(int)
        item = interaction[self.ITEM_ID].cpu().numpy().astype(int)

        all_scores = self._calculate_item_score(user)

        scores = all_scores[np.arange(len(user)), item]
        scores = torch.tensor(scores).to(self.device)

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy().astype(int)

        all_scores = self._calculate_item_score(user)
        all_scores = torch.tensor(all_scores).to(self.device)

        return all_scores