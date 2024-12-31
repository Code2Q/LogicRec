import os
import numpy as np
import math
import torch
import json
import torch.nn as nn
import scipy.sparse as sp
from torch.nn.init import normal_
from scipy.sparse import csr_matrix, coo_matrix,save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
from joblib import Parallel, delayed
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType, ModelType
from collections import defaultdict
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import scipy.sparse

# def print_gpu_memory_usage():
#     allocated_memory = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
#     reserved_memory = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
#     print(f"Allocated Memory: {allocated_memory:.2f} MB")
#     print(f"Reserved Memory: {reserved_memory:.2f} MB")
#     print("---")

# def valid_utid_set():
#     with open('/home/liushuchang/.jupyter/public_datasets/ui_logic_preprocessed/mmuusertag_coverset_v2.json', 'r', encoding='utf-8') as f:
#         mmuusertag_coverset_v2 = json.load(f)
#     with open('/home/liushuchang/.jupyter/public_datasets/ui_logic_preprocessed/ut_idmap.json', 'r', encoding='utf-8') as f:
#         ut_idmap = json.load(f)

#     valid_tags = []
#     for key, value in mmuusertag_coverset_v2.items():
#         if key in ut_idmap:
#             valid_tags.append = int(ut_idmap[key]) + 1
#     return valid_tags

# def valid_itid_set():
#     with open('/home/liushuchang/.jupyter/public_datasets/ui_logic_preprocessed/mmuitemtag_coverset_v2.json', 'r', encoding='utf-8') as f:
#         mmuitemtag_coverset_v2 = json.load(f)

#     with open('/home/liushuchang/.jupyter/public_datasets/ui_logic_preprocessed/it_idmap.json', 'r', encoding='utf-8') as f:
#         it_idmap = json.load(f)

#     valid_tags = []
#     for key, value in mmuitemtag_coverset_v2.items():
#         if key in it_idmap:
#             valid_tags.append = int(it_idmap[key]) + 1
#     return valid_tags


batch_idx = 0

# def draw_tag_distribution(multi_step_matrix, batch_idx):
#     step_num = multi_step_matrix.shape[0]

#     batch_idx += 1
#     for i in range(step_num):
#         pass
        


class ItemCF(GeneralRecommender):

    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(ItemCF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]
        self.RATING = config["RATING_FIELD"]
        self.use_logic = config["use_logic"]
        self.gamma = config["gamma"]
        self.draw = 0
        self.dataset = dataset


        if self.use_logic:
            self.item_feat = dataset.get_item_feature().to(self.device)
            print("---item to use/item tag features loaded---")
            print(f"dataset.field2id_token {dataset.field2id_token}")
            # self.utid2itid = dataset.utid2itid_feat
            # self.itid2utid = dataset.itid2utid_feat
            self.mode = config['mode']
            self.tag_decay = config['tag_decay']
            

            # self.max_user_tag = dataset.utid2itid_feat['user_tag_id'].max() + 1
            # self.max_item_tag = dataset.utid2itid_feat['item_tag_id_list'].max() + 1
            
            self.logic_step = config["logic_step"] if "logic_step" in config else 1

            self.itid_to_utid = load_npz("sparse_itid_to_utid.npz")
            self.itid_to_utid_norm = self.itid_to_utid.multiply(1 / (self.itid_to_utid.sum(axis=1) + 1e-9))
            self.max_item_tag, self.max_user_tag = self.itid_to_utid.shape
 
            self.item_tag_sparse, self.user_tag_sparse = self._id_to_sparse_tag()
            
            # print(f"---item_tag_sparse user_tag_sparse---")
            # print(self.item_tag_sparse.shape, self.item_tag_sparse.data)
            # print(self.user_tag_sparse.shape, self.user_tag_sparse.data)
            # (10396, 887293) (10396, 301909)
            # tensor(301908) tensor(0), tensor(887292) tensor(0)
            # print(dataset.utid2itid_feat['user_tag_id'].max(),dataset.utid2itid_feat['user_tag_id'].min())
            # print(dataset.utid2itid_feat['item_tag_id_list'].max(), dataset.utid2itid_feat['item_tag_id_list'].min())

            self.item_tag_sparse_norm = self.item_tag_sparse.multiply(1 / (self.item_tag_sparse.sum(axis=1) + 1e-9))
            self.user_tag_sparse_norm = self.user_tag_sparse.multiply(1 / (self.user_tag_sparse.sum(axis=1) + 1e-9))

            # if not os.path.exists("sparse_utid_to_itid.npz"):
            #     self.utid_to_itid = self._build_sparse_utid_to_itid()
            # else:
            self.utid_to_itid = load_npz("sparse_utid_to_itid.npz")
            self.utid_to_itid_norm = self.utid_to_itid.multiply(1 / (self.utid_to_itid.sum(axis=1) + 1e-9))
            
            

            # if not os.path.exists("sparse_itid_to_utid.npz"):
            #     self.itid_to_utid = self._build_sparse_itid_to_utid()
            # else:

            self.tag_weight = config["tag_weight"] if "tag_weight" in config else 0.01

            print("=======================")

            print(self.utid_to_itid.shape, self.itid_to_utid.shape, self.item_tag_sparse.shape, self.user_tag_sparse.shape)
            # (5594, 7178) (7178, 5593) (10396, 7178) (10396, 5593)


        # load parameters info
        self.inter_matrix_type = config["inter_matrix_type"]
        self.top_k = 50  # Number of top similar items to consider
        self.tag_weight = config["tag_weight"] if "tag_weight" in config else 0.01
        self.max_tag_branch = config["max_tag_branch"] if "max_tag_branch" in config else 1000


        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.batch_idx = 0
        

        # generate intermediate data
        if self.inter_matrix_type == "01":
            (
                self.history_user_id,
                self.history_user_value,
                self.history_user_length,
            ) = dataset.history_user_matrix()
            (
                self.history_item_id,
                self.history_item_value,
                self.history_item_length,
            ) = dataset.history_item_matrix()
            self.interaction_matrix = dataset.inter_matrix(form="csr").astype(np.float32)
        elif self.inter_matrix_type == "rating":
            (
                self.history_user_id,
                self.history_user_value,
                self.history_user_length,
            ) = dataset.history_user_matrix(value_field=self.RATING)
            (
                self.history_item_id,
                self.history_item_value,
                self.history_item_length,
            ) = dataset.history_item_matrix(value_field=self.RATING)
            self.interaction_matrix = dataset.inter_matrix(form="csr", value_field=self.RATING).astype(np.float32)
        else:
            raise ValueError(
                "The inter_matrix_type must be in ['01', 'rating'] but got {}".format(
                    self.inter_matrix_type
                )
            )

        self.max_rating = self.history_user_value.max()
        self.item_popular = self.history_user_length,
        self.item_similarity = self.calculate_item_similarity()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def save_propagated_tags(self, step, propagated_tags, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)

        # for i, tag_matrix in enumerate(propagated_tags):
        dense_matrix =propagated_tags.todense()  # 将稀疏矩阵转换为密集矩阵
        filename = os.path.join(directory, f'propagated_{self.mode}_step_{step}_bth_{self.batch_idx}.npz')
        np.savez(filename, dense_matrix)  # 使用 numpy.savez 保存密集矩阵
        print(f"Saved {filename}")
        self.batch_idx += 1

    def save_final_scores(self, scores, directory):

        if not os.path.exists(directory):
            os.makedirs(directory)

        # for i, tag_matrix in enumerate(scores):
        dense_matrix = scores.todense()  # 将稀疏矩阵转换为密集矩阵
        filename = os.path.join(directory, f'item_{self.mode}_bth_{self.batch_idx}.npz')
        np.savez(filename, dense_matrix)  # 使用 numpy.savez 保存密集矩阵
        print(f"Saved {filename}")


    def count_greater_than_threshold(self, scr, threshold=0.5):
        count_per_row = np.zeros(scr.shape[0], dtype=int)
        for i in range(scr.shape[0]):
            row_start = scr.indptr[i]
            row_end = scr.indptr[i + 1]
            row_data = scr.data[row_start:row_end]
            count_per_row[i] = (row_data > threshold).sum()
        return count_per_row

    # def _id_to_sparse_tag(self):
    #     item_ids = self.item_feat['item_id'].cpu().numpy()
    #     item_tag_id_lists = [tags.cpu().numpy() for tags in self.item_feat['item_tag_id_list']]
        
    #     # 初始化稀疏矩阵的行、列和数据
    #     rows, cols, data = [], [], []
    #     for i, (item_id, tags) in enumerate(zip(item_ids, item_tag_id_lists)):
    #         # 去掉数组末尾的零
    #         non_zero_tags = tags[:np.max(np.nonzero(tags)) + 1] if np.any(tags) else np.array([])
    #         rows.extend([item_id] * len(non_zero_tags))
    #         cols.extend(non_zero_tags)
    #         data.extend([1] * len(non_zero_tags))

    #     iid2itid_csr = sp.csr_matrix((data, (rows, cols)), shape=(self.n_items, self.max_item_tag))


    #     user_tag_id_lists = [tags.cpu().numpy() for tags in self.item_feat['user_tag_id_list']]
    #     ut_rows, ut_cols, ut_data = [], [], []
    #     for i, (ut_id, tags) in enumerate(zip(item_ids, user_tag_id_lists)):
    #         # 去掉数组末尾的零
    #         non_zero_tags = tags[:np.max(np.nonzero(tags)) + 1] if np.any(tags) else np.array([])
    #         ut_rows.extend([ut_id] * len(non_zero_tags))
    #         ut_cols.extend(non_zero_tags)
    #         ut_data.extend([1] * len(non_zero_tags))
    #     # print(ut_rows, ut_cols)
    #     # 创建 CSR 矩阵
    #     iid2utid_csr = sp.csr_matrix((ut_data, (ut_rows, ut_cols)), shape=(self.n_items, self.max_user_tag))
    #     print("@@@@@@@@@@")
    #     print(iid2itid_csr.data, iid2utid_csr.data)
    #     return iid2itid_csr, iid2utid_csr




    def _id_to_sparse_tag(self):
            item_ids = self.item_feat['item_id']
            item_tag_id_lists = [tags for tags in self.item_feat['item_tag_id_list']]
            
            rows, cols, data = [], [], []

            user_tag_id_lists = [tags for tags in self.item_feat['user_tag_id_list']]
            ut_rows, ut_cols, ut_data = [], [], []

            for i, (item_id, tags) in enumerate(zip(item_ids, item_tag_id_lists)):
                non_zero_tags = tags[tags > 0]  # 筛选大于0的标签
                non_zero_tags = non_zero_tags.cpu().numpy()  # 将张量从 GPU 移动到 CPU 并转换为 NumPy 数组
                true_tags = self.dataset.field2id_token['item_tag_id_list'][non_zero_tags].astype(float).astype(int)
                
                # 确保 true_tags 是一个数组或列表
                if np.isscalar(true_tags):
                    true_tags = [true_tags]
                
                rows.extend([item_id] * len(non_zero_tags))
                cols.extend(true_tags)
                data.extend([1] * len(non_zero_tags))

            # 确保 rows, cols, data 是 NumPy 数组
            rows = np.array([r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in rows])
            cols = np.array([c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in cols])
            data = np.array([d.cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data])

            iid2itid_csr = sp.csr_matrix((data, (rows, cols)), shape=(self.n_items, self.max_item_tag))

            for i, (ut_id, tags) in enumerate(zip(item_ids, user_tag_id_lists)):
                non_zero_tags = tags[tags > 0]  # 筛选大于0的标签
                non_zero_tags = non_zero_tags.cpu().numpy()  # 将张量从 GPU 移动到 CPU 并转换为 NumPy 数组
                true_tags = self.dataset.field2id_token['user_tag_id_list'][non_zero_tags].astype(float).astype(int)
                
                # 确保 true_tags 是一个数组或列表
                if np.isscalar(true_tags):
                    true_tags = [true_tags]
                
                ut_rows.extend([ut_id] * len(non_zero_tags))
                ut_cols.extend(non_zero_tags)
                ut_data.extend([1] * len(non_zero_tags))

            # 确保 ut_rows, ut_cols, ut_data 是 NumPy 数组
            ut_rows = np.array([r.cpu().numpy() if isinstance(r, torch.Tensor) else r for r in ut_rows])
            ut_cols = np.array([c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in ut_cols])
            ut_data = np.array([d.cpu().numpy() if isinstance(d, torch.Tensor) else d for d in ut_data])

            iid2utid_csr = sp.csr_matrix((ut_data, (ut_rows, ut_cols)), shape=(self.n_items, self.max_user_tag))
            return iid2itid_csr, iid2utid_csr
    

    # def _build_sparse_utid_to_itid(self):
    #     ut_ids = self.utid2itid['user_tag_id'].cpu().numpy()
    #     item_tag_id_lists = [tags.cpu().numpy() for tags in self.utid2itid['item_tag_id_list']]
    #     # item_tag_id_lists = [self.reindex_item_tag_list(tags).cpu().numpy() for tags in self.utid2itid['item_tag_id_list']]
    #     rows, cols, data = [], [], []

    #     user_tag_max = max(ut_ids)
    #     # user_tag_min = min([tags.min() for tags in user_tag_id_lists])
    #     print(f"User tag ID lists - Max: {user_tag_max}")
    #     # print(f"self.max_item_tag {self.max_item_tag}, self.max_user_tag {self.max_user_tag}")
        
    #     for i, (utid, tags) in enumerate(zip(ut_ids, item_tag_id_lists)):
    #         # 筛选大于0的标签
    #         non_zero_tags = tags[tags > 0]
    #         for tag in non_zero_tags:
    #             # if utid <= self.max_user_tag and tag <= self.max_item_tag:
    #             rows.append(utid)
    #             cols.append(tag)
    #             data.append(1)

    #     sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.max_user_tag, self.max_item_tag))
    #     print("[[build ut2it sparse_mat from scratch]]")
    #     sp.save_npz('sparse_utid_to_itid.npz', sparse_matrix)
    #     return sparse_matrix

    # def _build_sparse_itid_to_utid(self):
    #     it_ids = self.itid2utid['item_tag_id'].cpu().numpy()
    #     user_tag_id_lists = [tags.cpu().numpy() for tags in self.itid2utid['user_tag_id_list']]
    #     # user_tag_id_lists = [self.reindex_item_tag_list(tags).cpu().numpy() for tags in self.itid2utid['user_tag_id_list']]
    #     rows, cols, data = [], [], []
    #     for i, (itid, tags) in enumerate(zip(it_ids, user_tag_id_lists)):
    #         # 筛选大于0的标签
    #         non_zero_tags = tags[tags > 0]
    #         for tag in non_zero_tags:
    #             # if itid < self.max_item_tag and tag < self.max_user_tag:
    #             rows.append(itid)
    #             cols.append(tag)
    #             data.append(1)

    #     sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.max_item_tag, self.max_user_tag))
    #     print("[[build it2ut sparse_mat from scratch]]")
    #     sp.save_npz('sparse_itid_to_utid.npz', sparse_matrix)
    #     return sparse_matrix



    def get_sparse_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        interaction_matrix = self.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(interaction_matrix.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(interaction_matrix.shape))
        return SparseL

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def cosine_similarity_gpu(self, tensor):
    #     tensor = tensor.to_dense()
    #     norms = torch.norm(tensor, p=2, dim=1, keepdim=True)
    #     norms = norms + 1e-8
    #     normalized_tensor = tensor / norms
    #     dot_products = torch.sparse.mm(normalized_tensor, normalized_tensor.t())
    #     similarity = dot_products.to_dense()
    #     return similarity
    

    def calculate_item_similarity(self):

        item_user_matrix = self.get_sparse_mat().transpose(0, 1).to(self.device)
        item_degrees = torch.sparse.sum(item_user_matrix, dim=1).to_dense() + 1e-8
        co_occur = torch.sparse.mm(item_user_matrix, item_user_matrix.t()).to_dense()
        
        similarity_matrix = co_occur / torch.sqrt(torch.ger(item_degrees, item_degrees))    
        # Create a mask to set the diagonal elements to 0
        mask = torch.eye(similarity_matrix.size(0), device=self.device)
        similarity_matrix = torch.where(mask == 1, torch.tensor(0.0, device=self.device), similarity_matrix)

        return similarity_matrix

    # def calculate_itemtag_similarity(self):
    #     item_user_matrix = self.get_sparse_mat().transpose(0, 1).to(self.device)
    #     similarity_matrix = self.cosine_similarity_gpu(item_user_matrix)

    #     # Create a mask to set the diagonal elements to 0
    #     mask = torch.eye(similarity_matrix.size(0), device=self.device)
    #     similarity_matrix = torch.where(mask == 1, torch.tensor(0.0, device=self.device), similarity_matrix)

    #     return similarity_matrix


    def log_transform(self, x, count=1.0):
        print(f'tag score {x.data.min(), x.data.max()}')
        return sp.csr_matrix((np.maximum(np.log(count + x.data),0), x.indices, x.indptr), shape=x.shape)
    
    def print_max_min_value_and_index(self, scr, i):
        # 获取第 i 行的起始和结束位置
        row_start = scr.indptr[i]
        row_end = scr.indptr[i + 1]
        
        # 获取第 i 行的非零元素及其对应的列索引
        row_data = scr.data[row_start:row_end]
        row_indices = scr.indices[row_start:row_end]
        
        # 找到最大值及其对应的列索引
        if len(row_data) > 0:
            min_value = row_data.min()
            max_value = row_data.max()
            min_index = row_indices[row_data.argmin()]
            max_index = row_indices[row_data.argmax()]
            print(f"第 {i} 个user的tag freq最大值: {max_value}, 对应的列索引: {max_index}")
            print(f"第 {i} 个user的tag freq最小值: {min_value}, 对应的列索引: {min_index}")
        else:
            print(f"第 {i} 行没有非零元素")
    
    def get_top_k_indices(self, scr, k=1000):
        top_k_indices_per_row = []
        for i in range(scr.shape[0]):
            row_start = scr.indptr[i]
            row_end = scr.indptr[i + 1]
            row_data = scr.data[row_start:row_end]
            row_indices = scr.indices[row_start:row_end]
            
            if len(row_data) > k:
                sorted_indices = row_indices[np.argsort(row_data)[-k:][::-1]]  # 从大到小排序并提取前 k 大的索引
            else:
                sorted_indices = row_indices[np.argsort(row_data)[::-1]]
            
            top_k_indices_per_row.append(sorted_indices)
        return top_k_indices_per_row
    
    def retain_top_k_elements(self, scr, k=1000):
        top_k_indices_per_row = self.get_top_k_indices(scr, k)
        # 创建新的数据结构来存储结果
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for i in range(scr.shape[0]):
            row_start = scr.indptr[i]
            row_end = scr.indptr[i + 1]
            row_data = scr.data[row_start:row_end]
            row_indices = scr.indices[row_start:row_end]
            
            top_k_indices = top_k_indices_per_row[i]
            top_k_mask = np.isin(row_indices, top_k_indices)
            
            new_data.extend(row_data[top_k_mask])
            new_indices.extend(row_indices[top_k_mask])
            new_indptr.append(len(new_data))
        
        # 创建新的稀疏矩阵
        new_scr = sp.csr_matrix((new_data, new_indices, new_indptr), shape=scr.shape)
        return new_scr

    
    def calculate_tag_score(self, user_history_vector):

        propogated_tags = []
        for i in range(1, self.logic_step+1):
            if self.mode == 'item_tag':
                propagated_tag_step = self._calculate_item_tag_score(user_history_vector, i)

            elif self.mode == 'user_tag':
                propagated_tag_step = self._calculate_user_tag_score(user_history_vector, i)

            print(f"~~~~~propagated tags shape in step {i}: {propagated_tag_step.shape}")

            propogated_tags.append(propagated_tag_step)
            # self.count_greater_than_threshold(propagated_tag_step, 0.5)
        # draw_tag_distribution(propogated_tags, batch_idx)
        
        log_propagated_tags = sp.csr_matrix(propogated_tags[0].shape)

        counts_per_matrix = [self.count_greater_than_threshold(matrix, 0) for matrix in propogated_tags]
        for matrix_index, counts in enumerate(counts_per_matrix):
            print(f"矩阵 {matrix_index + 1} 每一行中元素大于 0 的个数:")
            for row_index, count in enumerate(counts):
                print(f"  第 {row_index + 1} 行: {count}")

        # weight = 10
        top_k_indices_per_matrix = [self.get_top_k_indices(matrix, 20) for matrix in propogated_tags]
        common_indices_per_row = []

        for i in range(propogated_tags[0].shape[0]):
            sets = [set(top_k_indices_per_matrix[m][i]) for m in range(len(propogated_tags))]
            common_indices = set.intersection(*sets)
            common_indices_per_row.append(common_indices)
        # for i, common_indices in enumerate(common_indices_per_row):
        #     print(f"第 {i+1} 行前20大的标签列索引相同的索引:")
        #     print(common_indices)
        #     # print(f"第 {i+1} 行共同索引个数：{len(common_indices)}")

        for i in range(len(propogated_tags)):
            print(f" ------- {i+1}-th tag frequency---------\n")
            self.print_max_min_value_and_index(propogated_tags[i], 0)
            self.save_propagated_tags(i, propogated_tags[i], '/home/liushuchang/.jupyter/yuqing_workspace/recbole/saved/itemcf_propogated_tags')
            # self.print_max_min_value_and_index(propogated_tags[i], 1)
            # self.print_max_min_value_and_index(propogated_tags[i], 2)
            # print(propogated_tags[i].max(), propogated_tags[i].min())
            weight = self.tag_decay
            log_propagated_tags += propogated_tags[i] * (weight ** i)
            non_zero_indices = np.nonzero(propogated_tags[i][0])
            # print(f"---- tags_in_step_{i+1} for user0 ---- {propogated_tags[i][0]}")
            print(f'----{i+1}-th non zero idx len:{len(non_zero_indices[0]),len(non_zero_indices[1])}')

            # Get the matching indices
        

        if self.mode == 'item_tag':
            # matching_indices = self.find_matching_indices(log_propagated_tags, self.item_tag_sparse)

            # # Print the matching indices for each row
            # for i, indices in enumerate(matching_indices):
            #     print(f"User {i}: Matching indices - {indices}")
            all_scores = log_propagated_tags @ self.item_tag_sparse.T
            
            print(f"scores {all_scores}")
            self.save_final_scores(all_scores, '/home/liushuchang/.jupyter/yuqing_workspace/recbole/saved/itemcf_final_scores')
        elif self.mode == 'user_tag':
            # matching_indices = self.find_matching_indices(log_propagated_tags, self.user_tag_sparse)
            # # Print the matching indices for each row
            # for i, indices in enumerate(matching_indices):
            #     print(f"User {i}: Matching indices - {indices}")
            all_scores = log_propagated_tags @ self.user_tag_sparse.T
            self.save_final_scores(all_scores, '/home/liushuchang/.jupyter/yuqing_workspace/recbole/saved/itemcf_final_scores')
            print(f"scores {all_scores}")
        
        # print(f"===all scores==")
        # all_scores = torch.log(all_scores)
        # all_scores = cosine_similarity(propagated_tag_scores, self.item_tag_sparse)
        # return all_scores
        return all_scores.toarray()

    def _calculate_item_tag_score(self, user_history_vector, logic_step):
        # user_histories = self.interaction_matrix[user_id] #[user x n_items]
        user_history_cpu = user_history_vector.to('cpu').numpy()
        user_histories_coo = coo_matrix(user_history_cpu)
        user_histories = user_histories_coo.tocsr()
        # utid_to_itid = self.utid_to_itid_norm
        # itid_to_utid = self.itid_to_utid_norm

        if logic_step % 2 == 1:
            propagated_item_tags = user_histories @ self.item_tag_sparse
            # propagated_item_tags = user_histories @ self.item_tag_sparse
            count = self.count_greater_than_threshold(propagated_item_tags, 0.1)

            print(f"logic step == 1 每一行中元素大于 0.1 的个数: {count}")
            count = self.count_greater_than_threshold(propagated_item_tags, 0.3)
            print(f"logic step == 1 每一行中元素大于 0.3 的个数: {count}")

            # propagated_item_tags = self.retain_top_k_elements(propagated_item_tags, k=self.max_tag_branch)
            propagated_item_tags = self.log_transform(propagated_item_tags)
            propagated_item_tags = sp.csr_matrix(normalize(propagated_item_tags, norm='l1', axis=1))
            # propogated_tags.append(propagated_item_tags)

            if logic_step > 1:
                iter_num = self.logic_step // 2
                for i in range(iter_num):
                    utids = propagated_item_tags @ self.itid_to_utid
                    # utids = self.retain_top_k_elements(utids, k=self.max_tag_branch)
                    propagated_user_tags = self.log_transform(utids)
                    propagated_user_tags = sp.csr_matrix(normalize(propagated_user_tags, norm='l1', axis=1))

                    itids = propagated_user_tags @ self.utid_to_itid
                    # itids = self.retain_top_k_elements(itids, k=self.max_tag_branch)
                    propagated_item_tags = self.log_transform(itids)
                    propagated_item_tags = sp.csr_matrix(normalize(propagated_item_tags, norm='l1', axis=1))
                    # propogated_tags.append(propagated_item_tags)
                    # propagated_item_tags = normalize(np.log(propagated_item_tags))
                    # propagated_user_tags = normalize(np.log(propagated_user_tags))

            log_propagated_item_tags = propagated_item_tags
        
        elif logic_step % 2 == 0:
            uid2utid = user_histories @ self.user_tag_sparse
            # uid2utid = self.retain_top_k_elements(uid2utid, k=self.max_tag_branch)
            uid2utid = self.log_transform(uid2utid)
            uid2utid = sp.csr_matrix(normalize(uid2utid, norm='l1', axis=1))
            # uid2utid = normalize(uid2utid)
            print(f"------------ {uid2utid.shape} {self.utid_to_itid.shape}")
            uid2itid = uid2utid @ self.utid_to_itid
            # uid2itid = self.retain_top_k_elements(uid2itid, k=self.max_tag_branch)
            uid2itid = self.log_transform(uid2itid)
            uid2itid = sp.csr_matrix(normalize(uid2itid, norm='l1', axis=1))
            # propogated_tags.append(uid2itid)
            # uid2itid = normalize(uid2itid)
            

            if logic_step > 2:
                iter_num = self.logic_step // 2
                for i in range(iter_num-1):
                    # 计算 uid2utid
                    uid2utid = uid2itid @ self.itid_to_utid
                    # uid2utid = self.retain_top_k_elements(uid2utid, k=self.max_tag_branch)
                    uid2utid = self.log_transform(uid2utid)
                    uid2utid = sp.csr_matrix(normalize(uid2utid, norm='l1', axis=1))

                    # 计算 uid2itid
                    uid2itid = uid2utid @ self.utid_to_itid
                    # uid2itid = self.retain_top_k_elements(uid2itid, k=self.max_tag_branch)
                    uid2itid = self.log_transform(uid2itid)
                    uid2itid = sp.csr_matrix(normalize(uid2itid, norm='l1', axis=1))

            log_propagated_item_tags = uid2itid
            # 保存到本地
        # np.save(f'tag_saved/log_propagated_item_tags_step_{logic_step}.npy', log_propagated_item_tags.toarray())

        return log_propagated_item_tags

    
    def _calculate_user_tag_score(self, user_history_vector, logic_step):
        user_history_cpu = user_history_vector.to('cpu').numpy()
        user_histories_coo = coo_matrix(user_history_cpu)
        user_histories = user_histories_coo.tocsr()
        # user_histories = self.interaction_matrix[user_id] #[user x n_items]
        # utid_to_itid = self.utid_to_itid_norm
        # itid_to_utid = self.itid_to_utid_norm

        if logic_step % 2 == 1:
            propagated_user_tags = user_histories @ self.user_tag_sparse
            # propagated_user_tags = self.retain_top_k_elements(propagated_user_tags, k=self.max_tag_branch)
            propagated_user_tags = self.log_transform(propagated_user_tags)
            # propagated_user_tags = torch.clamp(propagated_user_tags, min=0)
            propagated_user_tags = sp.csr_matrix(normalize(propagated_user_tags, norm='l1', axis=1))
            propagated_tag_scores = propagated_user_tags


            if logic_step > 1:
                iter_num = logic_step // 2
                            
                for i in range(iter_num):
                    propagated_item_tags = propagated_user_tags @ self.utid_to_itid
                    # propagated_item_tags = self.retain_top_k_elements(propagated_item_tags, k=self.max_tag_branch)
                    propagated_item_tags = self.log_transform(propagated_item_tags)
                    # propagated_item_tags = torch.clamp(propagated_item_tags, min=0)
                    propagated_item_tags = sp.csr_matrix(normalize(propagated_item_tags, norm='l1', axis=1))
                    
                    propagated_user_tags = propagated_item_tags @ self.itid_to_utid
                    # propagated_user_tags = self.retain_top_k_elements(propagated_user_tags, k=self.max_tag_branch)
                    propagated_user_tags = self.log_transform(propagated_user_tags)
                    # propagated_user_tags = torch.clamp(propagated_user_tags, min=0)
                    propagated_user_tags = sp.csr_matrix(normalize(propagated_user_tags, norm='l1', axis=1))
            propagated_tag_scores = propagated_user_tags

        elif logic_step % 2 == 0:
            # user 2 user tag [3, n_item] x [n_item, max_user_tag]
            uid2itid = user_histories @ self.item_tag_sparse #[user x max_user_tag]
            uid2itid = self.log_transform(uid2itid)
            # uid2itid = torch.clamp(uid2itid, min=0)
            uid2itid = sp.csr_matrix(normalize(uid2itid, norm='l1', axis=1))
            itid2utid = uid2itid @ self.itid_to_utid
            itid2utid = self.log_transform(itid2utid)
            # itid2utid = torch.clamp(itid2utid, min=0)
            itid2utid = sp.csr_matrix(normalize(itid2utid, norm='l1', axis=1))
            # propagated_tag_scores = itid2utid

            if logic_step > 2:
                iter_num = logic_step // 2
       
                for i in range(iter_num-1):
                    itid2itid = itid2utid @ self.utid_to_itid
                    # itid2itid = self.retain_top_k_elements(itid2itid, k=self.max_tag_branch)
                    itid2itid = self.log_transform(itid2itid)
                    # itid2itid = torch.clamp(itid2itid, min=0)
                    itid2itid = sp.csr_matrix(normalize(itid2itid, norm='l1', axis=1))

                    itid2utid = itid2itid @ self.itid_to_utid
                    # itid2utid = self.retain_top_k_elements(itid2utid, k=self.max_tag_branch)
                    itid2utid = self.log_transform(itid2utid)
                    # itid2utid = torch.clamp(itid2utid, min=0)
                    itid2utid = sp.csr_matrix(normalize(itid2utid, norm='l1', axis=1))
                    
                    # propogated_tags.append(itid2utid)
                    
            propagated_tag_scores = itid2utid
            

        return propagated_tag_scores
        


    def forward(self, user, item):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))


    def predict(self, interaction):
        # pass
        user = interaction[self.USER_ID].to(self.device)
        
        user_history_vector = torch.zeros(1, self.n_items).to(self.device)
        
        user_interactions = self.history_item_id[user]
        user_history_vector[0, user_interactions] = 1
        predict = torch.matmul(user_history_vector, self.item_similarity.T)
        return predict
    
    def find_matching_indices(self, log_propagated_tags, item_tag_sparse):
        matching_indices = []
        for i in range(log_propagated_tags.shape[0]):
            hit_item = 0
            count = 0
            # Get the non-zero indices for the current row in both matrices
            log_propagated_nonzero = log_propagated_tags[i].nonzero()[1]
            for j in range(item_tag_sparse.shape[0]):
                item_tag_nonzero = item_tag_sparse[j].nonzero()[1]
                common_indices = np.intersect1d(log_propagated_nonzero, item_tag_nonzero)
                # count += len(common_indices)
                # single_user_indices.append(item_hit_len)
                if len(common_indices) > 0:
                    count += len(common_indices)
                    hit_item += 1
                # single_user_indices.append(common_indices)
            matching_indices.append(count/hit_item)
        
        return matching_indices



    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID] # shape [3]
        user_history_vector = torch.zeros(len(user), self.n_items).to(self.device)

        user_interactions = self.history_item_id[user] #  [3, 374]
        for i, user_interactions_i in enumerate(user_interactions):
            non_zero_indices = torch.nonzero(user_interactions_i, as_tuple=True)[0].to(self.device)
            interacted_item_ids = user_interactions_i[non_zero_indices]
            # interacted_items_popular = torch.log1p(self.history_user_length[interacted_item_ids].float()).to(self.device)
            # interacted_items_popular = self.history_user_length[interacted_item_ids].to(self.device) / self.history_user_length.max().to(self.device)
            gamma = torch.tensor(self.gamma, dtype=torch.float32, device=self.device)
            powers = gamma ** (len(non_zero_indices) - (non_zero_indices + 1))
            user_history_vector[i, interacted_item_ids] = powers
            # user_history_vector[i, interacted_item_ids] = 1
            # print("===@@@@@@@@@@@")
            # print(user_history_vector[i, interacted_item_ids])
        
        # print(user_history_vector[0][:12])


        # scores = torch.matmul(self.interaction_matrix[user_id], self.item_similarity.T) # 3 x n_item
        scores = torch.matmul(user_history_vector, self.item_similarity.T) # 3 x n_item
        # scores = torch.matmul(user_history_vector_cf, self.item_similarity.T)

        # logic infer step:
        if self.use_logic:
            # if self.logic_step == 1:
            # user_cpu = user.cpu().numpy().astype(int)
            # normalization_factor = 1 / (user_history_vector.sum(dim=1) + 1e-9)
            # normalization_factor = normalization_factor.unsqueeze(1)
            # user_histories_norm = user_history_vector * normalization_factor

            # user_histories_norm = user_history_vector.multiply(1 / (user_history_vector.sum(axis=1) + 1e-9))

            # tag_scores = torch.tensor(self.calculate_tag_score(user_histories_norm)).to(self.device)
            scores0 = self.calculate_tag_score(user_history_vector)
            # self.save_final_scores(scores0)
            tag_scores = torch.tensor(scores0).to(self.device)
           
              
            # print(scores, tag_scores, scores.max(), scores.min(), tag_scores.max(), tag_scores.min())
            scores = scores + self.tag_weight * tag_scores
            # scores = self.tag_weight * tag_scores
        # scores = self.sigmoid(scores)

        return scores