import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn.init import normal_
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix,save_npz, load_npz
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from sklearn.preprocessing import normalize
from recbole.model.loss import BPRLoss, EmbLoss
import time

class TagMF(GeneralRecommender):


    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LogicMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]
        self.RATING = config["RATING_FIELD"]
        self.mode = config['mode']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_weight = config["reg_weight"] if "reg_weight" in config else 1e-05
         # float32 type: the weight decay for l2 normalization
        self.item_bias = config['item_bias'] if "item_bias" in config else 1.0
        self.user_bias = config['item_bias'] if "item_bias" in config else 1.0

        # load parameters info
        self.tag_embedding_size = config["tag_embedding_size"] if "tag_embedding_size" in config else 64
        self.item_embedding_size = config["item_embedding_size"] if "item_embedding_size" in config else 64
        self.inter_matrix_type = config["inter_matrix_type"]

        self.item_feat = dataset.get_item_feature()
        print("---item to tag features loaded---")

        # print(self.item_feat['user_tag_id_list'],self.item_feat['user_tag_id_list'].shape)
        # print(self.item_feat['item_tag_id_list'], self.item_feat['item_tag_id_list'].shape)

        # self.utid2itid = dataset.utid2itid_feat
        # self.itid2utid = dataset.itid2utid_feat
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.gamma = config["gamma"]
        self.tag_decay = config['tag_decay']
        self.tag_weight = config['tag_weight'] if "tag_weight" in config else 1.0
        # self.tag_bias = config['tag_bias']


        # first reindex, then bulid sp mat
        # self.max_item_tag, self.max_user_tag = self.item_feat['item_tag_id_list'].max(), self.item_feat['user_tag_id_list'].max()
        self.max_item_tag, self.max_user_tag = 6386, 4950
        # print(f"==='user_tag_id' {self.max_user_tag, self.min_user_tag},  ==='item_tag_id' {self.max_item_tag, self.min_item_tag}")
        self.item_tag_sparse, self.user_tag_sparse = self._id_to_sparse_tag()
        
        self.logic_step = config["logic_step"] if "logic_step" in config else 1
        
        # self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # generate intermediate data
        if self.inter_matrix_type == "01":
            (
                self.history_user_id,
                self.history_user_value,
                self.history_user_len,
            ) = dataset.history_user_matrix()
            (
                self.history_item_id,
                self.history_item_value,
                self.history_item_len,
            ) = dataset.history_item_matrix()
            self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        elif self.inter_matrix_type == "rating":
            (
                self.history_user_id,
                self.history_user_value,
                _,
            ) = dataset.history_user_matrix(value_field=self.RATING)
            (
                self.history_item_id,
                self.history_item_value,
                _,
            ) = dataset.history_item_matrix(value_field=self.RATING)
            self.interaction_matrix = dataset.inter_matrix(
                form="coo", value_field=self.RATING
            ).astype(np.float32)
        else:
            raise ValueError(
                "The inter_matrix_type must in ['01', 'rating'] but get {}".format(
                    self.inter_matrix_type
                )
            )
        self.interaction_tensor = self.csr_to_sparse_tensor(self.interaction_matrix)
        
        self.item_similarity = self.calculate_item_similarity() # dense tensor in cpu
        # print(self.item_similarity)


 
        self.cf_score = torch.sparse.mm(self.interaction_tensor, self.item_similarity).to_dense().cpu()
        print(f"cf score {self.cf_score, self.cf_score.shape}")

        if not os.path.exists("sparse_utid_to_itid.npz"):
            self.utid_to_itid = self._build_sparse_utid_to_itid()
        else:
            print("[ load ut2it sparse mat from local npz file ]")
            self.utid_to_itid = load_npz("sparse_utid_to_itid.npz")
        # self.utid_to_itid_norm = self.utid_to_itid.multiply(1 / (self.utid_to_itid.sum(axis=1) + 1e-9))
        # self.utid_to_itid_sp_tensor = self.csr_to_sparse_tensor(self.utid_to_itid).to(self.device)
        # self.utid_to_itid_norm = self.utid_to_itid.multiply(1 / (self.utid_to_itid.sum(axis=1) + 1e-9))
            

        if not os.path.exists("sparse_itid_to_utid.npz"):
            self.itid_to_utid = self._build_sparse_itid_to_utid()
            
        else:
            print("[ load it2ut sparse mat from local npz file ]")
            self.itid_to_utid = load_npz("sparse_itid_to_utid.npz")
        # self.itid_to_utid_sp_tensor = self.csr_to_sparse_tensor(self.itid_to_utid).to(self.device)
        # self.itid_to_utid_norm = self.itid_to_utid.multiply(1 / (self.itid_to_utid.sum(axis=1) + 1e-9))
        
        # self.itid_to_utid_norm = self.itid_to_utid.multiply(1 / (self.itid_to_utid.sum(axis=1) + 1e-9))
        # self.time_weight = self.time_weight().to(self.device)

        # define layers
        self.item_tag_embeddings = nn.Embedding(self.max_item_tag, self.tag_embedding_size)
        self.user_tag_embeddings = nn.Embedding(self.max_user_tag, self.tag_embedding_size)
        self.item_embeddings = nn.Embedding(self.n_items, self.item_embedding_size)
        # self.user_embeddings = nn.Embedding(self.n_users, self.user_embedding_size)
        self.sigmoid = nn.Sigmoid()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        # self.ce_loss = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def csr_to_sparse_tensor(self, sparse_matrix):
        coo = sparse_matrix.tocoo()
        indices_np = np.array([coo.row, coo.col])
        indices = torch.tensor(indices_np, dtype=torch.long)
        # indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
        return sparse_tensor

    def _id_to_sparse_tag(self):
        item_ids = self.item_feat['item_id']
        item_tag_id_lists = [tags for tags in self.item_feat['item_tag_id_list']]
        
        rows, cols, data = [], [], []
        # item_tag_max = max([tags.max() for tags in item_tag_id_lists])
        # item_tag_min = min([tags.min() for tags in item_tag_id_lists])
        # print(f"Item tag ID lists - Max: {item_tag_max}, Min: {item_tag_min}")

        user_tag_id_lists = [tags for tags in self.item_feat['user_tag_id_list']]
        ut_rows, ut_cols, ut_data = [], [], []
        # user_tag_max = max([tags.max() for tags in user_tag_id_lists])
        # user_tag_min = min([tags.min() for tags in user_tag_id_lists])
        # print(f"User tag ID lists - Max: {user_tag_max}, Min: {user_tag_min}")
        # print(f"self.max_item_tag {self.max_item_tag}, self.max_user_tag {self.max_user_tag}")

        for i, (item_id, tags) in enumerate(zip(item_ids, item_tag_id_lists)):
            non_zero_tags = tags[tags > 0]  # 筛选大于0的标签
            rows.extend([item_id] * len(non_zero_tags))
            cols.extend(non_zero_tags)
            data.extend([1] * len(non_zero_tags))

        iid2itid_csr = sp.csr_matrix((data, (rows, cols)), shape=(self.n_items, self.max_item_tag))


        for i, (ut_id, tags) in enumerate(zip(item_ids, user_tag_id_lists)):
            non_zero_tags = tags[tags > 0]  # 筛选大于0的标签
            ut_rows.extend([ut_id] * len(non_zero_tags))
            ut_cols.extend(non_zero_tags)
            ut_data.extend([1] * len(non_zero_tags))

        iid2utid_csr = sp.csr_matrix((ut_data, (ut_rows, ut_cols)), shape=(self.n_items, self.max_user_tag))
        # print(iid2itid_csr.data, iid2utid_csr.data)
        return iid2itid_csr, iid2utid_csr
    
    # def _build_sparse_utid_to_itid(self):
    #     ut_ids = torch.tensor([tag for tag in self.utid2itid['user_tag_id']], device=self.device).cpu().numpy()
    #     item_tag_id_lists = [tags.cpu().numpy() for tags in self.utid2itid['item_tag_id_list']]
    #     rows, cols, data = [], [], []
        
    #     for i, (utid, tags) in enumerate(zip(ut_ids, item_tag_id_lists)):
    #         # 筛选大于0的标签
    #         non_zero_tags = tags[tags > 0]
    #         for tag in non_zero_tags:
    #             if utid <= self.max_user_tag and tag <= self.max_item_tag:
    #                 rows.append(utid)
    #                 cols.append(tag)
    #                 data.append(1)

    #     sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.max_user_tag, self.max_item_tag))
    #     print("[[build ut2it sparse_mat from scratch]]")
    #     sp.save_npz('sparse_utid_to_itid.npz', sparse_matrix)
    #     return sparse_matrix

    # def _build_sparse_itid_to_utid(self):
    #     it_ids = torch.tensor([tag for tag in self.itid2utid['item_tag_id']], device=self.device).cpu().numpy()
    #     user_tag_id_lists = [tags.cpu().numpy() for tags in self.itid2utid['user_tag_id_list']]
    #     rows, cols, data = [], [], []
    #     for i, (itid, tags) in enumerate(zip(it_ids, user_tag_id_lists)):
    #         # 筛选大于0的标签
    #         non_zero_tags = tags[tags > 0]
    #         for tag in non_zero_tags:
    #             if 0 <= itid < self.max_item_tag and 0 <= tag < self.max_user_tag:
    #                 rows.append(itid)
    #                 cols.append(tag)
    #                 data.append(1)

    #     sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.max_item_tag, self.max_user_tag))
    #     print("[[build it2ut sparse_mat from scratch]]")
    #     sp.save_npz('sparse_itid_to_utid.npz', sparse_matrix)

    #     return sparse_matrix
    
    def calculate_item_similarity(self):

        item_user_matrix = self.get_sparse_mat().transpose(0, 1)
        item_degrees = torch.sparse.sum(item_user_matrix, dim=1).to_dense() + 1e-8
        co_occur = torch.sparse.mm(item_user_matrix, item_user_matrix.t()).to_dense()
        
        similarity_matrix = co_occur / torch.sqrt(torch.ger(item_degrees, item_degrees))    
        # Create a mask to set the diagonal elements to 0
        mask = torch.eye(similarity_matrix.size(0))
        similarity_matrix = torch.where(mask == 1, torch.tensor(0.0), similarity_matrix)

        return similarity_matrix
    
    def get_sparse_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        interaction_matrix = self.interaction_matrix
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(interaction_matrix.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(interaction_matrix.shape))
        return SparseL
    
    def time_weight(self):
        user_items = self.history_item_id # [n_users, num_his_items]
        # print(f"user item shape {user_items.shape}")
        num_his_items = user_items.shape[1]
        user_his_len = self.history_item_len.unsqueeze(1) #[n_users, 1]
        # print(f"user his len shape {user_his_len.shape}")
        # arange_tensor = torch.arange(num_his_items, device=self.device, dtype=torch.float32).unsqueeze(0).expand(self.n_users, -1)
        arange_tensor = torch.arange(num_his_items, dtype=torch.float32).unsqueeze(0).expand(self.n_users, -1)
        user_his_len_expanded = user_his_len.expand(-1, num_his_items)
        weights = self.gamma ** (user_his_len_expanded - 1 - arange_tensor)
        mask = arange_tensor < user_his_len_expanded
        weights = weights * mask.float()
        # print(f"weights {weights, weights.shape}")
        return weights  #[n_user, max_his_len]
        # his_len = self.history_item_len #[n_users, 1]

    
    def calculate_tag_score(self, user, item=None):
        # scores = torch.matmul(user_history_vector, self.item_similarity.T)
        # batch_users, num_his_items = user.shape
        user_items = self.history_item_id[user] #[batch_users, max_history_items]
        his_len = self.history_item_len[user] #[batch_users]
        non_zero_indices = torch.nonzero(user_items, as_tuple=True)
        non_zero_user_items = user_items[non_zero_indices].to(self.device)  # [total_history_items]

        item_embeds = self.item_embeddings(non_zero_user_items) #[total_items, embedding_dim] torch.Size([1, 415, 128])
        item_tag_embeds = self.item_tag_embeddings.weight  # [num_tags, embedding_dim]
        user_tag_embeds = self.user_tag_embeddings.weight  # [num_tags, embedding_dim]
        del non_zero_user_items

        id2it_score = torch.matmul(item_embeds, item_tag_embeds.T)  #[total_items, n_tags]
        lengths = his_len.tolist()  # Convert tensor to list of lengths
        split_scores = torch.split(id2it_score, lengths) #[n_batch, x, n_tags]
        # Sum the scores for each user
        summed_scores = [torch.sum(scores, dim=0) for scores in split_scores]  # [batch_users, n_tags]
        id2it_score = torch.stack(summed_scores)  # [batch_users, n_tags]

        id2ut_score = torch.matmul(item_embeds, user_tag_embeds.T)
        split_user_scores = torch.split(id2ut_score, lengths)
        summed_user_scores = [torch.sum(scores, dim=0) for scores in split_user_scores]
        id2ut_score = torch.stack(summed_user_scores)
        # id2ut_score = self.sigmoid(id2ut_score)
        # id2ut_score_weighted_sum = torch.sum(id2ut_score * weights, dim=1)
        # id2ut_score = self.sigmoid(id2ut_score_weighted_sum)
        
        propogated_tags = []
    
        for i in range(1, self.logic_step+1):
            if self.mode == 'item_tag':
                propagated_tag_step = self._calculate_item_tag_score(i, id2it_score, id2ut_score)

            elif self.mode == 'user_tag':
                propagated_tag_step = self._calculate_user_tag_score(i, id2it_score, id2ut_score)

            propogated_tags.append(propagated_tag_step)
        # weights = torch.tensor([self.tag_decay ** i for i in range(len(propogated_tags))])
        # weights = torch.tensor([self.tag_decay ** i for i in range(len(propogated_tags))], device=self.device)
        # log_propagated_tags = sum([w * t for w, t in zip(weights, propogated_tags)]) #[1, n_tags]
        log_propagated_tags = sum([t for t in  propogated_tags]) #([batch, 6386])
        # non_zero_counts = torch.count_nonzero(log_propagated_tags, dim=1)
        # for i, count in enumerate(non_zero_counts):
        #     print(f"Row {i} has {count.item()} non-zero elements.")

        if item is not None:
            if self.mode == 'item_tag':
                item_emb = self.item_embeddings(item) #[n_batch, 128] #[n_tags, 128]
                next_item_tag_score = torch.matmul(item_emb, self.item_tag_embeddings.weight.transpose(0, 1)) #[batch_size, n_tags]
                # next_item_tag_score = self.sigmoid(next_item_tag_score)  #[batch_size, n_tags] #([batch, 6386])
                # item_emb = self.item_embeddings(item).unsqueeze(1)
                # next_item_tag_score = torch.mul(item_emb, self.item_tag_embeddings.weight).sum(dim=2)
                all_scores = torch.mul(next_item_tag_score, log_propagated_tags).sum(dim=1) #[n_batch, 1]
                # all_scores = self.sigmoid(all_scores)
                # all_scores = torch.mul(next_item_tag_score, log_propagated_tags).sum(dim=1) #[n_batch, 1]
                # 使用 view 方法调整形状
                # all_scores = self.sigmoid(all_scores)
                # cf_scores = torch.sum(self.item_similarity[user_items] )
            elif self.mode == 'user_tag':
                item_emb = self.item_embeddings(item).unsqueeze(1)
                next_user_tag_score = torch.mul(item_emb, self.user_tag_embeddings.weight).sum(dim=2)
                all_scores = torch.mul(next_user_tag_score, log_propagated_tags).sum(dim=1)
            # cf_score = torch.sum(self.item_similarity[user_items] )

        elif item is None:
            # item_emb = self.item_embeddings.weight #[n_items, 128]
            # item_tag_score = torch.mul(item_emb, self.item_tag_embeddings.weight).sum(dim=2)
            item_tag_score = torch.matmul(self.item_embeddings.weight, self.item_tag_embeddings.weight.transpose(0, 1))#[n_items, n_tags]
            # item_tag_score = torch.clamp_min(item_tag_score, 0)
            # item_tag_score = self.sigmoid(item_tag_score)
            all_scores = torch.mul(item_tag_score, log_propagated_tags).sum(dim=1, keepdim=True)  #[n_items]
            # all_scores = self.sigmoid(all_scores)
            # all_scores = self.sigmoid(all_scores)
        torch.cuda.empty_cache()
        return all_scores

    def _calculate_item_tag_score(self, logic_step, id2it_score, id2ut_score):

        if logic_step % 2 == 1:
            propagated_item_tags = id2it_score #[1, n_tags]
            # propagated_item_tags = torch.log(self.item_bias + propagated_item_tags)
            # propagated_item_tags = torch.clamp_min(propagated_item_tags, 0)
            # count = self.count_greater_than_threshold(propagated_item_tags, 0.1)
            # print(f"logic step == 1 每一行中元素大于 0.1 的个数: {count}")
            # count = self.count_greater_than_threshold(propagated_item_tags, 0.3)
            # print(f"logic step == 1 每一行中元素大于 0.3 的个数: {count}")

            if logic_step > 1:
                iter_num = self.logic_step // 2
                # propagated_user_tags = (user_histories @ self.user_tag_sparse_norm)
                for i in range(iter_num):
                    utids = torch.matmul(propagated_item_tags, self.itid_to_utid)
                    propagated_user_tags = torch.log(self.user_bias + utids)
                    propagated_user_tags = torch.clamp_min(propagated_user_tags, 0)
                    propagated_user_tags /= torch.norm(propagated_user_tags, p=1, dim=1, keepdim=True)

                    itids = torch.matmul(propagated_user_tags,self.utid_to_itid)
                    propagated_item_tags = torch.log(self.item_bias + itids)
                    propagated_item_tags = torch.clamp_min(propagated_item_tags, 0)
                    propagated_item_tags /= torch.norm(propagated_item_tags, p=1, dim=1, keepdim=True)
                    # propagated_item_tags = sp.csr_matrix(normalize(propagated_item_tags, norm='l1', axis=1))


            log_propagated_item_tags = propagated_item_tags
        
        elif logic_step % 2 == 0:
            uid2utid = id2ut_score #[1, n_user_tags]

            uid2itid = torch.matmul(uid2utid, self.utid_to_itid) 
            # uid2itid = self.retain_top_k_elements(uid2itid, k=self.max_tag_branch)
            uid2itid = torch.log(self.item_bias + uid2itid)
            # uid2itid = torch.clamp_min(uid2itid, 0)
            uid2itid /= torch.norm(uid2itid, p=1, dim=1, keepdim=True)

            if logic_step > 2:
                iter_num = self.logic_step // 2
                for i in range(iter_num-1):
                    # 计算 uid2utid
                    uid2utid = torch.matmul(uid2itid, self.itid_to_utid)
                    uid2utid = torch.log(self.user_bias+uid2utid)
                    # uid2utid = torch.clamp_min(uid2utid, 0)
                    uid2utid /= torch.norm(uid2utid, p=1, dim=1, keepdim=True)

                    # 计算 uid2itid
                    uid2itid = torch.matmul(uid2utid, self.utid_to_itid_norm)
                    # uid2itid = self.retain_top_k_elements(uid2itid, k=self.max_tag_branch)
                    uid2itid = torch.log(self.item_bias+uid2itid)
                    uid2itid /= torch.norm(uid2itid, p=1, dim=1, keepdim=True)

            log_propagated_item_tags = uid2itid

        return log_propagated_item_tags

    
    def _calculate_user_tag_score(self, logic_step, id2it_score, id2ut_score):
        # user_history_cpu = user_history_vector.to('cpu').numpy()
        # user_histories_coo = coo_matrix(user_history_cpu)
        # user_histories = user_histories_coo.tocsr()
        # user_histories = self.interaction_matrix[user_id] #[user x n_items]
        # utid_to_itid = self.utid_to_itid_norm
        # itid_to_utid = self.itid_to_utid_norm

        # if logic_step % 2 == 1:
        #     propagated_user_tags = user_histories @ self.user_tag_sparse_norm
        #     # propagated_user_tags = self.retain_top_k_elements(propagated_user_tags, k=self.max_tag_branch)
        #     propagated_user_tags = self.log_transform(propagated_user_tags)
        #     propagated_user_tags = sp.csr_matrix(normalize(propagated_user_tags, norm='l1', axis=1))
        #     propagated_tag_scores = propagated_user_tags


        #     if logic_step > 1:
        #         iter_num = logic_step // 2
                            
        #         for i in range(iter_num):
        #             propagated_item_tags = propagated_user_tags @ utid_to_itid
        #             # propagated_item_tags = self.retain_top_k_elements(propagated_item_tags, k=self.max_tag_branch)
        #             propagated_item_tags = self.log_transform(propagated_item_tags)
        #             propagated_item_tags = sp.csr_matrix(normalize(propagated_item_tags, norm='l1', axis=1))
                    
        #             propagated_user_tags = propagated_item_tags @ itid_to_utid
        #             # propagated_user_tags = self.retain_top_k_elements(propagated_user_tags, k=self.max_tag_branch)
        #             propagated_user_tags = self.log_transform(propagated_user_tags)
        #             propagated_user_tags = sp.csr_matrix(normalize(propagated_user_tags, norm='l1', axis=1))
        #     propagated_tag_scores = propagated_user_tags

        # elif logic_step % 2 == 0:
        #     # user 2 user tag [3, n_item] x [n_item, max_user_tag]
        #     uid2itid = user_histories @ self.item_tag_sparse_norm #[user x max_user_tag]
        #     uid2itid = self.log_transform(uid2itid)
        #     uid2itid = sp.csr_matrix(normalize(uid2itid, norm='l1', axis=1))
        #     itid2utid = uid2itid @ itid_to_utid
        #     itid2utid = self.log_transform(itid2utid)
        #     itid2utid = sp.csr_matrix(normalize(itid2utid, norm='l1', axis=1))
        #     # propagated_tag_scores = itid2utid

        #     if logic_step > 2:
        #         iter_num = logic_step // 2
       
        #         for i in range(iter_num-1):
        #             itid2itid = itid2utid @ utid_to_itid
        #             # itid2itid = self.retain_top_k_elements(itid2itid, k=self.max_tag_branch)
        #             itid2itid = self.log_transform(itid2itid)
        #             itid2itid = sp.csr_matrix(normalize(itid2itid, norm='l1', axis=1))

        #             itid2utid = itid2itid @ itid_to_utid
        #             # itid2utid = self.retain_top_k_elements(itid2utid, k=self.max_tag_branch)
        #             itid2utid = self.log_transform(itid2utid)
        #             itid2utid = sp.csr_matrix(normalize(itid2utid, norm='l1', axis=1))
                    
        #             # propogated_tags.append(itid2utid)
                    
        #     propagated_tag_scores = itid2utid
            

        # return propagated_tag_scores
        return -1


    def _init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def forward(self, user, item=None):
        # pass
        tag_score = self.calculate_tag_score(user, item)
        return tag_score

    def calculate_loss(self, interaction):
        # return torch.nn.Parameter(torch.zeros(1))
        # start_time = time.time()
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        # print(f"user shape in calcualte loss {user.shape}")
        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)

        print(f"pos neg item score from tag {pos_item_score[:10], neg_item_score[:10]}")
        # print(f"pos neg item score from bpr {pos_score, neg_score}")
        mf_loss = self.bpr_loss(pos_item_score, neg_item_score)

        # print(f"Calculate loss time: {start_time - end_time:.6f} seconds")
        print(f"mf loss {mf_loss, type(mf_loss)}")

        # # calculate regularization Loss

        # user_tag_embeddings = self.user_tag_embeddings.weight
        item_tag_embeddings = self.item_tag_embeddings.weight
        pos_ego_embeddings = self.item_embeddings(pos_item)
        neg_ego_embeddings = self.item_embeddings(neg_item)

        reg_loss = self.reg_loss(
            # user_tag_embeddings,
            item_tag_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=False,
        )
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # id2it_score, id2ut_score = self.forward(user, item)
        predict = self.forward(user, item)
        return predict

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        # user_items = self.history_item_id[user] #[batch_users, max_history_items]
        # # his_len = self.history_item_len[user] #[batch_users]
       
        # # non_zero_user_items = user_items[torch.nonzero(user_items, as_tuple=True)] # [total_history_items]
        # # print(non_zero_user_items.shape, non_zero_user_items)

        # # 确保 non_zero_user_items 是一维的索引张量
        # if non_zero_user_items.dim() == 1:
        #     item_sim = self.item_similarity[user_items[torch.nonzero(user_items, as_tuple=True)]]
        # else:
        #     item_sim = self.item_similarity[non_zero_user_items.flatten()]

        # # item_sim = self.item_similarity[non_zero_user_items] #[total_items, embedding_dim] torch.Size([1, 415, 128])
        # lengths = self.history_item_len[user].tolist()  # Convert tensor to list of lengths
        # split_sim = torch.split(item_sim, lengths) #[n_batch, x, n_tags]
        # # Sum the scores for each user
        # summed_scores = [torch.sum(scores, dim=0) for scores in split_sim]  # [batch_users, n_tags]
        # cf_score = torch.stack(summed_scores).to(self.device)  # [batch_users, n_tags]
        # # user_items = self.history_item_id[users]
        # all_history_item_ids = []
        # user_offsets = []
        # current_offset = 0

        # for user in users:
        #     history_item_id = self.history_item_id[user]
        #     all_history_item_ids.extend(history_item_id)
        #     user_offsets.append((current_offset, current_offset + len(history_item_id)))
        #     current_offset += len(history_item_id)

        # all_history_item_ids = torch.LongTensor(all_history_item_ids)

        # # 使用 index_select 索引 item_similarity
        # similarity_matrix = self.item_similarity.index_select(0, all_history_item_ids)

        # # 初始化一个张量来存储每个用户的相似度求和结果
        # similarity_sums = torch.zeros((len(users), self.item_similarity.size(1)))

        # # 对每个用户的相似度进行求和
        # for i, (start, end) in enumerate(user_offsets):
        #     similarity_sums[i] = torch.sum(similarity_matrix[start:end], dim=0)
        # cf_score = similarity_sums.to(self.device)


        # user_item_matrix = self.interaction_tensor
        # # Convert user_item_matrix to dense if it is sparse
        # if user_item_matrix.is_sparse:
        #     user_item_matrix = user_item_matrix.to_dense()

        # # Perform matrix multiplication
        # cf_score = torch.sparse.mm(user_item_matrix[user].unsqueeze(0), self.item_similarity).squeeze(0)
        # cf_score = torch.mm(user_item_matrix[user], self.item_similarity.T)

        tag_score = self.forward(user).view(-1)
        cf_score = self.cf_score[user].to(self.device)
        # tag_score = torch.rand_like(cf_score) 

        print(f"cf score {cf_score, cf_score.shape}")
        print(f"tag score {tag_score, tag_score.shape}")
        score = self.tag_weight * tag_score + cf_score

        del cf_score
        del tag_score
        torch.cuda.empty_cache()
    
        return score


        # # similarity = torch.mm(user_emb, item_emb.t())
        # # similarity = self.sigmoid(similarity)
        # return predict.view(-1)