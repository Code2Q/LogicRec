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

class LogicMF(GeneralRecommender):
    r"""MF is a traditional matrix factorization model.
    The original interaction matrix of :math:`n_{users} \times n_{items}` is set as model input,
    we carefully design the data interface and use sparse tensor to train and test efficiently.
    """

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
        self.user_embedding_size = config["user_embedding_size"] if "item_embedding_size" in config else 64
        self.inter_matrix_type = config["inter_matrix_type"]

        self.item_feat = dataset.get_item_feature().to(self.device)
        print("---item to tag features loaded---")

        # print(self.item_feat['user_tag_id_list'],self.item_feat['user_tag_id_list'].shape)
        # print(self.item_feat['item_tag_id_list'], self.item_feat['item_tag_id_list'].shape)

        self.utid2itid = dataset.utid2itid_feat
        self.itid2utid = dataset.itid2utid_feat
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.gamma = config["gamma"]
        self.tag_decay = config['tag_decay']


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
        self.interaction_sp_tensor = self.csr_to_sparse_tensor(self.interaction_matrix).to(self.device)
        self.max_rating = self.history_user_value.max()
        self.item_similarity = self.calculate_item_similarity()

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
        self.time_weight = self.time_weight().to(self.device)

        # define layers
        self.item_tag_embeddings = nn.Embedding(self.max_item_tag, self.tag_embedding_size)
        self.user_tag_embeddings = nn.Embedding(self.max_user_tag, self.tag_embedding_size)
        self.item_embeddings = nn.Embedding(self.n_items, self.item_embedding_size)
        self.user_embeddings = nn.Embedding(self.n_users, self.user_embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

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
        item_ids = self.item_feat['item_id'].cpu().numpy()
        item_tag_id_lists = [tags.cpu().numpy() for tags in self.item_feat['item_tag_id_list']]
        
        rows, cols, data = [], [], []
        # item_tag_max = max([tags.max() for tags in item_tag_id_lists])
        # item_tag_min = min([tags.min() for tags in item_tag_id_lists])
        # print(f"Item tag ID lists - Max: {item_tag_max}, Min: {item_tag_min}")

        user_tag_id_lists = [tags.cpu().numpy() for tags in self.item_feat['user_tag_id_list']]
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
    
    def _build_sparse_utid_to_itid(self):
        ut_ids = torch.tensor([tag for tag in self.utid2itid['user_tag_id']], device=self.device).cpu().numpy()
        item_tag_id_lists = [tags.cpu().numpy() for tags in self.utid2itid['item_tag_id_list']]
        rows, cols, data = [], [], []
        
        for i, (utid, tags) in enumerate(zip(ut_ids, item_tag_id_lists)):
            # 筛选大于0的标签
            non_zero_tags = tags[tags > 0]
            for tag in non_zero_tags:
                if utid <= self.max_user_tag and tag <= self.max_item_tag:
                    rows.append(utid)
                    cols.append(tag)
                    data.append(1)

        sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.max_user_tag, self.max_item_tag))
        print("[[build ut2it sparse_mat from scratch]]")
        sp.save_npz('sparse_utid_to_itid.npz', sparse_matrix)
        return sparse_matrix

    def _build_sparse_itid_to_utid(self):
        it_ids = torch.tensor([tag for tag in self.itid2utid['item_tag_id']], device=self.device).cpu().numpy()
        user_tag_id_lists = [tags.cpu().numpy() for tags in self.itid2utid['user_tag_id_list']]
        rows, cols, data = [], [], []
        for i, (itid, tags) in enumerate(zip(it_ids, user_tag_id_lists)):
            # 筛选大于0的标签
            non_zero_tags = tags[tags > 0]
            for tag in non_zero_tags:
                if 0 <= itid < self.max_item_tag and 0 <= tag < self.max_user_tag:
                    rows.append(itid)
                    cols.append(tag)
                    data.append(1)

        sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.max_item_tag, self.max_user_tag))
        print("[[build it2ut sparse_mat from scratch]]")
        sp.save_npz('sparse_itid_to_utid.npz', sparse_matrix)

        return sparse_matrix
    
    def calculate_item_similarity(self):

        item_user_matrix = self.get_sparse_mat().transpose(0, 1).to(self.device)
        item_degrees = torch.sparse.sum(item_user_matrix, dim=1).to_dense() + 1e-8
        co_occur = torch.sparse.mm(item_user_matrix, item_user_matrix.t()).to_dense()
        
        similarity_matrix = co_occur / torch.sqrt(torch.ger(item_degrees, item_degrees))    
        # Create a mask to set the diagonal elements to 0
        mask = torch.eye(similarity_matrix.size(0), device=self.device)
        similarity_matrix = torch.where(mask == 1, torch.tensor(0.0, device=self.device), similarity_matrix)

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
        user_items = self.history_item_id[user].to(self.device) #[batch_users, max_history_items]
        his_len = self.history_item_len[user].to(self.device) #[batch_users]
        non_zero_indices = torch.nonzero(user_items, as_tuple=True)
        non_zero_user_items = user_items[non_zero_indices] # [total_history_items]
        # print(f"non zero items {non_zero_user_items}")

        # his_len = self.history_item_len[user].to(self.device) #[batch_users, 1]
        item_embeds = self.item_embeddings(non_zero_user_items) #[total_items, embedding_dim] torch.Size([1, 415, 128])
        item_tag_embeds = self.item_tag_embeddings.weight  # [num_tags, embedding_dim]
        user_tag_embeds = self.user_tag_embeddings.weight  # [num_tags, embedding_dim]
        weights = self.time_weight[user]  # [batch_users, max_history_items]
        # item_tag_embeds.shape = [n_tags, embedding_dim]

        id2it_score = torch.matmul(item_embeds, item_tag_embeds.T)  #[total_items, n_tags]

        # id2it_score = self.sigmoid(id2it_score)
        # Split id2it_score based on his_len
        lengths = his_len.tolist()  # Convert tensor to list of lengths
        split_scores = torch.split(id2it_score, lengths) #[n_batch, x, n_tags]
        # Sum the scores for each user
        summed_scores = [torch.sum(scores, dim=0) for scores in split_scores]  # [batch_users, n_tags]
        id2it_score = torch.stack(summed_scores)  # [batch_users, n_tags]
        id2it_score = self.sigmoid(id2it_score)
        # print(f" id 2 it score {id2it_score.shape}")

        # id2it_score = torch.sum(split_scores, dim=1) 
        # id2it_score_weighted_sum = torch.sum(id2it_score * weights, dim=1) 
        # id2it_score = self.sigmoid(id2it_score_weighted_sum)
        id2ut_score = torch.matmul(item_embeds, user_tag_embeds.T)
        id2ut_score = self.sigmoid(id2ut_score)
        split_user_scores = torch.split(id2ut_score, lengths)
        summed_user_scores = [torch.sum(scores, dim=0) for scores in split_user_scores]
        id2ut_score = torch.stack(summed_user_scores)
        id2ut_score = self.sigmoid(id2ut_score)
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
                all_scores = self.sigmoid(all_scores)
                # all_scores = torch.mul(next_item_tag_score, log_propagated_tags).sum(dim=1) #[n_batch, 1]
                # 使用 view 方法调整形状
                # all_scores = self.sigmoid(all_scores)
                # cf_scores = torch.sum(self.item_similarity[user_items] )
            elif self.mode == 'user_tag':
                item_emb = self.item_embeddings(item).unsqueeze(1)
                next_user_tag_score = torch.mul(item_emb, self.user_tag_embeddings.weight).sum(dim=2)
                all_scores = torch.mul(next_user_tag_score, log_propagated_tags).sum(dim=1)

            non_zero_user_items
            # cf_score = torch.sum(self.item_similarity[user_items] )

        elif item is None:
            item_emb = self.item_embeddings.weight #[n_items, 128]
            # item_tag_score = torch.mul(item_emb, self.item_tag_embeddings.weight).sum(dim=2)
            item_tag_score = torch.matmul(item_emb, self.item_tag_embeddings.weight.transpose(0, 1))#[n_items, n_tags]
            # item_tag_score = torch.clamp_min(item_tag_score, 0)
            # item_tag_score = self.sigmoid(item_tag_score)
            all_scores = torch.mul(item_tag_score, log_propagated_tags).sum(dim=1, keepdim=True)  #[n_items]
            all_scores = self.sigmoid(all_scores)
            # all_scores = self.sigmoid(all_scores)
        return all_scores




    def _init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def forward(self, user, item=None):
        # start_time = time.time()
        user_emb = self.user_embeddings(user)
        tag_score = self.calculate_tag_score(user, item)
        if item is not None:
            item_emb = self.item_embeddings(item)
            score = torch.mul(user_emb, item_emb).sum(dim=1)
        else:
            item_emb = self.item_embeddings.weight
            score = torch.mm(user_emb, item_emb.transpose(0, 1))
        # end_time = time.time()
        # print(f"Calculate tag score time: {start_time - end_time:.6f} seconds")
        # return vector
        return score, tag_score

    def calculate_loss(self, interaction):
        # start_time = time.time()
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        # print(f"user shape in calcualte loss {user.shape}")
        pos_score, pos_item_score = self.forward(user, pos_item)
        neg_score, neg_item_score = self.forward(user, neg_item)

        # end_time = time.time()
        print(f"pos neg item score from tag {pos_item_score, neg_item_score}")
        print(f"pos neg item score from bpr {pos_score, neg_score}")
        mf_loss = self.bpr_loss(pos_item_score, neg_item_score) + self.bpr_loss(pos_score, neg_score)

        # print(f"Calculate loss time: {start_time - end_time:.6f} seconds")
        print(f"mf loss {mf_loss, type(mf_loss)}")

        # # calculate regularization Loss

        # user_tag_embeddings = self.user_tag_embeddings.weight
        # item_tag_embeddings = self.item_tag_embeddings.weight
        pos_ego_embeddings = self.item_embeddings(pos_item)
        neg_ego_embeddings = self.item_embeddings(neg_item)

        reg_loss = self.reg_loss(
            # user_tag_embeddings,
            # item_tag_embeddings,
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
        vector, predict = self.forward(user, item)
        return predict

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        score, tag_score = self.forward(user, item=None)
        return score + tag_score


        # # similarity = torch.mm(user_emb, item_emb.t())
        # # similarity = self.sigmoid(similarity)
        # return predict.view(-1)