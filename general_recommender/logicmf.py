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
import torch.nn.functional as F

class LogicMF(GeneralRecommender):
    input_type = InputType.PAIRWISE
    # input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(LogicMF, self).__init__(config, dataset)

        # load dataset info
        self.LABEL = config["LABEL_FIELD"]
        self.RATING = config["RATING_FIELD"]
        self.mode = config['mode']
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reg_weight = config["reg_weight"] if "reg_weight" in config else 1e-05
         # float32 type: the weight decay for l2 normalization
        self.item_bias = config['item_bias'] if "item_bias" in config else 1.0
        self.user_bias = config['item_bias'] if "item_bias" in config else 1.0
        self.use_sigmoid = config['use_sigmoid'] if "use_sigmoid" in config else False
        print(f"use sigmod {self.use_sigmoid}")
        self.use_softmax = config['use_softmax'] if "use_softmax" in config else False
        self.eval_batch_idx = 0
        self.eval_batch_idx_item = 0
        
        # load parameters info
        self.tag_embedding_size = config["tag_embedding_size"] if "tag_embedding_size" in config else 64
        self.item_embedding_size = config["item_embedding_size"] if "item_embedding_size" in config else 64
        self.inter_matrix_type = config["inter_matrix_type"]
        # self.use_branch_factor = config['use_branch_factor'] if "use_branch_factor" in config else False
        self.branch_factor = config['branch_factor'] if "branch_factor" in config else -1
        

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
        # self.max_item_tag, self.max_user_tag = 6386, 4950
        self.max_item_tag, self.max_user_tag = 7179, 5594
        # print(f"==='user_tag_id' {self.max_user_tag, self.min_user_tag},  ==='item_tag_id' {self.max_item_tag, self.min_item_tag}")
        # self.item_tag_sparse, self.user_tag_sparse = self._id_to_sparse_tag()
        
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
        # self.interaction_tensor = self.csr_to_sparse_tensor(self.interaction_matrix)
        
        # self.item_similarity = self.calculate_item_similarity() # dense tensor in cpu
        # print(self.item_similarity)


 
        # self.cf_score = torch.sparse.mm(self.interaction_tensor, self.item_similarity).to_dense().cpu()
        # print(f"cf score {self.cf_score, self.cf_score.shape}")

        # if not os.path.exists("sparse_utid_to_itid.npz"):
        #     self.utid_to_itid = self._build_sparse_utid_to_itid()
        # else:
        #     print("[ load ut2it sparse mat from local npz file ]")
        #     self.utid_to_itid = load_npz("sparse_utid_to_itid.npz")
        # # self.utid_to_itid_norm = self.utid_to_itid.multiply(1 / (self.utid_to_itid.sum(axis=1) + 1e-9))
        # self.utid_to_itid_tensor = self.csr_to_sparse_tensor(self.utid_to_itid).to_dense().to(self.device)
        # # self.utid_to_itid_norm = self.utid_to_itid.multiply(1 / (self.utid_to_itid.sum(axis=1) + 1e-9))
            

        # if not os.path.exists("sparse_itid_to_utid.npz"):
        #     self.itid_to_utid = self._build_sparse_itid_to_utid()
            
        # else:
        #     print("[ load it2ut sparse mat from local npz file ]")
        #     self.itid_to_utid = load_npz("sparse_itid_to_utid.npz")
        # self.itid_to_utid_tensor = self.csr_to_sparse_tensor(self.itid_to_utid).to_dense().to(self.device)
        # # self.itid_to_utid_norm = self.itid_to_utid.multiply(1 / (self.itid_to_utid.sum(axis=1) + 1e-9))
        
        # # self.itid_to_utid_norm = self.itid_to_utid.multiply(1 / (self.itid_to_utid.sum(axis=1) + 1e-9))
        # # self.time_weight = self.time_weight().to(self.device)

        # define layers
        self.item_tag_embeddings = nn.Embedding(self.max_item_tag, self.tag_embedding_size)
        self.user_tag_embeddings = nn.Embedding(self.max_user_tag, self.tag_embedding_size)
        self.item_embeddings = nn.Embedding(self.n_items, self.item_embedding_size)
        self.item_embeddings_cf = nn.Embedding(self.n_items, self.item_embedding_size)
        self.user_embeddings_cf = nn.Embedding(self.n_users, self.item_embedding_size)
        # self.item_bias = nn.Parameter(torch.zeros(self.n_items))
        # self.user_embeddings = nn.Embedding(self.n_users, self.item_embedding_size)
        # self.bias = nn.Parameter(torch.zeros(self.item_embedding_size))

        # self.item_linear = nn.Linear(self.item_embedding_size,  self.item_embedding_size, bias=True)
        # self.user_embeddings = nn.Embedding(self.n_users, self.user_embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.ce_loss = nn.CrossEntropyLoss()

        self.ut2it = torch.load('ut2it_mse.pt').to(self.device).detach()
        self.it2ut = torch.load('it2ut_mse.pt').to(self.device).detach()
        # # 确保 ut2it 和 it2ut 不会计算梯度
        # self.ut2it.requires_grad_(False)
        # self.it2ut.requires_grad_(False)

        # print(self.ut2it, self.it2ut)
        # if self.use_sigmoid:
            # ut2it = torch.mul(self.user_tag_embeddings.weight.unsqueeze(1), self.item_tag_embeddings.weight.unsqueeze(0)).sum(dim=-1)
            # it2ut = torch.mul(self.item_tag_embeddings.weight.unsqueeze(1), self.user_tag_embeddings.weight.unsqueeze(0)).sum(dim=-1)
        # self.ut2it = F.softmax(self.ut2it, dim=1)
        # self.it2ut = F.softmax(self.it2ut, dim=1)
        # self.it2ut = self.sigmoid(self.it2ut)
        # self.ut2it = self.sigmoid(self.ut2it)
        # print(self.ut2it.shape, self.it2ut.shape)

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

        user_tag_id_lists = [tags for tags in self.item_feat['user_tag_id_list']]
        ut_rows, ut_cols, ut_data = [], [], []

        for i, (item_id, tags) in enumerate(zip(item_ids, item_tag_id_lists)):
            
            non_zero_tags = tags[tags > 0]  # 筛选大于0的标签
            true_tags = self.dataset.field2id_token['item_tag_id_list'][non_zero_tags].astype(float).astype(int)
            # print(true_tags)
            # 确保 true_tags 是一个数组或列表
            if np.isscalar(true_tags):
                true_tags = [true_tags]
            # print(true_tags)
            rows.extend([item_id] * len(non_zero_tags))
            cols.extend(true_tags)
            data.extend([1] * len(non_zero_tags))

        iid2itid_csr = sp.csr_matrix((data, (rows, cols)), shape=(self.n_items, self.max_item_tag))


        for i, (ut_id, tags) in enumerate(zip(item_ids, user_tag_id_lists)):
            non_zero_tags = tags[tags > 0]  # 筛选大于0的标签
            true_tags = self.dataset.field2id_token['user_tag_id_list'][non_zero_tags].astype(float).astype(int)
            # 确保 true_tags 是一个数组或列表
            if np.isscalar(true_tags):
                true_tags = [true_tags]
            ut_rows.extend([ut_id] * len(non_zero_tags))
            ut_cols.extend(non_zero_tags)
            ut_data.extend([1] * len(non_zero_tags))

        iid2utid_csr = sp.csr_matrix((ut_data, (ut_rows, ut_cols)), shape=(self.n_items, self.max_user_tag))
        # print(iid2itid_csr.data, iid2utid_csr.data)
        return iid2itid_csr, iid2utid_csr
    
    
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
    
    def topk_branch_tags(self, utids, top_k=200):
        if utids.is_sparse:
            utids = utids.to_dense()
        topk_values, topk_indices = torch.topk(utids, top_k, dim=1)
        mask = torch.zeros_like(utids)
        rows = torch.arange(utids.size(0)).unsqueeze(1).expand(-1, top_k)
        mask[rows, topk_indices] = 1

        # 应用 mask 到 utids
        utids_masked = utids * mask
        return utids_masked


    
    def calculate_tag_score(self, user, item=None):
        # scores = torch.matmul(user_history_vector, self.item_similarity.T)
        # batch_users, num_his_items = user.shape
        user_items = self.history_item_id[user] #[batch_users, max_history_items]
        his_len = self.history_item_len[user] #[batch_users]
        non_zero_indices = torch.nonzero(user_items, as_tuple=True)
        non_zero_user_items = user_items[non_zero_indices].to(self.device)  # [total_history_items]

        item_embeds = self.item_embeddings(non_zero_user_items) #[total_items, embedding_dim] torch.Size([1, 415, 128])
        # item_bias = self.item_bias[non_zero_user_items]
        # item_bias_expanded = item_bias.unsqueeze(-1).expand_as(item_embeds)
        item_tag_embeds = self.item_tag_embeddings.weight  # [num_tags, embedding_dim]
        user_tag_embeds = self.user_tag_embeddings.weight  # [num_tags, embedding_dim]
        del non_zero_user_items

        gamma = torch.tensor(self.gamma, dtype=torch.float32).to(self.device)  # 示例 gamma 值

        lengths = his_len.unsqueeze(1).to(self.device)  # [batch_users, 1]
        exponents = torch.arange(user_items.shape[1], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, max_len]

        weights = gamma ** (lengths - 1 - exponents)  # [batch_users, max_len]
        weights = weights * (exponents < lengths).float()  # 将超出历史长度的部分置为0
        weights = weights[weights != 0]  # [total_history_items]


        lengths = his_len.tolist()  # Convert tensor to list of lengths
        id2it_score = torch.matmul(item_embeds, item_tag_embeds.T)  #[total_items, n_tags]
        # id2it_score += item_bias_expanded
        split_scores = torch.split(id2it_score, lengths) #[n_batch, x, n_tags]
        # if self.use_sigmoid:
        #     split_scores = [torch.sigmoid(scores) for scores in split_scores]
        split_weights = torch.split(weights, lengths)  # ]
        # Sum the scores for each user
        # summed_scores = [torch.sum(scores, dim=0) for scores in split_scores]  # [batch_users, n_tags]
        summed_scores = [torch.sum(scores * weight.unsqueeze(1), dim=0) for scores, weight in zip(split_scores, split_weights)]  # [batch_users, n_tags]
        id2it_score = torch.stack(summed_scores)  # [batch_users, n_tags]

        id2ut_score = torch.matmul(item_embeds, user_tag_embeds.T)
        # id2ut_score += item_bias_expanded
        split_user_scores = torch.split(id2ut_score, lengths)
        summed_user_scores = [torch.sum(scores * weight.unsqueeze(1), dim=0) for scores, weight in zip(split_user_scores, split_weights)]  # [batch_users, n_tags]
        # summed_user_scores = [torch.sum(scores, dim=0) for scores in split_user_scores]
        id2ut_score = torch.stack(summed_user_scores)
        # id2ut_score = self.sigmoid(id2ut_score)
        # id2ut_score_weighted_sum = torch.sum(id2ut_score * weights, dim=1)
        # id2ut_score = self.sigmoid(id2ut_score_weighted_sum)

        # ut2it = torch.mul(self.user_tag_embeddings.weight.unsqueeze(1), self.item_tag_embeddings.weight.unsqueeze(0)).sum(dim=-1)
        # it2ut = torch.mul(self.item_tag_embeddings.weight.unsqueeze(1), self.user_tag_embeddings.weight.unsqueeze(0)).sum(dim=-1)

        if self.use_sigmoid:
            id2it_score = self.sigmoid(id2it_score)
            id2ut_score = self.sigmoid(id2ut_score)
            # ut2it = torch.mul(self.user_tag_embeddings.weight.unsqueeze(1), self.item_tag_embeddings.weight.unsqueeze(0)).sum(dim=-1)
            # it2ut = torch.mul(self.item_tag_embeddings.weight.unsqueeze(1), self.user_tag_embeddings.weight.unsqueeze(0)).sum(dim=-1)
            self.ut2it = self.sigmoid(self.ut2it)
            self.it2ut = self.sigmoid(self.it2ut)
        # if self.use_normalize:
        #     id2ut_score = F.normalize(id2ut_score, p=2, dim=1)
        #     id2it_score = F.normalize(id2it_score, p=2, dim=1)
            # self.ut2it = F.normalize(self.ut2it, p=2, dim=1)
            # self.it2ut = F.normalize(self.it2ut, p=2, dim=1)

        # if self.use_softmax:
        #     id2it_score = F.softmax(id2it_score, dim=1)
        #     id2ut_score = F.softmax(id2ut_score, dim=1)

        
        propogated_tags = []
    
        for i in range(1, self.logic_step+1):
            if self.mode == 'item_tag':
                propagated_tag_step = self._calculate_item_tag_score(i, id2it_score, id2ut_score)
            elif self.mode == 'user_tag':
                propagated_tag_step = self._calculate_user_tag_score(i, id2it_score, id2ut_score)
            # propagated_tag_step = torch.log1p(propagated_tag_step)

            # if self.use_sigmoid:
            #     propagated_tag_step = self.sigmoid(propagated_tag_step)

            # if self.use_softmax:
            #     propagated_tag_step = F.softmax(propagated_tag_step, dim=1)
            print(f"propagated_tag_step{i}: {torch.max(propagated_tag_step), torch.min(propagated_tag_step)}")
            propogated_tags.append(propagated_tag_step)
        propogated_tags_tensor = torch.stack(propogated_tags, dim=0) #[n_step, batch_size, n_tag]
        weights = torch.tensor([self.tag_decay ** i for i in range(propogated_tags_tensor.shape[0])], device=self.device) #[1, n_step]
        weights = weights.view(-1, 1, 1)  # 调整形状为 (n_step, 1, 1)
        # weighted_propagated_tags =  propogated_tags_tensor * weights#[n_step, batch_size, n_tag]
        # weighted_propagated_tags = torch.sum(propogated_tags_tensor * weights, dim=0) #[batch_size, n_tag]
        # weighted_propagated_tags = torch.log1p(torch.sum(weighted_propagated_tags, dim=0))

        log_propagated_tags = sum([w * t for w, t in zip(weights, propogated_tags)]) #[1, n_tags]
        # print(f"log propogated shape {log_propagated_tags.shape}")
        # log_propagated_tags = sum([t for t in  propogated_tags]) #([batch, 6386])
        # non_zero_counts = torch.count_nonzero(log_propagated_tags, dim=1)
        # for i, count in enumerate(non_zero_counts):
        #     print(f"Row {i} has {count.item()} non-zero elements.")

        if item is not None:
            item_emb = self.item_embeddings(item) #[n_batch, 128] #[n_tags, 128]
            if self.mode == 'item_tag':
                next_item_tag_score = torch.matmul(item_emb, self.item_tag_embeddings.weight.transpose(0, 1)) #[batch_size, n_tags]

                # if self.use_sigmoid:
                #     next_item_tag_score = self.sigmoid(next_item_tag_score)
                all_scores = torch.mul(log_propagated_tags, next_item_tag_score).sum(dim=1) #[n_batch]
                # cosine_sim = F.cosine_similarity(log_propagated_tags, next_item_tag_score, dim=1)
                # all_scores = torch.mul(next_item_tag_score, propogated_tags_tensor).sum(dim=1) #[n_batch, 1]
                # all_scores = self.sigmoid(all_scores)
                # all_scores = torch.mul(next_item_tag_score, log_propagated_tags).sum(dim=1) #[n_batch, 1]
                # 使用 view 方法调整形状
                # all_scores = self.sigmoid(all_scores)
            elif self.mode == 'user_tag':
                next_user_tag_score = torch.matmul(item_emb, self.user_tag_embeddings.weight.transpose(0, 1))
                # if self.use_sigmoid:
                #     next_user_tag_score = self.sigmoid(next_user_tag_score)
                # cosine_sim = F.cosine_similarity(log_propagated_tags, next_user_tag_score, dim=1)
                
                all_scores = torch.mul(log_propagated_tags, next_user_tag_score).sum(dim=-1)
            # cf_score = torch.sum(self.item_similarity[user_items] )

        elif item is None:
            # self.save_propagated_tags(propogated_tags_tensor, '/home/liushuchang/.jupyter/yuqing_workspace/recbole/saved/logicmf_propogated_tags_woit_sigmd')
            if self.mode == 'item_tag':
                tag_score = torch.matmul(self.item_embeddings.weight, self.item_tag_embeddings.weight.transpose(0, 1))#[n_items, n_tags]
            elif self.mode == 'user_tag':
                tag_score = torch.matmul(self.item_embeddings.weight, self.user_tag_embeddings.weight.transpose(0, 1))
            # if self.use_sigmoid:
            #     tag_score = self.sigmoid(tag_score) #[n_items, n_tags]

            tag_matrix_expanded = log_propagated_tags.unsqueeze(1)  # [batch_users, 1, n_tag]
            candidate_tag_matrix_expanded = tag_score.unsqueeze(0)  # [1, num_items, n_tag]
            all_scores = tag_matrix_expanded * candidate_tag_matrix_expanded  # [batch_users, num_items, n_tag]
            # cosine_sim = F.cosine_similarity(tag_matrix_expanded, candidate_tag_matrix_expanded, dim=2)

            # print(f"all scores multi shape {all_scores.shape}, tag_score {tag_score.shape}, ")
            #torch.Size([1, 1, 10396, 4950]),
            all_scores = all_scores.sum(dim=-1) # [batch_users, num_items]
            # if self.eval_batch_idx_item == 0:
                # self.save_final_scores(tag_score, '/home/liushuchang/.jupyter/yuqing_workspace/recbole/saved/logicmf_candidate_scores_woitd_sigmd')
        # if self.use_sigmoid:
        # all_scores = self.sigmoid(all_scores)
        torch.cuda.empty_cache()
        return all_scores
    
    def save_propagated_tags(self, tensor, path):
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = os.path.join(path, f'propagated_{self.mode}_bth_{self.eval_batch_idx}.pt')
        torch.save(tensor, file_name)
        print(f'Saved {file_name}')
        self.eval_batch_idx += 1

    def save_final_scores(self, scores, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, f'item_tagscore_{self.mode}.pt')
        torch.save(scores, filename)  # 使用 numpy.savez 保存密集矩阵
        print(f"Saved {filename}")
        self.eval_batch_idx_item += 1


    def _calculate_item_tag_score(self, logic_step, id2it_score, id2ut_score):
        id2it_score = id2it_score / id2it_score.sum(dim=1, keepdim=True)
        id2it_score = self.sigmoid(id2it_score)
        id2ut_score = id2ut_score / id2ut_score.sum(dim=1, keepdim=True)
        id2ut_score = self.sigmoid(id2ut_score)

        if logic_step % 2 == 1:
            itids = id2it_score #[1, n_tags]
            # itids = F.normalize(itids, p=2, dim=1)
            if self.branch_factor != -1:
                        itids = self.topk_branch_tags(itids, self.branch_factor)

            if logic_step > 1:
                iter_num = self.logic_step // 2
                # propagated_user_tags = (user_histories @ self.user_tag_sparse_norm)
                for i in range(iter_num):
                    utids = torch.mm(itids, self.it2ut)
                    utids = utids / utids.sum(dim=1, keepdim=True)
                    utids = self.sigmoid(utids)
                    # utids = F.normalize(utids, p=2, dim=1)
                    # propagated_user_tags = torch.log(self.user_bias + utids)
                    if self.branch_factor != -1:
                        utids = self.topk_branch_tags(utids, self.branch_factor)
                    # propagated_user_tags = torch.log(self.user_bias + utids)
                    # propagated_user_tags = torch.clamp_min(propagated_user_tags, 0)
                    # propagated_user_tags /= torch.norm(propagated_user_tags, p=1, dim=1, keepdim=True)

                    itids = torch.mm(utids, self.ut2it)
                    itids = itids / itids.sum(dim=1, keepdim=True)
                    itids = self.sigmoid(itids)
                    # itids = F.normalize(itids, p=2, dim=1)
                    if self.branch_factor != -1:
                        itids = self.topk_branch_tags(itids, self.branch_factor)
                    # propagated_item_tags = torch.log(self.item_bias + itids)
                    # propagated_item_tags = torch.clamp_min(propagated_item_tags, 0)
                    # propagated_item_tags /= torch.norm(propagated_item_tags, p=1, dim=1, keepdim=True)
                    # propagated_item_tags = sp.csr_matrix(normalize(propagated_item_tags, norm='l1', axis=1))
        
        elif logic_step % 2 == 0:
            # print(id2ut_score.shape, type(id2ut_score)) #torch.Size([4096, 4950])
            # print(self.utid_to_itid_tensor.shape, type(self.utid_to_itid_tensor)) #torch.Size([4950, 6386]) 
    
            itids = torch.mm(id2ut_score, self.ut2it) 
            itids = itids / itids.sum(dim=1, keepdim=True)
            itids = self.sigmoid(itids)
            # itids = F.softmax(itids, dim=1)
            # itids = F.normalize(itids, p=2, dim=1)
            if self.branch_factor != -1:
                        itids = self.topk_branch_tags(itids, self.branch_factor)
            if logic_step > 2:
                iter_num = self.logic_step // 2
                for i in range(iter_num-1):
                    # 计算 uid2utid
                    utids = torch.mm(itids, self.it2ut)
                    utids = utids / utids.sum(dim=1, keepdim=True)
                    utids = self.sigmoid(utids)
                    # utids = F.softmax(utids, dim=1)
                    # utids = F.normalize(utids, p=2, dim=1)
                    if self.branch_factor != -1:
                        utids = self.topk_branch_tags(utids, self.branch_factor)
                    itids = torch.mm(utids, self.ut2it)
                    itids = itids / itids.sum(dim=1, keepdim=True)
                    itids = self.sigmoid(itids)
                    # itids = F.softmax(itids, dim=1)
                    # itids = F.normalize(itids, p=2, dim=1)
                    if self.branch_factor != -1:
                        itids = self.topk_branch_tags(itids, self.branch_factor)

        return itids
    
    def _calculate_user_tag_score(self, logic_step, id2it_score, id2ut_score):
        # id2it_score = id2it_score / id2it_score.sum(dim=1, keepdim=True)
        # id2it_score = self.sigmoid(id2it_score)
        # id2ut_score = id2ut_score / id2ut_score.sum(dim=1, keepdim=True)
        # id2ut_score = self.sigmoid(id2ut_score)

        if logic_step % 2 == 1:
            utids = id2ut_score #[1, n_user_tags]
            # utids = F.normalize(utids, p=2, dim=1)
            if self.branch_factor != -1:
                        utids = self.topk_branch_tags(utids, self.branch_factor)
            # propagated_user_tags = torch.log(self.user_bias + propagated_user_tags)
            # propagated_user_tags = torch.clamp_min(propagated_user_tags, 0)
            # count = self.count_greater_than_threshold(propagated_user_tags, 0.1)
            # print(f"logic step == 1 每一行中元素大于 0.1 的个数: {count}")
            # count = self.count_greater_than_threshold(propagated_user_tags, 0.3)
            # print(f"logic step == 1 每一行中元素大于 0.3 的个数: {count}")

            if logic_step > 1:
                iter_num = self.logic_step // 2
                # propagated_item_tags = (user_histories @ self.user_tag_sparse_norm)
                for i in range(iter_num):
                    itids = torch.matmul(utids, self.ut2it)
                    itids = itids / itids.sum(dim=1, keepdim=True)
                    # itids = self.sigmoid(itids)
                    # itids = F.normalize(itids, p=2, dim=1)
                    # itids = torch.matmul(utids, self.utid_to_itid_tensor)
                    # itids = torch.log1p(itids)
                    if self.branch_factor != -1:
                        itids = self.topk_branch_tags(itids, self.branch_factor)
                    # utids = torch.matmul(itids, self.itid_to_utid_tensor)
                    utids = torch.matmul(itids, self.it2ut)
                    utids = utids / utids.sum(dim=1, keepdim=True)
                    # utids = self.sigmoid(utids)
                    # utids = F.normalize(utids, p=2, dim=1)
                    # utids = torch.log1p(utids)
                    if self.branch_factor != -1:
                        utids = self.topk_branch_tags(utids, self.branch_factor)
         
        elif logic_step % 2 == 0:
            # print(id2it_score.shape, self.itid_to_utid_tensor.shape)
            utids = torch.matmul(id2it_score, self.it2ut)
            # utids = utids / utids.sum(dim=1, keepdim=True)
            utids = self.sigmoid(utids)
            # utids = utids / utids.sum(dim=1, keepdim=True)
            # utids = torch.matmul(id2it_score, self.itid_to_utid_tensor) 
            # utids = F.normalize(utids, p=2, dim=1)
            # utids = F.softmax(utids, dim=1)
            if self.branch_factor != -1:
                        utids = self.topk_branch_tags(utids, self.branch_factor)
            if logic_step > 2:
                iter_num = self.logic_step // 2
                for i in range(iter_num-1):
                    itids = torch.matmul(utids,self.ut2it)
                    # itids = itids / itids.sum(dim=1, keepdim=True)
                    # itids = self.sigmoid(itids)
                    # itids = F.softmax(itids, dim=1)
                    # itids = F.normalize(itids, p=2, dim=1)
                    # itids = torch.matmul(utids, self.utid_to_itid_tensor)
                    # itids = torch.log1p(itids)
                    if self.branch_factor != -1:
                        utids = self.topk_branch_tags(utids, self.branch_factor)
                    utids = torch.matmul(itids, self.it2ut)
                    # utids = utids / utids.sum(dim=1, keepdim=True)
                    # utids = self.sigmoid(utids)
                    # utids = F.softmax(utids, dim=1)
                    # utids = F.normalize(utids, p=2, dim=1)
                    # utids = torch.matmul(itids, self.itid_to_utid_tensor)
                    # utids = torch.log1p(utids)
                    if self.branch_factor != -1:
                        utids = self.topk_branch_tags(utids, self.branch_factor)

        return utids

    def _init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)

    def forward(self, user, item=None):
        # pass
        tag_score = self.calculate_tag_score(user, item)
        it_score = None
        if item is not None:
            user_emb = self.user_embeddings_cf(user)
            item_emb = self.item_embeddings_cf(item)
            it_score = torch.mul(user_emb, item_emb).sum(dim=1)
        return tag_score, it_score
        # return tag_score

    def calculate_loss(self, interaction):
        # return torch.nn.Parameter(torch.zeros(1))
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        # if self.inter_matrix_type == "01":
        #     label = interaction[self.LABEL]
        # elif self.inter_matrix_type == "rating":
        #     label = interaction[self.RATING] * interaction[self.LABEL]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_tag_score, pos_it_score = self.forward(user, pos_item)
        # score = (self.tag_weight * pos_tag_score + pos_it_score).view(-1)
        neg_tag_score, neg_it_score = self.forward(user, neg_item)

        print(f"pos neg tag score {pos_tag_score[:5], neg_tag_score[:5]}")
        print(f"pos neg interaction score {pos_it_score[:5], neg_it_score[:5]}")
        # print(f"pos neg it score {}")
        # loss = self.bce_loss(score, label)
        # it_loss = self.bce_loss(pos_it_score, label)
        # print(f"rec loss {loss}")

        mf_loss = self.bpr_loss(pos_tag_score, neg_tag_score)
        it_loss = self.bpr_loss(pos_it_score, neg_it_score)
        print(f"tag loss {mf_loss}")
        print(f"iteraction loss {it_loss}")

        # # calculate regularization Loss

        user_tag_embeddings = self.user_tag_embeddings.weight
        item_tag_embeddings = self.item_tag_embeddings.weight
        pos_ego_embeddings = self.item_embeddings(pos_item)
        neg_ego_embeddings = self.item_embeddings(neg_item)
        pos_item_embeddings = self.item_embeddings_cf(pos_item)
        neg_item_embeddings = self.item_embeddings_cf(neg_item)
        
        user_embeddings = self.user_embeddings_cf(user)

        reg_loss = self.reg_loss(
            user_tag_embeddings,
            item_tag_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            pos_item_embeddings,
            neg_item_embeddings,
            user_embeddings,
            require_pow=False,
        )
        print(f"reg loss {reg_loss}")
        # loss_tag = self.bce_loss(pos_tag_score, label)
        # loss_it = self.bce_loss(pos_it_score, label)
        # print(f"tag loss {loss}")
        # print(f"iteraction loss {it_loss}")
        
        # loss = loss + self.reg_weight * reg_loss
        loss =  self.tag_weight * mf_loss + it_loss + self.reg_weight * reg_loss
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # id2it_score, id2ut_score = self.forward(user, item)
        tag_score, cf_score = self.forward(user, item)
        predict = self.tag_weight * tag_score + self.sigmoid(cf_score)
        # predict = self.sigmoid(score)
        return predict

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        tag_score, _ = self.forward(user)
        # tag_score = self.sigmoid(tag_score.view(-1))
        
        # cf_score = self.cf_score[user].to(self.device)
        # tag_score = torch.rand_like(cf_score) 

        # print(f"cf score {cf_score, cf_score.shape}")


        user_emb = self.user_embeddings_cf(user)
        item_emb = self.item_embeddings_cf.weight
        similarity = torch.mm(user_emb, item_emb.t())
        score = self.tag_weight * tag_score + similarity
        print(f"tag score it score {tag_score, similarity}")
        # print(".....")
        # print(tag_score.shape, similarity.shape)
        # score = self.sigmoid(similarity) + self.tag_weight * tag_score
        score = self.sigmoid(score)
        
        torch.cuda.empty_cache()
    
        # return self.tag_weight * tag_score + similarity
        return score


        # # similarity = torch.mm(user_emb, item_emb.t())
        # # similarity = self.sigmoid(similarity)
        # return predict.view(-1)