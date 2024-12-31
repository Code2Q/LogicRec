import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import (
    TransformerEncoder,
    # MultiHeadCrossAttention,
    VanillaAttention,
)
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

class SASTagRec2(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SASTagRec2, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.feat_hidden_dropout_prob = config["feat_hidden_dropout_prob"]
        self.feat_attn_dropout_prob = config["feat_attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"] if "layer_norm_eps" in config else 1e-12
        self.mode = config["mode"]
        self.tag_col_name = config["tag_col_name"]
        self.infer_mode = config["infer_mode"]
        # size_list = [self.inner_size] + self.hidden_size
        # self.mlp_layers = MLPLayers(size_list, dropout=self.hidden_dropout_prob)

        self.selected_features = config["selected_features"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]
        # self.num_feature_field = len(config["selected_features"])

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.lamda = config["lamda"]
        self.branch_factor = config["branch_factor"]
        self.softmax = nn.Softmax(dim=-1)
        # self.max_item_tag, self.max_user_tag = 7179, 5594
        self.item_feat = dataset.get_item_feature().to(self.device)
        self.ut2it_sp_tensor = torch.load('ut2it_sp.pt').to(self.device)
        self.it2ut_sp_tensor = torch.load('it2ut_sp.pt').to(self.device)
        self.i2it_sp_tensor = torch.load('i2it_sp.pt').to(self.device)
        self.i2ut_sp_tensor = torch.load('i2ut_sp.pt').to(self.device)
        self.max_item_tag, self.max_user_tag = self.it2ut_sp_tensor.shape

        

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.item_tag_embeddings = nn.Embedding(
            self.max_item_tag, self.hidden_size, padding_idx=0
        )
        self.user_tag_embeddings = nn.Embedding(
            self.max_user_tag, self.hidden_size, padding_idx=0
        )
        

        self.item_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        # self.mha_cross_att_layer = MultiHeadCrossAttention(
        #     self.n_heads, 
        #     self.hidden_size, self.hidden_dropout_prob, 
        #     self.attn_dropout_prob, self.layer_norm_eps
        # )

        self.feature_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.feat_hidden_dropout_prob,
            attn_dropout_prob=self.feat_attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.feat_dropout = nn.Dropout(self.feat_hidden_dropout_prob)
        self.prediction_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.tag_loss_fct = nn.BCEWithLogitsLoss()

        self.apply(self._init_weights)
        # self.other_parameter_name = ["feature_embed_layer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, item_tag_id_lists):
        item_emb = self.item_embedding(item_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)
        if self.mode == 'item_tag':
            it_seq_embedding = self.item_tag_embeddings(item_tag_id_lists)
        elif self.mode == 'user_tag':
            it_seq_embedding = self.user_tag_embeddings(item_tag_id_lists)


        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # get item_trm_input
        # item position add position embedding
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)

        feature_emb, attn_weight = self.feature_att_layer(it_seq_embedding)
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.feat_dropout(feature_emb)


        item_trm_output = self.item_trm_encoder(
            item_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )
        item_output = item_trm_output[-1]

        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]

        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)  # [B H]

        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.prediction_layer(output_concat)
        # tag_output = self.tag_prediction_layer(feature_output)
        # print(id_output.shape, tag_output.shape)

        # output = self.LayerNorm(output)
        # seq_output = self.dropout(output)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq] #(50, 15)

        output = self.forward(item_seq, item_seq_len, item_tag_id_lists) #[b, h]
        user = interaction[self.USER_ID]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        target_tags = self.item_feat[self.tag_col_name][pos_items]
        neg_target_tags = self.item_feat[self.tag_col_name][neg_items]

        if self.mode == 'item_tag':
            tag_embds = self.item_tag_embeddings(target_tags)
        elif self.mode == 'user_tag':
            tag_embds = self.user_tag_embeddings(target_tags) #[n_tags, h]

        # if self.loss_type == "BPR":
        #     neg_items = interaction[self.NEG_ITEM_ID]
        #     pos_items_emb = self.item_embedding(pos_items)
        #     neg_items_emb = self.item_embedding(neg_items)
        #     pos_score = torch.sum(output * pos_items_emb, dim=-1)  # [B]
        #     neg_score = torch.sum(output * neg_items_emb, dim=-1)  # [B]
        #     loss = self.loss_fct(pos_score, neg_score)
        #     return loss
        # else:  # self.loss_type = 'CE'
        test_item_emb = self.item_embedding.weight
        
        id_logits = torch.matmul(output, test_item_emb.transpose(0, 1))
        id_loss = self.loss_fct(id_logits, pos_items)

        if self.mode == 'item_tag':
            tag_embds = self.item_tag_embeddings(target_tags)
            neg_tag_embds = self.item_tag_embeddings(neg_target_tags)

        elif self.mode == 'user_tag':
            tag_embds = self.user_tag_embeddings(target_tags) #[n_tags, h]
            neg_tag_embds = self.user_tag_embeddings(neg_target_tags)

    
        tag_logits = torch.mul(output.unsqueeze(1), tag_embds).sum(dim=-1)
        binary_target_tags = torch.ones(tag_logits.shape).to(self.device)
        # [1, 128] [n_tags, 128]
        # print(self.item_embedding(pos_items).shape, tag_embds.shape)
        item_tag_logits = torch.mul(self.item_embedding(pos_items).unsqueeze(1), tag_embds).sum(dim=-1)
        # print(item_tag_logits.shape)

        # Negative tag logits
        neg_tag_logits = torch.mul(output.unsqueeze(1), neg_tag_embds).sum(dim=-1)  # [batch_size, num_negatives, max_tag_len]
        binary_neg_target_tags = torch.zeros(neg_tag_logits.shape).to(self.device)  # Negative tags are 0
        neg_item_tag_logits = torch.mul(self.item_embedding(neg_items).unsqueeze(1), neg_tag_embds).sum(dim=-1)  # [batch_size, num_negatives, max_tag_len]

        # Combine positive and negative tag logits and targets
        combined_tag_logits = torch.cat((tag_logits, neg_tag_logits), dim=0)
        combined_item_tag_logits = torch.cat((item_tag_logits, neg_item_tag_logits), dim=0)
        combined_binary_target_tags = torch.cat((binary_target_tags, binary_neg_target_tags), dim=0)
        # combined_binary_item_tags = torch.cat((item_tag_logits, neg_item_tag_logits), dim=0)

        # Calculate tag loss
        tag_loss = self.tag_loss_fct(combined_tag_logits, combined_binary_target_tags)
        item_tag_loss = self.tag_loss_fct(combined_item_tag_logits, combined_binary_target_tags)
        # mask = (target_tags != 0)
        # # filtered_target_tags = target_tags * mask
        # target_tags = binary_target_tags * mask
        # tag_loss = self.tag_loss_fct(tag_logits, binary_target_tags)
        # binary_target_tags.scatter_(1, filtered_target_tags, mask.float())
            
        # tag_loss = self.tag_loss_fct(tag_logits, binary_target_tags)
        loss = (1-self.lamda) * id_loss + self.lamda * tag_loss + self.lamda * item_tag_loss

        # tag_scores = torch.mul(tag_output.unsqueeze(1), test_item_tag_emb).sum(dim=1)
        print(f"id loss {id_loss}")
        print(f"tag loss {tag_loss}")
        print(f"item tag loss {item_tag_loss}")
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
      
        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq]
            # item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] #(50, 15)

        output = self.forward(item_seq, item_seq_len, item_tag_id_lists) #[b, emb]
        test_item_emb = self.item_embedding(test_item)

        target_tags = self.item_feat[self.tag_col_name][test_item] #[b, max_tags_len]
        if self.mode == 'item_tag':
            # test_item_tag_emb = self.item_tag_embeddings(target_tags) #[b, max_taglen, emb]
            all_tag_emb = self.item_tag_embeddings(target_tags)
        else:
            # test_item_tag_emb = self.user_tag_embeddings(target_tags)
            all_tag_emb = self.user_tag_embeddings(target_tags)
        id_scores = torch.mul(output, test_item_emb).sum(dim=1)  # [B]
        tag_scores = torch.mul(output.unsqueeze(1), all_tag_emb).sum(dim=-1)  #[b, max_tags_len]

        # mask = (target_tags != 0).float()
        # sum_scores = (tag_scores * mask).sum(dim=1)
        sum_tag_scores = tag_scores.sum(dim=1)
        # valid_counts = mask.sum(dim=1)
        average_tag_scores = tag_scores.mean(dim=1)

        # all_tag_scores = all_tag_scores.unsqueeze(1).expand(-1, self.n_items, -1) # batch_size, n_items, n_tags]
        # selected_scores = torch.gather(all_tag_scores, 2, target_tags.unsqueeze(0).expand(id_scores.shape[0], -1, -1))
        # # selected_scores = torch.gather(all_tag_scores, 1, target_tags)
        # # print(all_tag_scores.shape, target_tags.shape)
        # # selected_scores = torch.gather(all_tag_scores, 1, target_tags)
        # mask = (target_tags != 0).float()
        # masked_scores = selected_scores * mask.unsqueeze(0).expand(id_scores.shape[0], -1, -1)
        # sum_scores = masked_scores.sum(dim=2)

        # # mask = (target_tags != 0).float()
        # # masked_scores = selected_scores * mask
        # # sum_scores = masked_scores.sum(dim=1)
        # valid_counts = mask.sum(dim=1)
        # average_tag_scores = sum_scores / valid_counts.clamp(min=1)

        print(f"id scores {id_scores[:10]}, tag scores {average_tag_scores[:10]}")
        if self.infer_mode == 'sum':
            return id_scores + sum_tag_scores
        elif self.infer_mode == 'mean':
            return id_scores + average_tag_scores
        
    def infer_tags(self, all_tag_scores):
        if self.mode == 'user_tag':
             #[b, n_user_tags]
            b = all_tag_scores.shape[0]
            topk_ut_values, topk_ut_indices = torch.topk(all_tag_scores, self.branch_factor, dim=1) #[b, topk_user_tags]
            topk_ut_values = self.softmax(topk_ut_values)
            ut2it = self.ut2it_sp_tensor.to_dense() #【n_user_tags, n_item_tags]
            it2ut = self.it2ut_sp_tensor.to_dense() # [n_item_tags, n_user_tags]

            # 1. 进行 item tag 发散
            # 使用 topk_ut_indices 对 ut2it 进行行索引
            selected_ut2it = ut2it[topk_ut_indices]
            topk_infer_user_tags_expanded = topk_ut_values.unsqueeze(-1) #[b, topk_user_tags, 1]
            aggregated_item_tags = torch.bmm(selected_ut2it.transpose(1, 2), topk_infer_user_tags_expanded).squeeze(-1)  # [b, n_item_tags]
            topk_infer_item_tags, topk_it_indices = torch.topk(aggregated_item_tags, self.branch_factor, dim=-1)  # [b, topk_item_tags]
            topk_infer_item_tags = self.softmax(topk_infer_item_tags)
            # selected_it2ut = it2ut[topk_it_indices]  # [b, topk_item_tags, n_user_tags]
            # topk_infer_item_tags_expanded = topk_infer_item_tags.unsqueeze(-1)  # [b, topk_item_tags, 

            # topk_ut_indices_expanded = topk_ut_indices.unsqueeze(-1).expand(-1, -1, self.max_item_tag) #[b, topk, n_item_tags]
            # ut2it_expanded = ut2it.unsqueeze(0).expand(b, -1, -1) #【b, n_user_tags, n_item_tags]
            # infer_item_tags = torch.gather(ut2it_expanded, 1, topk_ut_indices_expanded) # [b, topk, n_item_tags]
            # infer_item_tags = infer_item_tags.sum(dim=1) # [b, n_item_tags]
            # infer_item_tags = self.softmax(infer_item_tags) # [b, n_item_tags]

            # topk_infer_item_tags, topk_it_indices = torch.topk(infer_item_tags, self.branch_factor, dim=-1) # [b, topk_item_tags]
            selected_it2ut = it2ut[topk_it_indices]  # [b, topk_item_tags, n_user_tags]
            topk_infer_item_tags_expanded = topk_infer_item_tags.unsqueeze(-1)  # [b, topk_item_tags, 1]
            aggregated_user_tags = torch.bmm(selected_it2ut.transpose(1, 2), topk_infer_item_tags_expanded).squeeze(-1)  # [b, n_user_tags]
            # topk_item_indices_expanded = topk_it_indices.unsqueeze(-1).expand(-1, -1, -1, self.max_user_tag)
            # it2ut_expanded = it2ut.unsqueeze(0).unsqueeze(0).expand(all_tag_scores.shape[0], self.branch_factor, -1, -1)
            # selected_it2ut = torch.gather(it2ut_expanded, 2, topk_item_indices_expanded)  # [b, topk_user_tags, topk_item_tags, n_user_tags]

            # # 聚合到 [b, topk_user_tags, topk_user_tags]
            # aggregated_user_tags = torch.sum(selected_it2ut, dim=2)  # [b, topk_user_tags, n_user_tags]

            # # 记录开始的 topk_ut_indices 和发散后的 topk_ut_indices2
            # topk_ut_indices2 = torch.topk(aggregated_user_tags, self.branch_factor, dim=2).indices  # [b, topk_user_tags, topk_user_tags]

            # # 展平 topk_ut_indices2
            # flattened_indices = topk_ut_indices2.view(b, -1).to(self.device)  # 将所有批次和用户标签展平

            # # 创建一个用于存储每个批次中每个用户标签出现次数的张量
            # counts = torch.zeros(b, self.max_user_tag, dtype=torch.float32, device=self.device)

            # # 使用 scatter_add 来计算每个批次中每个用户标签的出现次数
            # counts.scatter_add_(1, flattened_indices, torch.ones_like(flattened_indices, dtype=torch.float32))

            # # 计算频率
            # total_count = flattened_indices.size(1)
            # frequencies = counts / total_count  # [b, n_user_tags]
            # # infered_topk = self.branch_factor ** 2
            # topk_freq_infer, topk_indices_infer = torch.topk(frequencies, self.branch_factor, dim=1)
            topk_freq_infer, topk_indices_infer = torch.topk(aggregated_user_tags, self.branch_factor, dim=1)
            topk_freq_infer = self.softmax(topk_freq_infer)
            # topk_indices_infer = topk_ut_indices[topk_indices_infer]

            return topk_freq_infer, topk_indices_infer
        else:
            topk_values, topk_indices = torch.topk(all_tag_scores, self.branch_factor, dim=0)
            infer_item_tags = self.it2ut_sp_tensor.index_select(0, topk_indices)
            return infer_item_tags



    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq]
            # item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] #(50, 15)

        output = self.forward(item_seq, item_seq_len, item_tag_id_lists)
        test_items_emb = self.item_embedding.weight
        id_scores = torch.matmul(output, test_items_emb.transpose(0, 1))  # [B, n_items]

        target_tags = self.item_feat[self.tag_col_name] #[b, true_tags] [10396, 19]
        if self.mode == 'item_tag':
            # test_item_tag_emb = self.item_tag_embeddings(target_tags) #[b, max_taglen, emb]
            all_tag_emb = self.item_tag_embeddings.weight
        else:
            # test_item_tag_emb = self.user_tag_embeddings(target_tags)
            all_tag_emb = self.user_tag_embeddings.weight
        all_tag_scores = torch.matmul(output, all_tag_emb.transpose(0, 1)) #[b, n_tags]  [4096, 7179]

        infer_tags_freq, infer_tags_indices = self.infer_tags(all_tag_scores) # [b, topk_tags]

        selected_tag_emb = all_tag_emb[infer_tags_indices] #[b, topk_tags, hidden]

        # 使用 matmul 进行内积操作
        # 计算 selected_tag_emb 和 test_items_emb 的内积
        inner_product = torch.matmul(selected_tag_emb, test_items_emb.t())  # [b, topk_tags, n_items]

        # 扩展 infer_tags_freq 以便与 inner_product 匹配
        infer_tags_freq_expanded = infer_tags_freq.unsqueeze(-1)  # [b, topk_tags, 1]
        infer_average_score = (inner_product * infer_tags_freq_expanded).sum(dim=1) # [b, n_items]

        # infer_tags_freq_expanded = infer_tags_freq.unsqueeze(-1)  # [b, topk_tags, 1]
        # weighted_inner_product = inner_product * infer_tags_freq_expanded  # [b, topk_tags, n_items]

        # # 计算加权平均
        # sum_of_weights = infer_tags_freq.sum(dim=1, keepdim=True)  # [b, 1]
        # infer_average_score = weighted_inner_product.sum(dim=1) / sum_of_weights  # [b, n_items, 1]
        # infer_average_score = infer_average_score.squeeze(-1)  # [b, n_items]

       
        all_tag_scores = all_tag_scores.unsqueeze(1).expand(-1, self.n_items, -1) # batch_size, n_items, n_tags]
        # print(infer_tags.shape)
        selected_scores = torch.gather(all_tag_scores, 2, target_tags.unsqueeze(0).expand(id_scores.shape[0], -1, -1))
        # sum_scores = selected_scores.sum(dim=2)

        # if self.infer_mode == 'sum':
        #     print(f"id scores {torch.max(id_scores)}, {torch.min(id_scores)}, {id_scores[:10]}")
        #     print(f"tag scores {torch.max(sum_scores)}, {torch.min(sum_scores)}, {sum_scores[:10]}")
        #     return id_scores + sum_scores
        # elif self.infer_mode == 'mean':
            # valid_counts = mask.sum(dim=1)
            # average_tag_scores = sum_scores / valid_counts.clamp(min=1)     
        average_tag_scores = selected_scores.mean(dim=2)

        print(f"id scores {torch.max(id_scores)}, {torch.min(id_scores)}, {id_scores[:10]}")  # [b, n_items]
        print(f"tag scores {torch.max(average_tag_scores)}, {torch.min(average_tag_scores)}, {average_tag_scores[:10]}")  # [b, n_items]
        print(f"infer tag scores {torch.max(infer_average_score)}, {torch.min(infer_average_score)}, {infer_average_score[:10]}")  # [b, n_items]
        return id_scores + average_tag_scores + infer_average_score

