import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import (
    TransformerEncoder,
    FeatureSeqEmbLayer,
    VanillaAttention,
)
from recbole.model.loss import BPRLoss


class SASTagRecwoid(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SASTagRecwoid, self).__init__(config, dataset)

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

        self.selected_features = config["selected_features"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]
        # self.num_feature_field = len(config["selected_features"])

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.max_item_tag, self.max_user_tag = 7178, 5593
        self.item_feat = dataset.get_item_feature().to(self.device)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.item_tag_embeddings = nn.Embedding(
            self.max_item_tag+1, self.hidden_size, padding_idx=0
        )
        self.user_tag_embeddings = nn.Embedding(
            self.max_user_tag+1, self.hidden_size, padding_idx=0
        )

        # self.feature_embed_layer = FeatureSeqEmbLayer(
        #     dataset,
        #     self.hidden_size,
        #     self.selected_features,
        #     self.pooling_mode,
        #     self.device,
        # )
        

        # self.item_trm_encoder = TransformerEncoder(
        #     n_layers=self.n_layers,
        #     n_heads=self.n_heads,
        #     hidden_size=self.hidden_size,
        #     inner_size=self.inner_size,
        #     hidden_dropout_prob=self.hidden_dropout_prob,
        #     attn_dropout_prob=self.attn_dropout_prob,
        #     hidden_act=self.hidden_act,
        #     layer_norm_eps=self.layer_norm_eps,
        # )

        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        # For simplicity, we use same architecture for item_trm and feature_trm
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
        # self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.feat_dropout = nn.Dropout(self.feat_hidden_dropout_prob)
        # self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
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
        # item_emb = self.item_embedding(item_seq)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # get item_trm_input
        # item position add position embedding
        # item_emb = item_emb + position_embedding
        # item_emb = self.LayerNorm(item_emb)
        # item_trm_input = self.dropout(item_emb)

        # sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        # sparse_embedding = sparse_embedding["item"]
        # dense_embedding = dense_embedding["item"]

        # concat the sparse embedding and float embedding
        # feature_table = []
        # if sparse_embedding is not None:
        #     feature_table.append(sparse_embedding)
        # if dense_embedding is not None:
        #     feature_table.append(dense_embedding)

        
        # feature_table = torch.cat(feature_table, dim=-2)

        
        # weight [batch len num_features]
        # if only one feature, the weight would be 1.0
        if self.mode == 'item_tag':
            it_seq_embedding = self.item_tag_embeddings(item_tag_id_lists)
        elif self.mode == 'user_tag':
            it_seq_embedding = self.user_tag_embeddings(item_tag_id_lists)

        # [batch len num_features hidden_size]
        feature_emb, attn_weight = self.feature_att_layer(it_seq_embedding)
        # feature_emb [batch len hidden]
        # feature position add position embedding
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.feat_dropout(feature_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        # item_trm_output = self.item_trm_encoder(
        #     item_trm_input, extended_attention_mask, output_all_encoded_layers=True
        # )
        # item_output = item_trm_output[-1]

        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]

        # item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)  # [B H]

        # output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        # output = self.dense_layer(feature_output)
        # output = self.LayerNorm(feature_output)
        # seq_output = self.dropout(output)
        return feature_output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq] #(50, 15)

        seq_output = self.forward(item_seq, item_seq_len, item_tag_id_lists)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            tags = self.item_feat[self.tag_col_name]
            if self.mode == 'item_tag':
                tag_embs = self.item_tag_embeddings(tags)
            elif self.mode == 'user_tag':
                tag_embs = self.user_tag_embeddings(tags) #[b, len, h]
            item_tag_emb, attn_weight = self.feature_att_layer(tag_embs)

            # if self.pooling_mode == "max":
            #     item_tag_emb = tag_embs.max(dim=1)[0]
            # elif self.pooling_mode == "sum":
            #     item_tag_emb = tag_embs.sum(dim=1)
            # elif self.pooling_mode == "mean":
            #     item_tag_emb = tag_embs.mean(dim=1)
            
            logits = torch.matmul(seq_output, item_tag_emb.transpose(0, 1))

            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
      
        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq]
            # item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] #(50, 15)

        seq_output = self.forward(item_seq, item_seq_len, item_tag_id_lists)

        tags = self.item_feat[self.tag_col_name][test_item]
        if self.mode == 'item_tag':
            tag_embs = self.item_tag_embeddings(tags)
        elif self.mode == 'user_tag':
            tag_embs = self.user_tag_embeddings(tags) #[b, len, h]
        item_tag_emb, attn_weight = self.feature_att_layer(tag_embs)
        # if self.pooling_mode == "max":
        #     item_tag_emb = tag_embs.max(dim=1)[0]
        # elif self.pooling_mode == "sum":
        #     item_tag_emb = tag_embs.sum(dim=1)
        # elif self.pooling_mode == "mean":
        #     item_tag_emb = tag_embs.mean(dim=1)
        scores = torch.mul(seq_output, item_tag_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_tag_id_lists = self.item_feat[self.tag_col_name][item_seq]
            # item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] #(50, 15)

        seq_output = self.forward(item_seq, item_seq_len, item_tag_id_lists)

        tags = self.item_feat[self.tag_col_name] 
        if self.mode == 'item_tag':
            tag_embs = self.item_tag_embeddings(tags)
        elif self.mode == 'user_tag':
            tag_embs = self.user_tag_embeddings(tags) #[b, len, h]
        item_tag_emb, attn_weight = self.feature_att_layer(tag_embs)
        scores = torch.matmul(seq_output, item_tag_emb.transpose(0, 1))
        return scores
