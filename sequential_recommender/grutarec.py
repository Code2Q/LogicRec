import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import VanillaAttention

# class VanillaAttention(nn.Module):
#     """
#     Vanilla attention layer is implemented by linear layer.

#     Args:
#         input_tensor (torch.Tensor): the input of the attention layer

#     Returns:
#         hidden_states (torch.Tensor): the outputs of the attention layer
#         weights (torch.Tensor): the attention weights

#     """

#     def __init__(self, hidden_dim, attn_dim):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1)
#         )

#     def forward(self, input_tensor):
#         # (B, Len, num, H) -> (B, Len, num, 1)
#         energy = self.projection(input_tensor)
#         weights = torch.softmax(energy.squeeze(-1), dim=-1)
#         # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
#         hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
#         return hidden_states, weights
    
class GRUTARec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(GRUTARec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.tag_dropout_prob = config["tag_dropout_prob"]
        self.lamda = config["lamda"] if "lamda" in config else 1
        self.pooling_mode = config["pooling_mode"]
        self.mode = config["mode"]

        self.max_item_tag, self.max_user_tag = 7178, 5593
        self.item_feat = dataset.get_item_feature().to(self.device)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.item_tag_embeddings = nn.Embedding(
            self.max_item_tag+1, self.embedding_size, padding_idx=0
        )
        self.user_tag_embeddings = nn.Embedding(
            self.max_user_tag+1, self.embedding_size, padding_idx=0
        )

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.tag_dropout = nn.Dropout(self.tag_dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.tag_gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.feature_att_layer = VanillaAttention(self.embedding_size, self.embedding_size)

        # self.dense_layer = nn.Linear(self.hidden_size * 2, self.embedding_size * 2)
        self.dense_layer = nn.Linear(self.hidden_size * 2, self.embedding_size)
        self.tag_dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def pad_to_length(self, tensor, length, pad_value=0):
        if tensor.size(0) < length:
            padding = torch.full((length - tensor.size(0),), pad_value, dtype=tensor.dtype).to(self.device)
            tensor = torch.cat([tensor, padding], dim=0)
        else:
            tensor = tensor[-length:]
        return tensor
    
    # def tag_forward(self, item_seq, item_seq_len):
    #     item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] # (batch_size, item_seq_len, item_tag_len)
    #     # print(f"item_tag_id_lists {item_tag_id_lists.shape}") # torch.Size([1, 100, 19]) # (512, 19)
    #     last_column = item_tag_id_lists[:, :, -1]  # 形状为 (batch_size, item_seq_len)
    #     non_zero_elements = last_column[last_column != 0]

    #     item_tag_id_lists = self.pad_to_length(non_zero_elements, 200).unsqueeze(0) # 构建新的张量，形状为 (1, 200)

    #     # print(f"New item_tag_id_lists Shape: {item_tag_id_lists.shape}")  # torch.Size([1, 200])
    #     # user_tag_id_lists = self.item_feat['user_tag_id_list'][item_seq]
    #     # valid_ut = user_tag_id_lists.nonzero(as_tuple=True)
    #     # user_tag_id_lists = user_tag_id_lists[valid_ut]
    #     # user_tag_id_lists = self.pad_to_length(user_tag_id_lists, 200, 0)
    #     it_seq_embedding = self.item_tag_embeddings(item_tag_id_lists) # ([1, 200, 64]) 
        

    #     # ut_seq_emb = self.user_tag_embeddings(user_tag_id_lists)
    #     it_seq_emb_dropout = self.emb_dropout(it_seq_embedding)
    #     # ut_seq_emb_dropout = self.emb_dropout(ut_seq_emb)
    #     it_gru_output, _ = self.tag_gru_layers(it_seq_emb_dropout)
    #     # ut_gru_output, _ = self.tag_gru_layers(ut_seq_emb_dropout)


    #     it_gru_output = self.tag_dense(it_gru_output)
    #     # print(f"it gru output {it_gru_output, it_gru_output.shape}") # torch.Size([1, 200, 64]))
    #     # ut_gru_output = self.tag_dense(ut_gru_output)
    #     it_seq_output = self.gather_indexes(it_gru_output, torch.tensor([200-1]).to(self.device))
    #     # ut_seq_output = self.gather_indexes(ut_gru_output, valid_ut[0])
    #     return it_seq_output



    def forward(self, item_seq, item_seq_len, item_tag_id_lists):
        item_seq_emb = self.item_embedding(item_seq).to(self.device) # (batch_size, item_seq_len, embedding_size)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        item_gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        # print(f"item seq emb {item_seq_emb_dropout.shape}") #torch.Size([1, 50, 64]) 

        
        # item_tag_embeds = self.item_tag_embeddings(item_tag_id_lists) # (50, 15, 64)
        # last_column = item_tag_id_lists[:, :, -1]  # 形状为 (batch_size, item_seq_len)
        # non_zero_elements = last_column[last_column != 0]
        # item_tag_id_lists = self.pad_to_length(non_zero_elements, 200).unsqueeze(0) # 构建新的张量，形状为 (1, 200)
        if self.mode == 'item_tag':
            it_seq_embedding = self.item_tag_embeddings(item_tag_id_lists)
        elif self.mode == 'user_tag':
            it_seq_embedding = self.user_tag_embeddings(item_tag_id_lists)

        # [batch len num_features hidden_size]
        feature_emb, attn_weight = self.feature_att_layer(it_seq_embedding)


        # if self.pooling_mode == "max":
        #     it_seq_embedding = torch.max(it_seq_embedding, dim=-2)[0]

        # elif self.pooling_mode == "mean":
        #     it_seq_embedding = torch.mean(it_seq_embedding, dim=-2)

        # elif self.pooling_mode == "sum":
        #     it_seq_embedding = torch.sum(it_seq_embedding, dim=-2)

        # print(f"item tag seq embedding pooling {it_seq_embedding.shape}") #torch.Size([1, 200, 64]) torch.Size([1, 50, 19, 64])  

        it_seq_emb_dropout = self.tag_dropout(feature_emb)
        # ut_seq_emb_dropout = self.emb_dropout(ut_seq_emb)
        it_gru_output, _ = self.tag_gru_layers(it_seq_emb_dropout)



        output_concat = torch.cat(
            (item_gru_output, it_gru_output), -1
        )  # [B Len 2*H]
        output = self.dense_layer(output_concat)
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output
        # return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        if self.mode =='item_tag':
            item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq]
        elif self.mode == 'user_tag':
            item_tag_id_lists = self.item_feat['user_tag_id_list'][item_seq] #(50, 15)

        seq_output = self.forward(item_seq, item_seq_len, item_tag_id_lists) #[b, embed_size]
        
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            if self.mode =='user_tag':
                pos_item_tag_lists = self.item_feat['user_tag_id_list'][pos_items]
                neg_item_tag_lists = self.item_feat['user_tag_id_list'][neg_items] #[512, 19]
                pos_it_embeds = self.user_tag_embeddings(pos_item_tag_lists)
                neg_it_embeds = self.user_tag_embeddings(neg_item_tag_lists) 
            elif self.mode == 'item_tag':
                pos_item_tag_lists = self.item_feat['item_tag_id_list'][pos_items]
                neg_item_tag_lists = self.item_feat['item_tag_id_list'][neg_items] #[512, 19]
                pos_it_embeds = self.item_tag_embeddings(pos_item_tag_lists)
                neg_it_embeds = self.item_tag_embeddings(neg_item_tag_lists) 

            # if self.pooling_mode == "max":
            #     pos_it_emb = torch.max(pos_it_embeds, dim=1)[0]
            #     neg_it_emb = torch.max(neg_it_embeds, dim=1)[0]
            # elif self.pooling_mode == "mean":
            #     pos_it_emb = torch.mean(pos_it_embeds, dim=1) 
            #     neg_it_emb = torch.mean(neg_it_embeds, dim=1)
            # elif self.pooling_mode == "sum":
            #     pos_it_emb = torch.sum(pos_it_embeds, dim=1) 
            #     neg_it_emb = torch.sum(neg_it_embeds, dim=1)
            
            # pos_it_emb = torch.cat((pos_items_emb, pos_it_emb), dim=-1) #[512, 64+embed_size]
            # neg_it_emb = torch.cat((neg_items_emb, neg_it_emb), dim=-1) #[512, 64+embed_size]

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            # pos_item_tag_emb = self.item_tag_embeddings(pos_it)
            # neg_item_tag_emb = self.item_tag_embeddings(neg_it) #[b, 200, embed_size]
            # pos_it_score = torch.sum(it_seq_output * pos_it_emb, dim=-1)  # [B]
            # neg_it_score = torch.sum(it_seq_output * neg_it_emb, dim=-1)  # [B]
            # loss_it = self.loss_fct(pos_it_score, neg_it_score)
            # print(f"loss {loss} loss_it {loss_it}")
            # loss =  self.lamda * loss_it + (1-self.lamda) * loss
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

            # it_emb = self.item_tag_embeddings(self.item_feat['item_tag_id_list'])
            # logits += torch.matmul(seq_output, it_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        if self.mode == "item_tag":
            item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] #(50, 15)
            tags = self.item_feat['item_tag_id_list'][test_item]
            tag_embs = self.item_tag_embeddings(tags)
        else:
            item_tag_id_lists = self.item_feat['user_tag_id_list'][item_seq] #(50, 15)
            tags = self.item_feat['user_tag_id_list'][test_item]
            tag_embs = self.user_tag_embeddings(tags)
        seq_output = self.forward(item_seq, item_seq_len, item_tag_id_lists)
        test_item_emb = self.item_embedding(test_item)

        # item_tags = self.item_feat['item_tag_id_list'][test_item]
        # item_tag_embs = self.item_tag_embeddings(item_tags)
        
        # if self.pooling_mode == "max":
        #     item_tag_emb = tag_embs.max(dim=1)[0]
        # elif self.pooling_mode == "sum":
        #     item_tag_emb = tag_embs.sum(dim=1)
        # elif self.pooling_mode == "mean":
        #     item_tag_emb = tag_embs.mean(dim=1)
        # item_tag_emb = torch.cat((test_item_emb, item_tag_emb), dim=-1)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # item_tag_id_lists = self.item_feat['item_tag_id_list'][item_seq] #(50, 15)
        if self.mode == "item_tag":
            tags = self.item_feat['item_tag_id_list'][item_seq]
            all_tags = self.item_feat['item_tag_id_list']
            all_tag_embs = self.item_tag_embeddings(all_tags)
        else:
            tags = self.item_feat['user_tag_id_list'][item_seq]
            all_tags = self.item_feat['user_tag_id_list']
            all_tag_embs = self.user_tag_embeddings(all_tags)

        seq_output = self.forward(item_seq, item_seq_len, tags)
        test_items_emb = self.item_embedding.weight

        # if self.pooling_mode == "max":
        #     item_tag_emb = all_tag_embs.max(dim=1)[0]
        # elif self.pooling_mode == "sum":
        #     item_tag_emb = all_tag_embs.sum(dim=1)
        # elif self.pooling_mode == "mean":
        #     item_tag_emb = all_tag_embs.mean(dim=1)
        # test_items_embeds = torch.cat((test_items_emb, item_tag_emb), dim=-1)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
    
