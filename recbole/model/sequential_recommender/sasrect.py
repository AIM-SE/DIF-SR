# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, PosTransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F
from recbole.model.sequential_recommender.model_bert import *
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

class SASRecT(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRecT, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.device = config['device']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.attr_loss = config['attr_loss']
        self.selected_features = config['selected_features']
        self.attribute_reg_indexs = config['attr_regi']
        self.attr_lamdas = config['attr_lamdas']
        self.attr_multi_lamda = config['attr_multi_lamda']

        if type(self.attr_lamdas) == int:
            self.attr_lamdas = [self.attr_lamdas]
        if type(self.attribute_reg_indexs) == int:
            self.attribute_reg_indexs = [self.attribute_reg_indexs]

        self.vis = config['vis'] > 0
        self.pos_atten = config['pos_atten'] > 0
        self.prefix = config['exp']
        self.fusion_type = config['fusion_type']


        self.logger.info("Start to load text data")
        self.text_field = config['text_field']
        self.item_text = dataset.item_feat[self.text_field]
        self.item_text_context = dataset.id2token(self.text_field, self.item_text).tolist()
        self.text_n_heads = 20

        self.logger.info("Start to calculate text")
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        tokens = tokenizer(self.item_text_context, return_tensors="pt", padding=True)
        del tokenizer

        self.logger.info("Start to retrieve text emb")

        # GPU version
        bert_encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        text_encoder = TextEncoder(768,self.text_n_heads, 200, 0.2, config['use_gpu'])
        text_embs_batch = []
        batch = 2048
        for i in tqdm(range(0, len(self.item_text_context), batch)):
            ids = tokens['input_ids'][i:i+batch].to(self.device)
            mask = tokens['attention_mask'][i:i+batch].to(self.device)
            type_ids = tokens['token_type_ids'][i:i+batch].to(self.device)
            with torch.no_grad():
                token_emb = bert_encoder(ids, mask, type_ids)
                token_embs_device = token_emb[0]
                text_embs = text_encoder(token_embs_device, mask)
            text_embs_batch.append(text_embs.cpu())
            del ids, mask, type_ids, token_emb, text_embs
            torch.cuda.empty_cache()
        import pdb; pdb.set_trace()
        self.text_embs = torch.cat(text_embs_batch).to(self.device)

        # CPU version
        # bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        # token_embs = bert_encoder(tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'])[0]
        # token_embs = bert_encoder(tokens['input_ids'].to(self.device), tokens['attention_mask'].to(self.device), tokens['token_type_ids'].to(self.device))
        del bert_encoder, text_encoder
        torch.cuda.empty_cache()

        self.logger.info("Finish to calculate text")

        self.reduce_dim_linear = nn.Linear(self.text_n_heads * 20,
                                           self.hidden_size)


        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = PosTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            feat_num=len(self.selected_features),
            max_len=self.max_seq_length,
            fusion_type=self.fusion_type
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)


        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")



        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def run_per_epoch(self, epoch):
        if self.vis and epoch % 2 == 0:
            self.vis_emb(self.item_embedding, epoch, exp=self.prefix+"_cat", labels=self.item_attributes[0].detach().cpu().numpy())
            self.vis_emb(self.item_embedding, epoch, exp=self.prefix+"_pop")

    def convert_one_hot(self, feature, size):
        """ Convert user and item ids into one-hot format. """
        batch_size = feature.shape[0]
        # seq_size = feature.shape[1]
        feature = feature.view(batch_size, 1)
        f_onehot = torch.FloatTensor(batch_size, size).to(self.device)
        f_onehot.zero_()
        f_onehot.scatter_(-1, feature, 1)

        return f_onehot
        # return f_onehot.view(batch_size, size)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        text_emb = self.text_embs[item_seq]
        text_emb = self.reduce_dim_linear(text_emb)
        input_emb = item_emb + text_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, position_embedding, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def to_onehot(self, labels, n_categories, dtype=torch.float32):
        batch_size = len(labels)
        one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype).to(self.device)
        for i, label in enumerate(labels):
            # Subtract 1 from each LongTensor because your
            # indexing starts at 1 and tensor indexing starts at 0
            # label = (torch.LongTensor(label).cpu() - 1).to(self.device)
            one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
        one_hot_labels[:, 0] = 0.0
        return one_hot_labels

    def calculate_uniform_loss(self, x):
        x = F.normalize(x, dim=-1)
        return -torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            losses = [loss]

            if self.uniform_lamda != 0:
                if self.uniform_level == 'batch':
                    items_in_batch = item_seq.unique()
                    embs_in_batch = self.item_embedding(items_in_batch)
                    uni_loss = self.calculate_uniform_loss(embs_in_batch)
                elif self.uniform_level == 'label':
                    pos_items = pos_items.unique()
                    pos_items_emb = self.item_embedding(pos_items)
                    uni_loss = self.calculate_uniform_loss(pos_items_emb)
                else:
                    uni_loss = self.calculate_uniform_loss(self.item_embedding.weight)
                # import pdb; pdb.set_trace()
                losses.append(uni_loss*self.uniform_lamda)

            if self.attr_loss == "predict":
                for i, attr_layer in enumerate(self.attr_layers):
                    attribute_logits = attr_layer(test_item_emb)
                    attribute_labels = self.item_attributes[i]
                    attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.item_attribute_counts[i])
                    attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels.float())
                    attr_loss = attribute_loss if self.loss_redu else torch.mean(attribute_loss)
                    losses.append(self.attr_lamdas[i]*attr_loss)
            elif self.attr_loss == "multi":
                attribute_logits = self.multi_attr_layer(test_item_emb)
                attribute_labels = self.raw_item_attributes
                # attribute_labels = nn.functional.one_hot(attribute_labels, num_classes=self.all_attribute_count)
                attribute_labels = self.to_onehot(attribute_labels, self.all_attribute_count)
                # import pdb; pdb.set_trace()
                # attribute_labels = attribute_labels.sum(dim=1)
                attribute_loss = self.attribute_loss_fct(attribute_logits, attribute_labels.float())
                attr_loss = attribute_loss if self.loss_redu else torch.mean(attribute_loss)
                losses.append(self.attr_multi_lamda * attr_loss)
            else:
                test_attr_emb = self.attr_embedding.weight
                attr_logits = torch.matmul(test_item_emb, test_attr_emb.transpose(0, 1))
                attr_loss = self.loss_fct(attr_logits, self.item_attribute)
                losses.append(self.attr_lamdas[0] * attr_loss)

            return tuple(losses)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
