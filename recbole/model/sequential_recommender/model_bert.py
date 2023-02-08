import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math


class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        #       [bz, 20, seq_len, 20] x [bz, 20, 20, seq_len] -> [bz, 20, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        #       [bz, 20, seq_len, seq_len] x [bz, 20, seq_len, 20] -> [bz, 20, seq_len, 20]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, enable_gpu):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        self.enable_gpu = enable_gpu

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        batch_size, seq_len, _ = Q.shape

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len) #  [bz, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [bz, 20, seq_len, seq_len]

        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)  # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  #self.layer_norm(output + residual)


class WeightedLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(WeightedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight_softmax =  nn.Softmax(dim=-1)(self.weight)
        return F.linear(input, weight_softmax)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

class TextEncoder(torch.nn.Module):
    def __init__(self,
                 word_embedding_dim,
                 num_attention_heads,
                 query_vector_dim,
                 dropout_rate,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        #self.word_embedding = word_embedding
        self.dropout_rate = dropout_rate
        self.multihead_attention = MultiHeadAttention(word_embedding_dim,
                                                      num_attention_heads, 20,
                                                      20, enable_gpu)
        self.additive_attention = AdditiveAttention(num_attention_heads*20,
                                                    query_vector_dim)

    def forward(self, word_emb, mask=None):
        """
        Args:
            text: Tensor(batch_size) * num_words_text * embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_text
        # import pdb; pdb.set_trace()
        # batch_size, num_words = text.shape
        # num_words = num_words // 3
        # text_ids = torch.narrow(text, 1, 0, num_words)
        # text_type = torch.narrow(text, 1, num_words, num_words)
        # text_attmask = torch.narrow(text, 1, num_words*2, num_words)
        # word_emb = self.bert_model(text_ids, text_type, text_attmask)[2][8]
        text_vector = F.dropout(word_emb,
                                p=self.dropout_rate,
                                training=self.training)
        # batch_size, num_words_text, word_embedding_dim
        multihead_text_vector = self.multihead_attention(
            text_vector, text_vector, text_vector, mask)
        multihead_text_vector = F.dropout(multihead_text_vector,
                                          p=self.dropout_rate,
                                          training=self.training)
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(multihead_text_vector, mask)
        return text_vector


class ElementEncoder(torch.nn.Module):
    def __init__(self, num_elements, embedding_dim, enable_gpu=True):
        super(ElementEncoder, self).__init__()
        self.enable_gpu = enable_gpu
        self.embedding = nn.Embedding(num_elements,
                                      embedding_dim,
                                      padding_idx=0)

    def forward(self, element):
        # batch_size, embedding_dim
        element_vector = self.embedding(
            (element.cuda() if self.enable_gpu else element).long())
        return element_vector


class NewsEncoder(torch.nn.Module):
    def __init__(self, args, bert_model, category_dict_size,
                 domain_dict_size, subcategory_dict_size):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.attributes2length = {
            'title': args.num_words_title * 3,
            'abstract': args.num_words_abstract * 3,
            'body': args.num_words_body * 3,
            'category': 1,
            'domain': 1,
            'subcategory': 1
        }
        for key in list(self.attributes2length.keys()):
            if key not in args.news_attributes:
                self.attributes2length[key] = 0

        self.attributes2start = {
            key: sum(
                list(self.attributes2length.values())
                [:list(self.attributes2length.keys()).index(key)])
            for key in self.attributes2length.keys()
        }

        assert len(args.news_attributes) > 0
        text_encoders_candidates = ['title', 'body']

        self.text_encoders = nn.ModuleDict({
            'title':
            TextEncoder(bert_model,
                        args.word_embedding_dim,
                        args.num_attention_heads, args.news_query_vector_dim,
                        args.drop_rate, args.enable_gpu)
        })
    
        self.newsname=[name for name in set(args.news_attributes) & set(text_encoders_candidates)]


            
        name2num = {
            "category": category_dict_size + 1,
            "domain": domain_dict_size + 1,
            "subcategory": subcategory_dict_size + 1
        }
        element_encoders_candidates = ['category', 'domain', 'subcategory']
        self.element_encoders = nn.ModuleDict({
            name: ElementEncoder(name2num[name], 
                                args.num_attention_heads * 20,
                                 args.enable_gpu)
            for name in (set(args.news_attributes)
                         & set(element_encoders_candidates))
        })
        #if len(args.news_attributes) > 1:
        #    self.final_attention = AdditiveAttention(
        #        args.num_attention_heads * 20, args.news_query_vector_dim)

        self.reduce_dim_linear = nn.Linear(args.num_attention_heads * 20,
                                           args.news_dim)

        if args.use_pretrain_news_encoder:
            self.reduce_dim_linear.load_state_dict(
                torch.load(os.path.join(args.pretrain_news_encoder_path, 
                'reduce_dim_linear.pkl'))
            )

    def forward(self, news):
        """
        Args:
        Returns:
            (shape) batch_size, news_dim
        """
        text_vectors = [
            self.text_encoders['title'](
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]))
            for name in self.newsname
        ]

        element_vectors = [
            encoder(
                torch.narrow(news, 1, self.attributes2start[name],
                             self.attributes2length[name]).squeeze(dim=1))
            for name, encoder in self.element_encoders.items()
        ]

        all_vectors = text_vectors + element_vectors
        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:

            final_news_vector = torch.mean(
                torch.stack(all_vectors, dim=1),
                dim=1
             )
        
        # batch_size, news_dim
        final_news_vector = self.reduce_dim_linear(final_news_vector)
        return final_news_vector


class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.news_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)
        if args.use_padded_news_embedding:
            # self.news_padded_news_embedding = nn.Embedding(1, args.news_dim)
            self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        else:
            # self.news_padded_news_embedding = None
            self.pad_doc = None
        

    def _process_news(self, vec, mask, pad_doc,
                    additive_attention, use_mask=False, 
                    use_padded_embedding=False):
        assert not (use_padded_embedding and use_mask), 'Conflicting config'
        if use_padded_embedding:
            # batch_size, maxlen, dim
            batch_size = vec.shape[0]
            padding_doc = pad_doc.expand(batch_size, self.args.news_dim).unsqueeze(1).expand( \
                                         batch_size, self.args.user_log_length , self.args.news_dim)
            # batch_size, maxlen, dim
            vec = vec * mask.unsqueeze(2).expand(-1, -1, self.args.news_dim) + padding_doc * (1 - mask.unsqueeze(2).expand(-1, -1, self.args.news_dim))
        # batch_size, news_dim
        vec = additive_attention(vec,
                                 mask if use_mask else None)
        return vec

    
    def forward(self, log_vec, log_mask):
        """
        Returns:
            (shape) batch_size,  news_dim
        """
        # batch_size, news_dim
        log_vec = self._process_news(log_vec, log_mask, self.pad_doc,
                                     self.news_additive_attention, self.args.user_log_mask,
                                     self.args.use_padded_news_embedding)
        
        user_log_vecs = log_vec


        return user_log_vecs


class ModelBert(torch.nn.Module):
    """
    UniUM network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,
                 args,
                 bert_model,
                 category_dict_size=0,
                 domain_dict_size=0,
                 subcategory_dict_size=0):
        super(ModelBert, self).__init__()
        self.args = args


        self.news_encoder = NewsEncoder(args,
                                        bert_model,
                                        category_dict_size, 
                                        domain_dict_size,
                                        subcategory_dict_size)
        self.user_encoder = UserEncoder(args)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                input_ids,
                log_ids,
                log_mask,
                targets=None,
                compute_loss=True):
        """
        Returns:
          click_probability: batch_size, 1 + K
        """
        # input_ids: batch, history, num_words
        ids_length = input_ids.size(2)
        input_ids = input_ids.view(-1, ids_length)
        news_vec = self.news_encoder(input_ids)
        news_vec = news_vec.view(-1, 1 + self.args.npratio, self.args.news_dim)

        # batch_size, news_dim
        log_ids = log_ids.view(-1, ids_length)
        log_vec = self.news_encoder(log_ids)
        log_vec = log_vec.view(-1, self.args.user_log_length,
                               self.args.news_dim)

        user_vector = self.user_encoder(log_vec, log_mask)

        # batch_size, 2
        score = torch.bmm(news_vec, user_vector.unsqueeze(-1)).squeeze(dim=-1)
        if compute_loss:
            loss = self.criterion(score, targets)
            return loss, score
        else:
            return score

