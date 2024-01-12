import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# needs modify
class RelativeAttention(Attention):
    def __init__(
            self, nx, n_ctx, n_head, scale=False, dropout=None, additive=False,
            use_tree=False, use_seq=False, rel_vocab_size=None, use_global=True, use_local=True, args=None
    ):
        super(RelativeAttention, self).__init__(nx, n_ctx, n_head, scale, dropout)
        self.additive = additive
        self.use_seq = use_seq
        self.use_tree = use_tree

        self.use_global = use_global
        self.use_local = use_local

        self.rel_weights = nn.Embedding(rel_vocab_size, n_head)

 


    def matmul_with_relative_representations(self, q, rel, transpose_rel=False):
        # sequential relative attention helper function
        # yian: Masked matrix?
        # q: [b, h, n, dh] -> [n, b, h, dh] -> [n, b*h, dh] : batch, head, seq, q_feature
        # rel: [n, n, dh] -> [n, dh, n] 
        # return: [b, h, n, n]
        nb, nh, nt, _ = q.size()
        q = q.permute(2, 0, 1, 3).contiguous()
        q = q.reshape(q.size(0), nb * nh, q.size(-1))
        if not transpose_rel:
            rel = rel.permute(0, 2, 1)
        x = torch.matmul(q, rel)
        x = x.reshape(nt, nb, nh, -1)
        x = x.permute(1, 2, 0, 3).contiguous()
        return x

    def _attn(self, q, k, v, tds=None, lr=None):
        # (batch, head, seq_length, head_features)
        # q = xW^Q
        # k = xW^K
        w = torch.matmul(q, k) # alpha_ij
        nd, ns = w.size(-2), w.size(-1)

        if self.use_global:
            B = torch.matmul(tds.unsqueeze(3), tds.unsqueeze(2).transpose(-1, -2))
            w = w + B

        if self.scale:
            w = w / math.sqrt(v.size(-1))

        
        r = self.rel_matrix(lr)
        if self.use_local:
            first_term = torch.matmul(q.unsqueeze(3), r.transpose(-1, -2))
            second_term = torch.matmul(r, k.unsqueeze(2).transpose(-1, -2))
            Y = first_term + second_term
            w = w + Y


        w_normed = nn.Softmax(dim=-1)(w)  # calc attention scores
        if self.dropout is not None:
            w_normed = self.dropout(w_normed)

        ret = torch.matmul(w_normed, v)

        return ret

    def self_attention(self, query, key, value, rel, tds, lr):
        a = self._attn(query, key, value, rel, tds, lr)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

    def forward(self, x, rel=None, tds=None, lr=None):
        query, key, value = self.get_q_k_v(x)
        if self.use_tree:
            rel = self.rel_weights(rel)
            rel = rel.permute(0, 3, 1, 2)
        return self.self_attention(query, key, value, rel, tds, lr)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# needs modify
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.fc2 = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        #tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        """dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)"""
        #print(enc_output.shape)
        output = self.fc2(torch.mean(enc_output, -2))
        return output