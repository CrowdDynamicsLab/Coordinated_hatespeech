import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.pos_keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.pos_queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Linear transformations for the rel tensor
        self.rel_keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.rel_queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, pos, rel, mask, mode):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        #print(N, query_len, self.heads, self.head_dim, rel.size())
        # Split into multiple heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        pos = pos.reshape(N, query_len, self.heads, self.head_dim)
        rel = rel.reshape(N, query_len, query_len, self.heads, self.head_dim)
        #rel = rel.reshape(N, query_len, -1, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mode == 'only':
            energy = (attention_scores) / math.sqrt(2)
        
        elif mode == 'rel':
            # Transform the rel tensor
            rel_keys = self.rel_keys(rel)
            rel_queries = self.rel_queries(rel)
            # Calculate the rel scores
            #rel_scores_1 = torch.einsum("nqhd,nkhd->nhqk", [queries, rel_keys])
            #rel_scores_2 = torch.einsum("nqhd,nkhd->nhqk", [rel_queries, keys])
            rel_scores_1 = torch.einsum("bqhd,bqkhd->bhqk", [queries, rel_keys])
            rel_scores_2 = torch.einsum("bqkhd,bkhd->bhqk", [rel_queries, keys])
            
            all_scores = attention_scores + rel_scores_1 + rel_scores_2
            energy = all_scores / math.sqrt(2)
            
        elif mode == 'pos':
            pos_queries = self.pos_queries(pos)  # Assuming the same projection for pos as for queries
            pos_keys = self.pos_keys(pos)
            pos_scores = torch.einsum("nqhd,nkhd->nhqk", [pos_queries, pos_keys])
            all_scores = attention_scores + pos_scores
            energy = all_scores / math.sqrt(2)
            
        elif mode == 'all':
            pos_queries = self.pos_queries(pos)  # Assuming the same projection for pos as for queries
            pos_keys = self.pos_keys(pos)
            pos_scores = torch.einsum("nqhd,nkhd->nhqk", [pos_queries, pos_keys])
            
            # Transform the rel tensor
            rel_keys = self.rel_keys(rel)
            rel_queries = self.rel_queries(rel)

            # Calculate the rel scores
            rel_scores_1 = torch.einsum("bqhd,bqkhd->bhqk", [queries, rel_keys])
            rel_scores_2 = torch.einsum("bqkhd,bkhd->bhqk", [rel_queries, keys])
            
            all_scores = attention_scores + pos_scores + rel_scores_1 + rel_scores_2
            energy = all_scores / math.sqrt(2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Reshape mask to [N, 1, 1, seq_len]
            mask_temp = 1.0 - mask
            mask_temp2 = mask_temp * -1e9  # Transform the mask
            energy = energy + mask_temp2  # Apply mask to energy

        attention = torch.softmax(energy, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, pos, rel, mask, mode):
        attention = self.attention(value, key, query, pos, rel, mask, mode)

        # Add skip connection, run through normalization and finally dropout
        temp = attention + query
        x = self.dropout(self.norm1(temp))
        forward = self.feed_forward(x)
        temp2 = forward + x
        out = self.dropout(self.norm2(temp2))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size

        # Create a long enough 'pe' matrix that can be sliced according to max_len
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embed_size)))

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Make the positional encoding as long as the input sequence
        x = x + self.pe[:, :x.size(1)].detach()
        return x

class CustomTransformerModel(nn.Module):
    def __init__(self, feature_dim, global_dim, embed_size, num_classes, num_heads, num_layers, 
                 dropout, forward_expansion, max_len, mode):
        super().__init__()
        self.mode = mode
        self.feature_to_embedding = nn.Linear(feature_dim, embed_size)
        self.global_to_embedding = nn.Linear(global_dim, embed_size)
        self.local_to_embedding = nn.Linear(global_dim, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, num_classes)
        self.prob_out = nn.Linear(embed_size, 8)

    def forward(self, x, pos, rel, mask):
        out = self.feature_to_embedding(x)
        #out = self.positional_encoding(out)
        pos_emb = self.global_to_embedding(pos)
        rel_emb = self.local_to_embedding(rel)
        for layer in self.layers:
            out = layer(out, out, out, pos_emb, rel_emb, mask, self.mode)
        edge_prob = self.prob_out(out)
        #prob_out = F.softmax(edge_prob, dim=-1) 
        return self.fc_out(out), out
    
class StratModel(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.prob_out = nn.Linear(embed_size, 8)

    def forward(self, out):
        edge_prob = self.prob_out(out)
        prob_out = F.softmax(edge_prob, dim=-1) 
        return prob_out


        
