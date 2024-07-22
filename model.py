import torch
import math
import torch.nn as nn
import random
from torch.nn import functional as F
from data_loader import *
from util import *

class MLP(nn.Module):
    
    def __init__(self, seq_len, n_hidden, pred_len):
        super().__init__()
        self.mlp = nn.Sequential(
                nn.Linear(seq_len, n_hidden),
                nn.LayerNorm(n_hidden),
                nn.SiLU(),
                nn.Linear(n_hidden, pred_len)
            )
    
    def forward(self, x):
        return self.mlp(x)

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, n_heads):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features // n_heads
        self.alpha = alpha
        self.n_heads = n_heads
        self.W = nn.Parameter(torch.zeros(size=(n_heads, in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        batch_size, num_nodes = input.size(0), input.size(1)
        h = torch.einsum('bnd,hde->bhne', input, self.W)  # (batch_size, n_heads, num_nodes, out_features)
        f_expand = h.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        b_expand = h.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1)
        a_input = torch.cat([f_expand, b_expand], dim=-1)  # (batch_size, n_heads, num_nodes, num_nodes, 2 * out_features)
        a_input = a_input.view(batch_size * self.n_heads, num_nodes * num_nodes, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        e = e.view(batch_size, self.n_heads, num_nodes, num_nodes)
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('bhij,bhje->bhie', attention, h)
        h_prime = h_prime.view(batch_size, num_nodes, -1)  
        return h_prime

class Transformer(nn.Module):
    
    def __init__(self,
                 seq_len: int,
                 pred_len: int,
                 n_hidden: int, 
                 n_layers: int, 
                 n_heads: int, 
                 dropout=0.1, 
    ):
        super().__init__()
        
        self.in_proj = nn.Linear(seq_len, n_hidden)
        self.blocks = nn.ModuleList([EncoderBlock(n_hidden, 
                                                  n_heads, dropout, 
                                                 ) for l in range(n_layers)])

        self.out_proj = nn.Linear(n_hidden, pred_len)
        self.initialize_weights()

    def initialize_weights(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if type(m) == nn.Embedding:
                nn.init.normal_(m.weight)
            if type(m) == nn.LayerNorm:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, x):
        # x (Batch, 208, seq_len)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        # x (Batch, 208, pred_len)
        return self.out_proj(x)

class SelfAttention(nn.Module):
    
    def __init__(self, 
                 n_heads: int, 
                 n_hidden: int, 
                 dropout=0.1, 
                 in_proj_bias=False, 
                 out_proj_bias=False,
    ):
        super().__init__()
        
        self.q_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)
        self.k_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)
        self.v_proj = nn.Linear(n_hidden, n_hidden, bias=in_proj_bias)

        self.out_proj = nn.Linear(n_hidden, n_hidden, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = n_hidden // n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x (Batch, 208, n_hidden)

        input_shape = x.shape
        batch_size, sequence_length, n_embd = input_shape
        
        interm_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # (Batch, 208, n_hidden) -> (Batch, 208, n_heads, d_head) -> (Batch, n_heads, 208, d_head)
        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        if mask is not None:
            weight = torch.masked_fill(weight, mask.unsqueeze(1), value=-1e7)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)

        # (Batch, n_heads, 208, 208) @ (Batch, n_heads, 208, d_head) -> (B, n_heads, 208, d_head)
        output = weight @ v
        
        # (B, n_heads, 208, d_head) -> (B, 208, n_heads, d_head)
        output = output.transpose(1, 2).contiguous()
        
        # (B, 208, n_heads, d_head) -> (B, 208, n_hidden)
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        
        # (B, 208, n_hidden)
        return output

class FeedFoward(nn.Module):

    def __init__(self, 
                 n_hidden, 
                 dropout=0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_hidden, 2 * n_hidden),
            nn.SiLU(),
            nn.Linear(2 * n_hidden, n_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):

    def __init__(self, 
                 n_hidden, 
                 n_heads, 
                 dropout=0.1, 
    ):
        super().__init__()
        self.attention = SelfAttention(n_heads, n_hidden, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.ffwd = FeedFoward(n_hidden, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_hidden)
        
    def forward(self, x):
        x = x + self.attention(self.ln1(x), mask=None)
        x = x + self.ffwd(self.ln2(x))
            
        return x

class TransformerWithGAT(nn.Module):
    def __init__(self, seq_len, pred_len, n_hidden, n_layers, n_heads, dropout, adj, n_heads_gat, alpha, combine_factor=0.05):
        super().__init__()
        self.in_proj = nn.Linear(seq_len, n_hidden)
        self.adj = adj
        self.gat = GATLayer(n_hidden, n_hidden, dropout=dropout, alpha=alpha, n_heads=n_heads_gat)
        self.blocks = nn.ModuleList([EncoderBlock(n_hidden, n_heads, dropout) for _ in range(n_layers)])
        self.out_proj = nn.Linear(n_hidden, pred_len)
        self.combine_factor = combine_factor
        self.initialize_weights()

    def initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)

    def forward(self, x):
        x_proj = self.in_proj(x)
        x_gat = self.gat(x_proj, self.adj)
        x = (1 - self.combine_factor) * x_proj + self.combine_factor * x_gat
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

