import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from einops import rearrange
import torch.nn.functional as F
import copy
import math

class PreNorm(Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Attention(Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias=False)
        self.to_out = nn.Sequential(nn.Linear(2 * inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        (b, n, d) = x.size()
        qkvt = self.to_qkv(x).chunk(4, dim=-1)
        (q, k, v, t) = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkvt)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots)
        tmp_ones = torch.ones(n).cuda()
        tmp_n = torch.linspace(1, n, n).cuda()
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1, 1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.0)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b, self.heads, 1, 1)
        out = torch.cat([torch.matmul(attn1, v), torch.matmul(attn2, t)], dim=-1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = nn.LayerNorm(size)

    def forward(self, q, k, v):
        (q, k, v) = (self.norm(q), self.norm(k), self.norm(v))
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0])
        return self.sublayer[1](q, self.feed_forward)

class SelfAttentionBlock(nn.Module):

    def __init__(self, attention_layer):
        super(SelfAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size

    def forward(self, feature):
        feature_sa = self.layer(feature, feature, feature)
        return feature_sa

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

def attention(query, key, value, masksize, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if masksize != 1:
        masksize = int(masksize / 2)
        mask = torch.ones(scores.size()).cuda()
        for i in range(mask.shape[2]):
            if i - masksize > 0:
                mask[:, :, i, :i - masksize] = 0
            if i + masksize + 1 < mask.shape[3]:
                mask[:, :, i, masksize + i + 1:] = 0
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return (torch.matmul(p_attn, value), p_attn)

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, masksize=1, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.masksize = masksize
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        (query, key, value) = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for (l, x) in zip(self.linears, (query, key, value))]
        (x, self.attn) = attention(query, key, value, self.masksize, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return (out, self.attn)

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output

class Transformer(Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for (attn, ff) in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x