import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorGuidedContextualInteraction(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim * 2, dim), nn.LayerNorm(dim), nn.Dropout(dropout))
        self.temp = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, src, tgt):
        (b, t, _) = src.shape
        q_src = self.to_q(src).view(b, t, self.heads, self.dim_head).transpose(1, 2)
        k_tgt = self.to_k(tgt).view(b, t, self.heads, self.dim_head).transpose(1, 2)
        v_tgt = self.to_v(tgt).view(b, t, self.heads, self.dim_head).transpose(1, 2)
        q_tgt = self.to_q(tgt).view(b, t, self.heads, self.dim_head).transpose(1, 2)
        k_src = self.to_k(src).view(b, t, self.heads, self.dim_head).transpose(1, 2)
        v_src = self.to_v(src).view(b, t, self.heads, self.dim_head).transpose(1, 2)
        attn_src2tgt = F.softmax(torch.matmul(q_src, k_tgt.transpose(-1, -2)) / (self.dim_head ** 0.5 * self.temp), dim=-1)
        out_src2tgt = torch.matmul(attn_src2tgt, v_tgt).transpose(1, 2).reshape(b, t, self.inner_dim)
        attn_tgt2src = F.softmax(torch.matmul(q_tgt, k_src.transpose(-1, -2)) / (self.dim_head ** 0.5 * self.temp), dim=-1)
        out_tgt2src = torch.matmul(attn_tgt2src, v_src).transpose(1, 2).reshape(b, t, self.inner_dim)
        out = self.to_out(torch.cat([out_src2tgt, out_tgt2src], dim=-1))
        return out + src

class PrototypeGuidedRectification(nn.Module):

    def __init__(self, dim, subspace_dim=64):
        super().__init__()
        self.dim = dim
        self.subspace_dim = subspace_dim
        self.global_proj = nn.Linear(dim, subspace_dim)
        self.adaptive_weight = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, main_feat, aux_feat):
        (b, t, d) = main_feat.shape
        main_proj = self.global_proj(main_feat)
        aux_proj = self.global_proj(aux_feat)
        weight = self.adaptive_weight(main_feat)
        aligned_feat = weight * main_proj + (1 - weight) * aux_proj
        return aligned_feat

class TemporalConsistency(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size, padding=1, groups=dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, feat, seq_len):
        (b, t, d) = feat.shape
        mask = torch.zeros(b, t, 1).cuda()
        for i in range(b):
            mask[i, :seq_len[i], :] = 1
        feat_t = feat.transpose(1, 2)
        temporal_feat = self.temporal_conv(feat_t).transpose(1, 2)
        temporal_feat = self.layer_norm(temporal_feat)
        return feat * mask + temporal_feat * mask

class UAPA(nn.Module):

    def __init__(self, main_dim=128, aux_dim=32, heads=4, subspace_dim=64):
        super().__init__()
        self.aux_proj = nn.Linear(aux_dim, main_dim) if aux_dim != main_dim else nn.Identity()
        self.cross_attn = AnchorGuidedContextualInteraction(main_dim, heads=heads)
        self.dynamic_subspace = PrototypeGuidedRectification(main_dim, subspace_dim=subspace_dim)
        self.temporal_consistency = TemporalConsistency(subspace_dim)
        self.out_proj = nn.Linear(subspace_dim, main_dim)

    def forward(self, main_feat, aux_feat, seq_len):
        aux_feat = self.aux_proj(aux_feat)
        attn_feat = self.cross_attn(main_feat, aux_feat)
        subspace_feat = self.dynamic_subspace(attn_feat, aux_feat)
        aligned_subspace = self.temporal_consistency(subspace_feat, seq_len)
        output = self.out_proj(aligned_subspace)
        return output