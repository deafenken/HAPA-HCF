import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from models.self_attention import MultiHeadAttention

class HCF(nn.Module):

    def __init__(self, input_dims, hidden_dim=64):
        super().__init__()
        self.fc_layers = nn.ModuleList()
        for dim in input_dims:
            self.fc_layers.append(nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)))

    def forward(self, features):
        weights = []
        for (i, feat) in enumerate(features):
            w = self.fc_layers[i](feat)
            weights.append(w)
        weights = torch.stack(weights, dim=-1)
        weights = F.softmax(weights, dim=-1).squeeze(-2)
        fused = torch.zeros_like(features[0])
        for i in range(len(features)):
            fused += features[i] * weights[..., i:i + 1]
        return fused

class CrossModalContextualInteraction(nn.Module):

    def __init__(self, dims, heads=4, dropout=0.1):
        super().__init__()
        self.dims = dims
        self.n_modal = len(dims)
        self.cross_attns = nn.ModuleList()
        for i in range(self.n_modal):
            for j in range(self.n_modal):
                if i != j:
                    self.cross_attns.append(MultiHeadAttention(heads, dims[i], dropout=dropout))
        self.projections = nn.ModuleList()
        self.target_dim = max(dims)
        for dim in dims:
            if dim != self.target_dim:
                self.projections.append(nn.Linear(dim, self.target_dim))
            else:
                self.projections.append(nn.Identity())

    def forward(self, features):
        (batch_size, seq_len) = (features[0].shape[0], features[0].shape[1])
        projected = [self.projections[i](feat) for (i, feat) in enumerate(features)]
        enhanced = []
        idx = 0
        for i in range(self.n_modal):
            feat_i = projected[i]
            for j in range(self.n_modal):
                if i != j:
                    feat_j = projected[j]
                    (attn_out, _) = self.cross_attns[idx](feat_i, feat_j, feat_j)
                    feat_i = feat_i + attn_out
                    idx += 1
            enhanced.append(feat_i)
        return enhanced

class FusionMILHead(nn.Module):

    def __init__(self, input_dim, h_dim=512, dropout_rate=0.0):
        super(FusionMILHead, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(input_dim, h_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(h_dim, 32), nn.Dropout(dropout_rate), nn.Linear(32, 1), nn.Sigmoid())

    def filter(self, logits, seq_len):
        instance_logits = torch.zeros(0).cuda()
        for i in range(logits.shape[0]):
            if seq_len is None:
                return logits
            else:
                k = max(1, int(seq_len[i] // 16 + 1))
                (tmp, _) = torch.topk(logits[i][:seq_len[i]], k=k, largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        return instance_logits

    def forward(self, avf_out, seq_len):
        avf_out = self.regressor(avf_out)
        avf_out = avf_out.squeeze()
        mmil_logits = self.filter(avf_out, seq_len)
        return (mmil_logits, avf_out)

class HCFNet(nn.Module):

    def __init__(self, input_size, h_dim=32, feature_dim=64):
        super().__init__()
        self.modal_dims = [128, 128, 128]
        self.embedding = nn.Sequential(nn.Linear(128, input_size // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(input_size // 2, feature_dim), nn.ReLU())
        self.tcn = TemporalConvNet(num_inputs=feature_dim, num_channels=[feature_dim, feature_dim * 2, feature_dim], dropout=0.1)
        self.cross_modal_enhancer = CrossModalContextualInteraction(self.modal_dims)
        self.dynamic_fusion = HCF(self.modal_dims)
        self.mil = FusionMILHead(input_dim=feature_dim, h_dim=h_dim)

    def forward(self, data, seq_len=None):
        v_feat = data[..., :128]
        a_feat = data[..., 128:256]
        f_feat = data[..., 256:]
        enhanced_feats = self.cross_modal_enhancer([v_feat, a_feat, f_feat])
        fused_feat = self.dynamic_fusion(enhanced_feats)
        fused_feat = self.embedding(fused_feat)
        fused_feat = self.tcn(fused_feat.permute(0, 2, 1)).permute(0, 2, 1)
        (output, avf_out) = self.mil(fused_feat, seq_len)
        return {'output': output, 'avf_out': avf_out, 'satt_f': fused_feat}