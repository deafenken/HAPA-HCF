import torch
import torch.nn as nn
import torch.nn.functional as F
from models.uapa_module import UAPA

def CosineDistanceLoss(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].reshape(a[item].shape[0], -1), b[item].reshape(b[item].shape[0], -1)))
    return loss

class UnimodalMILLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def get_loss(self, result, label):
        output = result['output']
        output_loss = self.bce(output, label)
        return output_loss

    def forward(self, v_result, a_result, f_result, label):
        label = label.float()
        v_loss = self.get_loss(v_result, label)
        a_loss = self.get_loss(a_result, label)
        f_loss = self.get_loss(f_result, label)
        U_MIL_loss = v_loss + a_loss + f_loss
        loss_dict = {}
        loss_dict['U_MIL_loss'] = U_MIL_loss
        return (U_MIL_loss, loss_dict)

class UAPAHCFLoss(nn.Module):

    def __init__(self, ra_alignment, rf_alignment) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.triplet = nn.TripletMarginLoss(margin=5)
        self.ra_alignment = ra_alignment
        self.rf_alignment = rf_alignment

    def norm(self, data):
        l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
        return torch.div(data, l2)

    def get_seq_matrix(self, seq_len):
        N = seq_len.size(0)
        M = seq_len.max().item()
        seq_matrix = torch.zeros((N, M))
        for (j, val) in enumerate(seq_len):
            seq_matrix[j, :val] = 1
        seq_matrix = seq_matrix.cuda()
        return seq_matrix

    def compute_fusion_mil_loss(self, result, label):
        output = result['output']
        output_loss = self.bce(output, label)
        return output_loss

    def cross_entropy_loss(self, q, p):
        epsilon = 1e-06
        p = torch.clamp(p, epsilon, 1 - epsilon)
        q = torch.clamp(q, epsilon, 1 - epsilon)
        cross_entropy_loss = -torch.mean(p * torch.log(q) + (1 - p) * torch.log(1 - q))
        return cross_entropy_loss

    def compute_alignment_loss(self, v_result, va_result, vf_result, seq_len):

        def distance(x, y):
            return CosineDistanceLoss(x, y)
        V = v_result['satt_f'].detach().clone()
        A = va_result['satt_f']
        F = vf_result['satt_f']
        batch_size = V.shape[0]
        A_aligned = self.ra_alignment(V, A, seq_len)
        F_aligned = self.rf_alignment(V, F, seq_len)
        d_VA = distance(V, A_aligned) / batch_size
        d_VF = distance(V, F_aligned) / batch_size
        d_AF = distance(A_aligned, F_aligned) / batch_size
        seq_matrix = self.get_seq_matrix(seq_len)
        temporal_loss_VA = torch.mean(torch.abs(V - A_aligned) * seq_matrix.unsqueeze(-1))
        temporal_loss_VF = torch.mean(torch.abs(V - F_aligned) * seq_matrix.unsqueeze(-1))
        total_alignment_loss = d_VA + d_VF + d_AF + 0.1 * (temporal_loss_VA + temporal_loss_VF)
        V_score = v_result['avf_out'].detach().clone() * seq_matrix
        A_score = va_result['avf_out'] * seq_matrix
        F_score = vf_result['avf_out'] * seq_matrix
        ce_VA = self.cross_entropy_loss(V_score, A_score)
        ce_VF = self.cross_entropy_loss(V_score, F_score)
        ce_AF = self.cross_entropy_loss(A_score, F_score)
        return total_alignment_loss + ce_VA + ce_VF + ce_AF

    def compute_triplet_loss(self, hcf_result, label, seq_len):
        if torch.sum(label) == label.shape[0] or torch.sum(label) == 0:
            return 0.0
        N_label = label == 0
        A_label = label == 1
        sigout = hcf_result['avf_out']
        feature = hcf_result['satt_f']
        N_feature = feature[N_label]
        A_feature = feature[A_label]
        N_sigout = sigout[N_label]
        A_sigout = sigout[A_label]
        N_seq_len = seq_len[N_label]
        A_seq_len = seq_len[A_label]
        anchor = torch.zeros(N_feature.shape[0], N_feature.shape[-1]).cuda()
        for i in range(N_sigout.shape[0]):
            (_, index) = torch.topk(N_sigout[i][:N_seq_len[i]], k=int(N_seq_len[i]), largest=True)
            tmp = N_feature[i, index, :]
            anchor[i] = tmp.mean(dim=0)
        anchor = anchor.mean(dim=0)
        positivte = torch.zeros(A_feature.shape[0], A_feature.shape[-1]).cuda()
        negative = torch.zeros(A_feature.shape[0], A_feature.shape[-1]).cuda()
        for i in range(A_sigout.shape[0]):
            (_, index) = torch.topk(A_sigout[i][:A_seq_len[i]], k=int(A_seq_len[i] // 16 + 1), largest=False)
            tmp = A_feature[i, index, :]
            positivte[i] = tmp.mean(dim=0)
            (_, index) = torch.topk(A_sigout[i][:A_seq_len[i]], k=int(A_seq_len[i] // 16 + 1), largest=True)
            tmp = A_feature[i, index, :]
            negative[i] = tmp.mean(dim=0)
        positivte = positivte.mean(dim=0)
        negative = negative.mean(dim=0)
        triplet_margin_loss = self.triplet(self.norm(anchor), self.norm(positivte), self.norm(negative))
        return triplet_margin_loss

    def forward(self, v_result, va_result, vf_result, hcf_result, label, seq_len, lambda1, lambda2, lambda3):
        label = label.float()
        a_loss = self.compute_fusion_mil_loss(va_result, label)
        f_loss = self.compute_fusion_mil_loss(vf_result, label)
        raf_loss = self.compute_fusion_mil_loss(hcf_result, label)
        ma_loss = self.compute_alignment_loss(v_result, va_result, vf_result, seq_len)
        ma_loss = ma_loss + 0.01 * (a_loss + f_loss)
        triplet_loss = self.compute_triplet_loss(hcf_result, label, seq_len)
        total_loss = lambda1 * ma_loss + lambda2 * raf_loss + lambda3 * triplet_loss
        loss_dict = {}
        loss_dict['L_align'] = ma_loss
        loss_dict['L_fusion'] = raf_loss
        loss_dict['L_triplet'] = triplet_loss
        loss_dict['MA_loss'] = ma_loss
        loss_dict['M_MIL_loss'] = raf_loss
        loss_dict['Triplet_loss'] = triplet_loss
        return (total_loss, loss_dict)