import torch
from tqdm import tqdm
import os

def train(v_net, a_net, f_net, va_net, vf_net, hcf_net, ra_alignment, rf_alignment, dataloader, optimizer, criterion, criterion_disl, index, lambda1, lambda2, lambda3):
    with torch.set_grad_enabled(True):
        (f_v, f_a, f_f, label) = next(dataloader)
        v_net.train()
        a_net.train()
        f_net.train()
        va_net.train()
        vf_net.train()
        hcf_net.train()
        ra_alignment.train()
        rf_alignment.train()
        v_net.load_state_dict(torch.load(os.path.join('./saved_models/seed_500/', 'v_model.pth')))
        a_net.load_state_dict(torch.load(os.path.join('./saved_models/seed_500/', 'a_model.pth')))
        f_net.load_state_dict(torch.load(os.path.join('./saved_models/seed_500/', 'f_model.pth')))
        hcf_net.load_state_dict(torch.load(os.path.join('./saved_models/seed_500/', 'hcf_model.pth')))
        ra_alignment.load_state_dict(torch.load(os.path.join('./saved_models/seed_500/', 'ra_alignment.pth')))
        rf_alignment.load_state_dict(torch.load(os.path.join('./saved_models/seed_500/', 'rf_alignment.pth')))
        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
        f_v = f_v[:, :torch.max(seq_len), :]
        f_a = f_a[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]
        v_data = f_v.cuda()
        a_data = f_a.cuda()
        f_data = f_f.cuda()
        label = label.cuda()
        v_predict = v_net(v_data, seq_len)
        a_predict = a_net(a_data, seq_len)
        f_predict = f_net(f_data, seq_len)
        (total_loss, loss_dict_list) = criterion(v_predict, a_predict, f_predict, label)
        v_feat = v_predict['satt_f']
        a_feat = a_predict['satt_f']
        f_feat = f_predict['satt_f']
        a_aligned = ra_alignment(v_feat, a_feat, seq_len)
        f_aligned = rf_alignment(v_feat, f_feat, seq_len)
        hcf_input = torch.cat([v_feat, a_aligned, f_aligned], dim=-1)
        va_output = a_net(a_feat, seq_len, em_flag=False)
        vf_output = f_net(f_feat, seq_len, em_flag=False)
        hcf_output = hcf_net(hcf_input, seq_len)
        (total_loss_disl, loss_dict_list_disl) = criterion_disl(v_predict, va_output, vf_output, hcf_output, label, seq_len.cuda(), lambda1, lambda2, lambda3)
        total_loss = total_loss + total_loss_disl
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return (loss_dict_list, loss_dict_list_disl)