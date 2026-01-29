import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def test(v_net, a_net, f_net, va_net, vf_net, hcf_net, ra_alignment, rf_alignment, test_loader, gt, test_info, epoch):
    with torch.no_grad():
        v_net.eval()
        a_net.eval()
        f_net.eval()
        va_net.eval()
        vf_net.eval()
        hcf_net.eval()
        ra_alignment.eval()
        rf_alignment.eval()
        m_pred = torch.zeros(0).cuda()
        for (i, (f_v, f_a, f_f)) in tqdm(enumerate(test_loader)):
            v_data = f_v.cuda()
            a_data = f_a.cuda()
            f_data = f_f.cuda()
            v_res = v_net(v_data)
            a_res = a_net(a_data)
            f_res = f_net(f_data)
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1).cuda()
            v_feat = v_res['satt_f']
            a_feat = a_res['satt_f']
            f_feat = f_res['satt_f']
            a_aligned = ra_alignment(v_feat, a_feat, seq_len)
            f_aligned = rf_alignment(v_feat, f_feat, seq_len)
            hcf_input = torch.cat([v_feat, a_aligned, f_aligned], dim=-1)
            m_out = hcf_net(hcf_input)
            m_out = torch.mean(m_out['output'], 0)
            m_pred = torch.cat((m_pred, m_out))
        m_pred = list(m_pred.cpu().detach().numpy())
        (precision, recall, th) = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)
        test_info['iteration'].append(epoch)
        test_info['m_ap'].append(m_ap)
        print(f'Test Epoch: {epoch}, AP: {m_ap:.4f}')
        return m_ap