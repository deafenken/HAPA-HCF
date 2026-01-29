import os
import torch
import numpy as np
import torch.utils.data as data
import cfg.options_test as opt
from run.train import *
from run.test import *
from util.utils import *
from loss.uapahcf_losses import *
from models.unimodal import *
from models.hcf_module import HCFNet
from models.projection import *
from models.uapa_module import UAPA
from data.dataset_loader import *
if __name__ == '__main__':
    args = opt.parser.parse_args()
    args = opt.init_args(args)
    test_loader = data.DataLoader(Dataset(args, test_mode=True), batch_size=5, shuffle=False, num_workers=args.workers, pin_memory=True)
    v_net = Unimodal(input_size=1024, h_dim=128, feature_dim=128)
    a_net = Unimodal(input_size=128, h_dim=64, feature_dim=32)
    f_net = Unimodal(input_size=1024, h_dim=128, feature_dim=64)
    v_net.load_state_dict(torch.load(os.path.join(args.save_model_path, 'v_model.pth')))
    a_net.load_state_dict(torch.load(os.path.join(args.save_model_path, 'a_model.pth')))
    f_net.load_state_dict(torch.load(os.path.join(args.save_model_path, 'f_model.pth')))
    v_net = v_net.cuda()
    a_net = a_net.cuda()
    f_net = f_net.cuda()
    va_net = Projection(32, 32, 32)
    vf_net = Projection(64, 64, 64)
    va_net.load_state_dict(torch.load(os.path.join(args.save_model_path, 'va_model.pth')))
    vf_net.load_state_dict(torch.load(os.path.join(args.save_model_path, 'vf_model.pth')))
    va_net = va_net.cuda()
    vf_net = vf_net.cuda()
    ra_alignment = UAPA(main_dim=128, aux_dim=32).cuda()
    rf_alignment = UAPA(main_dim=128, aux_dim=64).cuda()
    ra_alignment.load_state_dict(torch.load(os.path.join(args.save_model_path, 'ra_alignment.pth')))
    rf_alignment.load_state_dict(torch.load(os.path.join(args.save_model_path, 'rf_alignment.pth')))
    hcf_net = HCFNet(input_size=384, h_dim=128, feature_dim=64)
    hcf_net.load_state_dict(torch.load(os.path.join(args.save_model_path, 'hcf_model.pth')))
    hcf_net = hcf_net.cuda()
    test_info = {'iteration': [], 'm_ap': []}
    gt = np.load(args.gt)
    test(v_net, a_net, f_net, va_net, vf_net, hcf_net, ra_alignment, rf_alignment, test_loader, gt, test_info, 0)
    best_ap = test_info['m_ap'][-1]
    utils.save_best_record(test_info, os.path.join(args.output_path, 'result.txt'))
    print('AP: {:.2f}'.format(best_ap * 100))