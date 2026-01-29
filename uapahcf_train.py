import os
import torch
import numpy as np
import torch.utils.data as data
import cfg.options as options
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
    args = options.parser.parse_args()
    args.save_model_path = f'saved_models/seed_{args.seed}/'
    args = options.init_args(args)
    set_seed(args.seed)
    (lambda1, lambda2, lambda3) = (args.lambda1, args.lambda2, args.lambda3)
    test_loader = data.DataLoader(Dataset(args, test_mode=True), batch_size=5, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_loader = data.DataLoader(Dataset(args, test_mode=False), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    v_net = Unimodal(input_size=1024, h_dim=128, feature_dim=128)
    a_net = Unimodal(input_size=128, h_dim=64, feature_dim=32)
    f_net = Unimodal(input_size=1024, h_dim=128, feature_dim=64)
    v_net = v_net.cuda()
    a_net = a_net.cuda()
    f_net = f_net.cuda()
    va_net = Projection(32, 32, 32)
    vf_net = Projection(64, 64, 64)
    va_net = va_net.cuda()
    vf_net = vf_net.cuda()
    ra_alignment = UAPA(main_dim=128, aux_dim=32).cuda()
    rf_alignment = UAPA(main_dim=128, aux_dim=64).cuda()
    hcf_net = HCFNet(input_size=384, h_dim=128, feature_dim=64)
    hcf_net = hcf_net.cuda()
    optimizer = torch.optim.Adam(list(v_net.parameters()) + list(a_net.parameters()) + list(f_net.parameters()) + list(va_net.parameters()) + list(vf_net.parameters()) + list(hcf_net.parameters()) + list(ra_alignment.parameters()) + list(rf_alignment.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    criterion = UnimodalMILLoss()
    criterion_disl = UAPAHCFLoss(ra_alignment=ra_alignment, rf_alignment=rf_alignment)
    best_ap = 0.0
    test_info = {'iteration': [], 'm_ap': []}
    gt = np.load(args.gt)
    for step in range(1, args.num_steps + 1):
        if (step - 1) % len(train_loader) == 0:
            train_loader_iter = iter(train_loader)
        (loss_dict_list, loss_dict_list_disl) = train(v_net, a_net, f_net, va_net, vf_net, hcf_net, ra_alignment, rf_alignment, train_loader_iter, optimizer, criterion, criterion_disl, step, lambda1, lambda2, lambda3)
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"Step: {step}, U_MIL_loss: {loss_dict_list['U_MIL_loss']:.6f}, MA_loss: {loss_dict_list_disl['MA_loss']:.6f}, M_MIL_loss: {loss_dict_list_disl['M_MIL_loss']:.6f}, Triplet_loss: {loss_dict_list_disl['Triplet_loss']:.6f}, LR: {current_lr:.6f} ")
        if step % 10 == 0:
            test_ap = test(v_net, a_net, f_net, va_net, vf_net, hcf_net, ra_alignment, rf_alignment, test_loader, gt, test_info, step)
            if test_info['m_ap'][-1] > best_ap:
                best_ap = test_info['m_ap'][-1]
                utils.save_best_record(test_info, os.path.join(args.output_path, 'best_record_{}.txt'.format(args.seed)))
                torch.save(v_net.state_dict(), os.path.join(args.save_model_path, 'v_model.pth'))
                torch.save(a_net.state_dict(), os.path.join(args.save_model_path, 'a_model.pth'))
                torch.save(f_net.state_dict(), os.path.join(args.save_model_path, 'f_model.pth'))
                torch.save(va_net.state_dict(), os.path.join(args.save_model_path, 'va_model.pth'))
                torch.save(vf_net.state_dict(), os.path.join(args.save_model_path, 'vf_model.pth'))
                torch.save(hcf_net.state_dict(), os.path.join(args.save_model_path, 'hcf_model.pth'))
                torch.save(ra_alignment.state_dict(), os.path.join(args.save_model_path, 'ra_alignment.pth'))
                torch.save(rf_alignment.state_dict(), os.path.join(args.save_model_path, 'rf_alignment.pth'))