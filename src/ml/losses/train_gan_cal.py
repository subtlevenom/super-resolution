# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import os
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter

from models import *
from data import Provider, SRBenchmark
from utils import PSNR, _rgb2ycbcr, seed_everything
from losses import L1Loss, PerceptualLoss, GANLoss, lda_loss, cos_similarity, load_balancing_loss

device = 'cuda:1' 

chekpoint_folder = 'checkpoint_Y_upscale_CAL_GAN_ft_23'

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
        
class OrthorTransform(nn.Module):
    def __init__(self, c_dim, feat_hw, groups): #feat_hw: width or height (let width == heigt)
        super(OrthorTransform, self).__init__()

        self.groups = groups
        self.c_dim = c_dim
        self.feat_hw = feat_hw
        self.weight = nn.Parameter(torch.randn(1, feat_hw, c_dim))
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.99))    
    def forward(self, feat):
        pred = feat * self.weight.expand_as(feat)
        return pred, self.weight.view(self.groups, -1)

class CodeReduction(nn.Module):
    def __init__(self, c_dim, feat_hw, blocks = 4, prob=False):
        super(CodeReduction, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(c_dim, c_dim*blocks),
            nn.LeakyReLU(0.2, True)
        )
        self.trans = OrthorTransform(c_dim=c_dim*blocks, feat_hw=feat_hw, groups = blocks)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
    def forward(self, feat):
        feat = self.body(feat)
        feat, weight = self.trans(feat)
        feat = self.leakyrelu(feat)
        return feat, weight

# num_in_ch: 3
# num_feat: 64
# num_expert: 12
class MOD(nn.Module):

    def __init__(self, num_in_ch=3, num_feat=64, num_expert=12):
        super(MOD, self).__init__()
        self.num_expert = num_expert
        self.num_feat = num_feat

        self.FE = nn.Sequential(
            nn.Conv2d(3, num_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat,num_feat, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat,num_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat*2,num_feat*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_feat*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat*2, num_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat*4,num_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat*4,num_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_feat*4),
            nn.LeakyReLU(0.2, True),
        )
        
        self.w_gating1 = nn.Parameter(torch.randn(num_feat*4, self.num_expert))

        m_classifier = [
            nn.Linear(num_feat*4, num_feat//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(num_feat//2, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)
        self.classifiers = nn.ModuleList()
        for _ in range(self.num_expert):
            self.classifiers.append(self.classifier)

        self.orthonet = CodeReduction(c_dim = num_feat*4, feat_hw =1, blocks=self.num_expert)

    def forward(self, x, routing = None):
        feature = self.FE(x)
        B, C, H, W = feature.shape
        feature = feature.view(B, -1, H*W).permute(0,2,1)
        if routing == None:
            routing = torch.einsum('bnd,de->bne', feature, self.w_gating1)
            routing = routing.softmax(dim=-1)

        feature, ortho_weight = self.orthonet(feature)
        feature = torch.split(feature, [feature.shape[-1]//self.num_expert]*self.num_expert, dim = -1)

        # soft routing
        # output =  self.classifiers[0](feature[0]) * routing[:,:,[0]]
        # for i in range(1, self.num_expert1):
        #     output = output + self.classifiers[i](feature[i]) * routing[:,:,[i]]
        
        # hard routing
        routing_top = torch.max(routing, dim=-1)[1].unsqueeze(-1).float() 
        for i in range(self.num_expert):
            if i==0:
                output = self.classifiers[0](feature[0])
            else:
                output = torch.where(routing_top == i, self.classifiers[i](feature[i]), output)
        return output, routing, feature, ortho_weight


import lpips

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import joblib
brisque_model = joblib.load('/wd/HKLUT/brisque/tid2008.pkl')


def parse_args():
    parser = argparse.ArgumentParser("Training Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int,  default=8)
    parser.add_argument("--train-dir", type=str, default='/data/datasets/blurred_downscaled/mix01-1/train_23/',
                        help="Training images")
    parser.add_argument("--val-dir", type=str, default='/data/datasets/blurred_downscaled/mix01-1/valid_23/',
                        help="Validation images")
    parser.add_argument("--i-display", type=int, default=2500000,
                        help="display info every N iteration")
    parser.add_argument("--i-validate", type=int, default=5000,
                        help="validation every N iteration")
    parser.add_argument("--i-save", type=int, default=5000,
                        help="save checkpoints every N iteration")

    parser.add_argument("--upscale", nargs='+', type=int, default=[2],
                        help="upscaling factors")
    parser.add_argument("--crop-size", type=int, default=64,
                        help="input LR training patch size")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="training batch size")
    parser.add_argument("--start-iter", type=int, default=250000,
                        help="Set 0 for from scratch, else will load saved params and trains further")
    parser.add_argument("--train-iter", type=int, default=300000,
                        help="number of training iterations")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--wd', type=float, default=0,  help='weight decay')

    parser.add_argument('--msb', type=str, default='hdb', choices=['hdtb', 'hdb', 'hd'])
    parser.add_argument('--lsb', type=str, default='hd', choices=['hdtb', 'hdb', 'hd'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['silu', 'relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    parser.add_argument("--device", type=str, default=device, help="device")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb:{}-lsb:{}-act:{}-nf:{}-{}".format(args.msb, args.lsb, args.act_fn, args.n_filters, factors)

    act_fn_dict = {'silu': nn.SiLU, 'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': nn.LeakyReLU, 'starrelu': StarReLU}
    args.act_fn = act_fn_dict[args.act_fn]

    return args

def SaveCheckpoint(models, opt_G, i, args, best=False):
    if best:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), '/wd/HKLUT/{}/{}/model_G_S{}_best.pth'.format(chekpoint_folder, args.exp_name, stage))
            else:
                torch.save(model.state_dict(), '/wd/HKLUT/{}/{}/model_G_S{}_best.pth'.format(chekpoint_folder,args.exp_name, stage))
        torch.save(opt_G.state_dict(), '/wd/HKLUT/{}/{}/opt_G_best.pth'.format(chekpoint_folder,args.exp_name))
        print("Best checkpoint saved")
    else:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), '/wd/HKLUT/{}/{}/model_G_S{}_i{:06d}.pth'.format(chekpoint_folder,args.exp_name, stage, i))
            else:
                torch.save(model.state_dict(), '/wd/HKLUT/{}/{}/model_G_S{}_i{:06d}.pth'.format(chekpoint_folder,args.exp_name, stage, i))
        torch.save(opt_G.state_dict(), '/wd/HKLUT/{}/{}/opt_G_i{:06d}.pth'.format(chekpoint_folder,args.exp_name, i))
        print("Checkpoint saved {}".format(str(i)))


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    print(args)
    args.seed = 42
    seed_everything(args.seed)

    ### Tensorboard for monitoring ###
    writer = SummaryWriter(log_dir='./log/{}'.format(args.exp_name))

    models = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)
    
    for s in args.upscale:
        models.append(HKNet(msb=args.msb, lsb=args.lsb, nf=args.n_filters, upscale=s, act=args.act_fn).to(device))

    model_D = MOD().to(device)

    params_D = list(filter(lambda p: p.requires_grad, model_D.parameters()))
    opt_D = optim.Adam(params_D, lr=1e-5)
    scheduler_d = optim.lr_scheduler.MultiStepLR(opt_D, milestones=[260000, 280000], gamma=0.1)

    ## Optimizers
    opt_G = optim.Adam([{'params': list(filter(lambda p: p.requires_grad, model.parameters()))} for model in models], 
                       lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd, eps=1e-8, amsgrad=False)

    scheduler = optim.lr_scheduler.MultiStepLR(opt_G, milestones=[100000, 270000], gamma=0.1)

    ## Prepare directories
    if not os.path.isdir('/wd/HKLUT/{}'.format(chekpoint_folder)):
        os.mkdir('/wd/HKLUT/{}'.format(chekpoint_folder))
    if not os.path.isdir('/wd/HKLUT/{}/{}'.format(chekpoint_folder, args.exp_name)):
        os.mkdir('/wd/HKLUT/{}/{}'.format(chekpoint_folder, args.exp_name))
    if not os.path.isdir('/wd/HKLUT/{}/{}/'.format(chekpoint_folder, args.exp_name)):
        os.mkdir('/wd/HKLUT/{}/{}/'.format(chekpoint_folder, args.exp_name))
    if not os.path.isdir('log'):
        os.mkdir('log')

    ## Load saved params
    if args.start_iter > 0:
        for stage in range(n_stages):
            lm = torch.load('/wd/HKLUT/{}/{}/model_G_S{}_i{:06d}.pth'.format(chekpoint_folder, args.exp_name, stage, args.start_iter))
            models[0].load_state_dict(lm, strict=True)

        # lm = torch.load('/wd/HKLUT/{}/{}/opt_G_i{:06d}.pth'.format(chekpoint_folder, args.exp_name, args.start_iter))
        # opt_G.load_state_dict(lm)
    
    # Training dataset
    train_loader = Provider(args.batch_size, args.n_workers, sr_scale, args.train_dir, args.crop_size)

    # Validation dataset
    valid_datasets = [args.val_dir]
    valid_loader = SRBenchmark(args.val_dir, datasets=valid_datasets, scale=sr_scale)

    # l_accum = [0.,0.,0.]
    l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    dT = 0.
    rT = 0.
    n_mix = 0
    accum_samples = 0

    # losses
    # Pixel with weight 0.01
    cri_pix = L1Loss(1e-2)

    # Perxeptual loss with weight 1.0
    cri_percep = PerceptualLoss(
        {'conv1_2': 0.1,
         'conv2_2': 0.1,
         'conv3_4': 1,
         'conv4_4': 1,
         'conv5_4': 1
        },
        vgg_type='vgg19',
        use_input_norm=True,
        range_norm=False,
        perceptual_weight=1.0,
        style_weight=0,
        criterion='l1'
    )

    # GAN loss with weight 0.005
    cri_gan = GANLoss(gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=5e-3)

    ### TRAINING
    best_psnr = 0.0
    for i in tqdm(range(args.start_iter+1, args.train_iter+1)):
        
        model_D.train()

        for model in models:
            model.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H = train_loader.next()

        cb = batch_L[:,1:2,:,:]    
        cr = batch_L[:,2:3,:,:]
        batch_L = batch_L[:,0:1,:,:]

        batch_H = batch_H.to(device)      # BxCxHxW (32, 3, 192, 192), range [0,1]
        batch_L = batch_L.to(device)      # BxCxHxW (32, 3, 48, 48), range [0,1]        
    
        dT += time.time() - st

        ## TRAIN G
        st = time.time()

        for p in model_D.parameters():
            p.requires_grad = False
        
        opt_G.zero_grad()

        for model in models:
            batch_S = model(batch_L)
        batch_S = torch.clamp(batch_S, 0, 1)  # [-2, 2] -> [0, 1]

        if sr_scale != 1:

            cb = F.interpolate(cb, scale_factor=sr_scale,  mode='bilinear', antialias=True)
            cr = F.interpolate(cr, scale_factor=sr_scale,  mode='bilinear', antialias=True)
        
        batch_S = torch.cat([batch_S, cb, cr], dim=1)

        l_g_total = 0

        # pixel loss
        if cri_pix:

            l_g_pix = cri_pix(batch_S, batch_H)
            l_g_total += l_g_pix

        # perceptual loss
        if cri_percep:
            l_g_percep, l_g_style = cri_percep(batch_S, batch_H)

            if l_g_percep is not None:
                l_g_total += l_g_percep

            if l_g_style is not None:
                l_g_total += l_g_style

        # gan loss (relativistic gan)
        real_d_pred, routing, _, _ = model_D(batch_H)
        real_d_pred = real_d_pred.detach()
        fake_g_pred, _, _, _ = model_D(batch_S, routing.detach())
        l_g_real = cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
        l_g_fake = cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
        l_g_gan = (l_g_real + l_g_fake) / 2
        l_g_total += l_g_gan

        # Update
        l_g_total.backward()
        opt_G.step()

        # optimize net_d
        for p in model_D.parameters():
            p.requires_grad = True

        torch.autograd.set_detect_anomaly(True)
        opt_D.zero_grad()
        
        # Real
        real_d_pred, routing, feature, weight = model_D(batch_H)
        fake_d_pred, _, _, _ = model_D(batch_H, routing.detach())
        fake_d_pred = fake_d_pred.detach()
        # adversarial loss
        l_d_real = cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        # orthogonal loss
        l_d_real += cos_similarity(weight) * 10.
        # LDA loss
        l_d_real += lda_loss(feature) * 10.
        # load_balancing_loss
        l_d_real += load_balancing_loss(routing) * 0.05
        l_d_real.backward()

        # fake
        fake_d_pred, _, _, _ = model_D(batch_S.detach(), routing.detach())
        l_d_fake = cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        opt_D.step()

        scheduler.step()
        scheduler_d.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += args.batch_size
        accum_samples += args.batch_size

        ## Show information
        if i % args.i_display == 0:
            writer.add_scalar('loss_Pixel', l_accum[0]/args.i_display, i)
            print("Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(i, accum_samples, l_accum[0]/args.i_display, dT/args.i_display, rT/args.i_display))
            l_accum = [0.,0.,0.]
            dT = 0.
            rT = 0.

        ## Save models
        if i % args.i_save == 0:
            SaveCheckpoint(models, opt_G, i, args)

        ## Validation
        if i % args.i_validate == 0:

            print('Learning rate: {}'.format(opt_G.param_groups[0]['lr']))

            with torch.no_grad():
                for model in models:
                    model.eval()


                for j in range(len(valid_datasets)):
                    psnrs = []
                    # brisques = []
                    # lpipses_1 = []
                    # lpipses_2 = []
                    files = valid_loader.files[valid_datasets[j]]

                    for k in range(len(files)):
                        key = valid_datasets[j] + files[k][:-4]

                        img_gt = valid_loader.ims[key] # (512, 512, 3) range [0, 255]
                        input_im = valid_loader.ims[key + 'x%d' % sr_scale] # (128, 128, 3) range [0, 255]

                        input_im = input_im[:,:,0:1]

                        input_im = input_im.astype(np.float32)/255.0
                        val_L = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).to(device) # (1, 3, 128, 128)

                        x = val_L
                        for model in models:
                            x = model(x)

                        # Output 
                        image_out = (x).cpu().data.numpy() # (1, 3, 512, 512)
                        image_out = np.transpose(np.clip(image_out[0], 0. , 1.), [1,2,0]) # BxCxHxW -> HxWxC
                        image_out = (image_out*255.0).astype(np.uint8)

                        # PSNR on Y channel
                        # psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], sr_scale))
                        psnrs.append(PSNR(img_gt, image_out, sr_scale))

                        tensor_gt = torch.from_numpy(np.expand_dims(img_gt.transpose(2,0,1), 0))
                        tensor_out = torch.from_numpy(np.expand_dims(image_out.transpose(2,0,1), 0))

                        # lpipses_1.append(loss_fn_alex(tensor_gt, tensor_out))    
                        # lpipses_2.append(loss_fn_vgg(tensor_gt, tensor_out)) 

                    mean_psnr = np.mean(np.asarray(psnrs))
                    # mean_lpips_1 = np.mean(np.asarray(lpipses_1))
                    # mean_lpips_2 = np.mean(np.asarray(lpipses_2))

                    # save best psnr for set5
                    if mean_psnr > best_psnr:
                        best_psnr = np.mean(np.asarray(psnrs))
                        SaveCheckpoint(models, opt_G, i, args, best=True)
                    
                    print('Iter {} | Dataset {} | AVG Val PSNR: {:2f}\t |  Loss_G: {:.5f}\t |  Loss_D_real: {:.5f}\t |  Loss_D_fake: {:.5f}'.format(i, valid_datasets[j], mean_psnr, l_g_total, l_d_real, l_d_fake))
                    writer.add_scalar('PSNR_valid/{}'.format(valid_datasets[j]), mean_psnr, i)
                    writer.flush()

                    with open('/wd/HKLUT/{}/PSNRs_{}.txt'.format(chekpoint_folder, args.exp_name), 'a') as f:
                        f.write('Iteration {} - PSNR: {:2f}\t - Loss_G: {:.5f}\t -  Loss_D_real: {:.5f}\t -  Loss_D_fake: {:.5f}\n'.format(i, mean_psnr, l_g_total, l_d_real, l_d_fake))

    print(f'Best PSNR: {best_psnr}')