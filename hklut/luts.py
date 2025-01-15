import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_slice(img, x, y, rot):

    if rot == 0 or rot == 4:
        dx, dy = img.shape
        img = np.pad(img, (0, 2), mode='reflect').astype(np.int64)
        return img[x:x + dx, y:y + dy]
    elif rot == 1:
        img = img[:, ::-1].T
        dx, dy = img.shape
        img = np.pad(img, (0, 2), mode='reflect').astype(np.int64)
        return img[x:x + dx, y:y + dy]
    elif rot == 2:
        img = img[::-1, ::-1]
        dx, dy = img.shape
        img = np.pad(img, (0, 2), mode='reflect').astype(np.int64)
        return img[x:x + dx, y:y + dy]
    elif rot == 3:
        img = img[::-1, :].T
        dx, dy = img.shape
        img = np.pad(img, (0, 2), mode='reflect').astype(np.int64)
        return img[x:x + dx, y:y + dy]

class HDLUT(nn.Module):
    def __init__(self, lsb_weight, L, upscale=2):
        super(HDLUT, self).__init__()
        self.lsb_weight = lsb_weight
        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):

        out = np.zeros((img_lr.shape[0] * self.upscale, img_lr.shape[1] * self.upscale))

        for ktype in ['h', 'd']:
            for r in [0, 1, 2, 3]:
                img_a = get_slice(img_lr, 0, 0, r)
                if ktype == 'h':
                    weight = self.lsb_weight[0]
                    img_b = get_slice(img_lr, 0, 1, r)
                else:
                    weight = self.lsb_weight[1]
                    img_b = get_slice(img_lr, 1, 1, r)

                tmp = weight[img_a.flatten() * self.L + img_b.flatten()].reshape(
                    (img_a.shape[0], img_a.shape[1], self.upscale, self.upscale)).transpose(
                    (0, 2, 1, 3)).reshape((img_a.shape[0] * self.upscale, img_a.shape[1] * self.upscale))
                out = out + get_slice(tmp, 0, 0, 4 - r)

        return out / 2

        
class HDBLUT(nn.Module):
    def __init__(self, msb_weight, L, upscale=2):
        super(HDBLUT, self).__init__()
        self.msb_weight = msb_weight
        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):

        out = np.zeros((img_lr.shape[0] * self.upscale, img_lr.shape[1] * self.upscale))

        for ktype in ['h', 'd', 'b']:
            for r in [0, 1, 2, 3]:
                img_a = get_slice(img_lr, 0, 0, r)
                if ktype == 'h':
                    weight = self.msb_weight[0]
                    img_b = get_slice(img_lr, 0, 1, r)
                    img_c = get_slice(img_lr, 0, 2, r)
                elif ktype == 'd':
                    weight = self.msb_weight[1]
                    img_b = get_slice(img_lr, 1, 1, r)
                    img_c = get_slice(img_lr, 2, 2, r)
                else:
                    weight = self.msb_weight[2]
                    img_b = get_slice(img_lr, 1, 2, r)
                    img_c = get_slice(img_lr, 2, 1, r)

                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()].reshape(
                    (img_a.shape[0], img_a.shape[1], self.upscale, self.upscale)).transpose(
                    (0, 2, 1, 3)).reshape((img_a.shape[0] * self.upscale, img_a.shape[1] * self.upscale))
                out = out + get_slice(tmp, 0, 0, 4 - r)

        return out / 3

class HDTBLUT(nn.Module):
    def __init__(self, h_weight, d_weight, t_weight, b_weight, L, upscale=2):
        super(HDTBLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.t_weight = t_weight
        self.b_weight = b_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 't': [0, 1, 2, 3], 'b': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 3, 0, 3), 'd': (0, 3, 0, 3), 't': (0, 3, 0, 3), 'b': (0, 3, 0, 3)}

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 't', 'b']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='reflect').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                    img_d = img_in[:, :, 0:0+H, 3:3+W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]
                    img_d = img_in[:, :, 3:3+H, 3:3+W]
                elif ktype == 't':
                    weight = self.t_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 2:2+H, 1:1+W]
                    img_c = img_in[:, :, 3:3+H, 1:1+W]
                    img_d = img_in[:, :, 3:3+H, 2:2+W]
                else:
                    weight = self.b_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 2:2+W]
                    img_c = img_in[:, :, 1:1+H, 3:3+W]
                    img_d = img_in[:, :, 2:2+H, 3:3+W]

                tmp = weight[img_a.flatten()*self.L*self.L*self.L + img_b.flatten()*self.L*self.L + img_c.flatten()*self.L + img_d.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = torch.permute(tmp, (0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out / 4.