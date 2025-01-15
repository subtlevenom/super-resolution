import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import numpy_bit_plane_slicing, decode_bit_mask
from .luts import HDLUT, HDBLUT, HDTBLUT 


class HKLut(nn.Module):

    def __init__(self,
                 msb_weights,
                 lsb_weights,
                 msb='hdb',
                 lsb='hd',
                 upscale=2):
        super(HKLut, self).__init__()

        self.upscale = upscale
        self.bit_mask = '11110000'
        self.msb_bits, self.lsb_bits, self.msb_step, self.lsb_step = decode_bit_mask(
            self.bit_mask)

        # MSB
        if msb == 'hd':
            msb_lut = HDLUT
        elif msb == 'hdb':
            msb_lut = HDBLUT
        else:
            msb_lut = HDTBLUT

        self.msb_lut = msb_lut(msb_weights, 2**self.msb_bits, upscale=upscale)

        # LSB
        if lsb == 'hd':
            lsb_lut = HDLUT
        elif lsb == 'hdb':
            lsb_lut = HDBLUT
        else:
            lsb_lut = HDTBLUT

        self.lsb_lut = lsb_lut(lsb_weights, 2**self.lsb_bits, upscale=upscale)

    def forward(self, x):

        y = x[:, 0:1]
        cb = x[:, 1:2]
        cr = x[:, 2:3]

        y_np = (y.cpu().data.numpy()).squeeze()
        img_lr_255 = np.floor(y_np * 255)
        img_lr_msb, img_lr_lsb = numpy_bit_plane_slicing(img_lr_255, self.bit_mask)

        # msb
        img_lr_msb = np.floor_divide(img_lr_msb, self.msb_step)
        MSB_out = self.msb_lut(img_lr_msb) / 255.

        # lsb
        img_lr_lsb = np.floor_divide(img_lr_lsb, self.lsb_step)
        LSB_out = self.lsb_lut(img_lr_lsb) / 255.

        sb = np.tanh(MSB_out + LSB_out)
        sb = torch.Tensor(np.expand_dims(np.transpose(np.expand_dims(sb, axis=2), [2, 0, 1]), axis=0)).to('cuda:0')

        y = F.interpolate(y, scale_factor=self.upscale, mode='bilinear')
        cb = F.interpolate(cb, scale_factor=self.upscale, mode='bilinear')
        cr = F.interpolate(cr, scale_factor=self.upscale, mode='bilinear')

        y = torch.cat([y+sb, cb, cr], axis=1)

        return torch.clamp(y, 0, 1)
