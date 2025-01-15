import torch
import torch.nn as nn
from einops import rearrange

from .units import *
from .utils import bit_plane_slicing, floor_func


class HKNet(nn.Module):

    def __init__(self, msb='hdb', lsb='hd', nf=64, upscale=1, act=nn.ReLU, intermediate=False):
        super(HKNet, self).__init__()

        self.msb = msb
        self.lsb = lsb
        self.nf = nf
        self.upscale = upscale
        self.act = act
        self.intermediate = intermediate

        # msb
        if msb=='hd':
            msb_unit = HDUnit
        elif msb=='hdb':
            msb_unit = HDBUnit
        else:
            msb_unit = HDBPlusUnit

        self.msb_rot_dict = msb_unit.rot_dict
        self.msb_pad_dict = msb_unit.pad_dict
        self.msb_avg_factor = msb_unit.avg_factor
        for ktype in msb:
            setattr(self, f'msb_{msb}_lut_{ktype}', msb_unit(ktype=ktype, nf=nf, upscale=upscale, act=act))

        # lsb
        if lsb=='hd':
            lsb_unit = HDUnit
        elif lsb=='hdb':
            lsb_unit = HDBUnit
        else:
            lsb_unit = HDBPlusUnit

        self.lsb_rot_dict = lsb_unit.rot_dict
        self.lsb_pad_dict = lsb_unit.pad_dict
        self.lsb_avg_factor = lsb_unit.avg_factor
        for ktype in lsb:
            setattr(self, f'lsb_{lsb}_lut_{ktype}', lsb_unit(ktype=ktype, nf=nf, upscale=upscale, act=act))

    def lut_forward(self, x, branch, ktype):
        unit = self.msb if branch == 'msb' else self.lsb
        lut = getattr(self, f'{branch}_{unit}_lut_{ktype}')
        return lut(x)

    def forward(self, x):
        # Prepare inputs for two branches

        y = x[:,0:1]
        cb = x[:,1:2]
        cr = x[:,2:3]

        batch_L255 = torch.floor(y * 255)
        MSB, LSB = bit_plane_slicing(batch_L255, bit_mask='11110000')
        MSB = MSB / 255.0
        LSB = LSB / 255.0

        bias = 127
        # MSB
        MSB_out = 0.0
        for ktype in self.msb:
            for r in self.msb_rot_dict[ktype]:
                batch = self.lut_forward(F.pad(torch.rot90(MSB, r, [2, 3]), self.msb_pad_dict[ktype], mode='reflect'), branch='msb', ktype=ktype)
                batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                MSB_out += floor_func(batch)

        MSB_out = MSB_out / self.msb_avg_factor / 255.
        # MSB_out = torch.tanh(MSB_out)

        # LSB
        LSB_out = 0.0
        for ktype in self.lsb:
            for r in self.lsb_rot_dict[ktype]:
                batch = self.lut_forward(F.pad(torch.rot90(LSB, r, [2,3]), self.lsb_pad_dict[ktype], mode='reflect'), branch='lsb', ktype=ktype)
                batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                LSB_out += floor_func(batch)
        LSB_out = LSB_out / self.lsb_avg_factor / 255.
        # LSB_out = torch.tanh(LSB_out)

        sb = torch.tanh(MSB_out + LSB_out)

        y = F.interpolate(y, scale_factor=self.upscale, mode='bilinear')
        cb = F.interpolate(cb, scale_factor=self.upscale, mode='bilinear')
        cr = F.interpolate(cr, scale_factor=self.upscale, mode='bilinear')

        if self.intermediate:
            return y, cb, cr, sb

        y = torch.concat([y + sb, cb, cr], axis=1)

        return y
