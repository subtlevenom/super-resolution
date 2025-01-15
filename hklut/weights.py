import os
from pathlib import Path
import torch
import numpy as np

from hklut.hknet import HKNet
from hklut.utils import decode_bit_mask


pixel_dict = {'hdtb': 4, 'hdb': 3, 'hd': 2}


def process_lut(model, kind, bits, step, bit_order, device, lut_path):
    luts = filter(lambda a: kind in a and 'lut' in a , dir(model))
    for lut in luts:
        unit = model.__getattr__(lut)
        lut_input = get_input_tensor(bits, step, n_pixels=pixel_dict[bit_order])

        lut_input = unit.get_lut_input(lut_input).to(device)

        batch_size = 1000
        num_batches = (lut_input.size(0) + batch_size -1) // batch_size
        outputs = []

        for b in range(num_batches):
            batch_input = lut_input[b * batch_size:(b + 1) * batch_size]
            batch_output = unit(batch_input)
            results = torch.floor(torch.clamp(batch_output, -1, 1) * 127)
            results = results.data.numpy().astype(np.int8)
            outputs.append(results)
        
        outputs = np.concatenate(outputs, 0)
        B = lut[-1].upper()
        path_to_save = lut_path.joinpath(f'{kind.upper()}_{bit_order.upper()}_LUT_{B}_x{model.upscale}_4bit_int8.npy')
        np.save(path_to_save, outputs)
        print("Resulting LUT size: ", outputs.shape, "Saved to", path_to_save)


def get_input_tensor(bits, base_steps, n_pixels=3):
    L = 2**bits
    base_step_ind = torch.arange(0, L, 1)
    base = base_steps * base_step_ind / 255.0
    index_nD = torch.meshgrid(*[base for _ in range(n_pixels)])
    input_tensor = torch.cat(
        [index_nD[i].flatten().unsqueeze(1) for i in range(len(index_nD))],
        1).unsqueeze(1)
    return input_tensor 


def transfer_weights(model: HKNet, lut_path: Path):
    device = next(model.parameters()).device
    model.eval()
    msb_bits, lsb_bits, msb_step, lsb_step = decode_bit_mask('11110000')

    process_lut(model, 'msb', msb_bits, msb_step, model.msb, device, lut_path)
    process_lut(model, 'lsb', lsb_bits, lsb_step, model.lsb, device, lut_path)


def load_weights(msb, lsb, upscale, lut_path: Path):
    msb_weights = []
    for ktype in msb:
        weight = np.load(
                os.path.join(
                    lut_path,
                    f'MSB_{msb.upper()}_LUT_{ktype.upper()}_x{upscale}_4bit_int8.npy'
                )).astype(np.int_)
        msb_weights.append(weight)

    lsb_weights = []
    for ktype in lsb:
        weight = np.load(
                os.path.join(
                    lut_path,
                    f'LSB_{lsb.upper()}_LUT_{ktype.upper()}_x{upscale}_4bit_int8.npy'
                )).astype(np.int_)
        lsb_weights.append(weight)

    return msb_weights, lsb_weights
