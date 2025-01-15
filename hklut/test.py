import time
import os

import numpy as np
import torch
import torch.nn.functional as F

from utils import _ycbcr2rgb, _rgb2ycbcr
from PIL import Image

from luts import LUT

STAGES = 1
SCALE = [2]
LUT_FOLDER = "/wd/HKLUT/luts/final_luts/x{}".format(SCALE[0])
MSB = "HDB"
LSB = "HD"
TEST_DATASET = [
    "/wd/data/input_images/x{}".format(SCALE[0])
]

if __name__ == "__main__":

    # Load LUTs
    print("LUT path: ", LUT_FOLDER)

    LUTs = []

    for stage in range(STAGES):
        # msb
        msb_weights = []
        for ktype in MSB:
            weight = np.load(os.path.join(LUT_FOLDER, f'S{stage}_MSB_{MSB}_LUT_{ktype.upper()}_x{SCALE[stage]}_4bit_int8.npy')).astype(
                np.int_)
            msb_weights.append(weight)

        # lsb
        lsb_weights = []
        for ktype in LSB:
            weight = np.load(os.path.join(LUT_FOLDER, f'S{stage}_LSB_{LSB}_LUT_{ktype.upper()}_x{SCALE[stage]}_4bit_int8.npy')).astype(
                np.int_)
            lsb_weights.append(weight)

        LUTs.append(LUT(msb_weights, lsb_weights, SCALE[stage]))

    save_path = '/wd/data/output_images/x{}/'.format(np.prod(SCALE))

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for j in range(len(TEST_DATASET)):

        times = []

        folder = TEST_DATASET[j]

        files = os.listdir(folder)
        files.sort()

        for file in files:

            input_im = Image.open(os.path.join(folder, file))

            input_im = np.array(input_im)

            input_im = _rgb2ycbcr(np.array(input_im))

            cb = input_im[:, :, 1:2]
            cr = input_im[:, :, 2:3]

            input_im = input_im[:, :, 0:1]

            cb = cb.astype(np.float32) / 255.0
            cr = cr.astype(np.float32) / 255.0

            input_im = input_im.astype(np.float32) / 255.0

            start = time.time()

            # Output
            input_im = input_im.squeeze()
            for stage in range(STAGES):
                input_im = LUTs[stage].inference(input_im)
            image_out = input_im

            times.append(time.time() - start)

            cb_tensor = torch.from_numpy(cb.transpose([2, 0, 1])).unsqueeze(0)
            cr_tensor = torch.from_numpy(cr.transpose([2, 0, 1])).unsqueeze(0)

            if SCALE != 1:

                cb_upscaled = F.interpolate(cb_tensor, size=(cb.shape[0] * np.prod(SCALE), cb.shape[1] * np.prod(SCALE)),
                                            mode='bilinear', antialias=True)
                cr_upscaled = F.interpolate(cr_tensor, size=(cr.shape[0] * np.prod(SCALE), cr.shape[1] * np.prod(SCALE)),
                                            mode='bilinear', antialias=True)

            else:

                cb_upscaled = cb_tensor
                cr_upscaled = cr_tensor

            image_out = np.expand_dims(image_out*255, axis=2)
            image_out = np.clip(image_out, 0, 255)

            cb_upscaled = np.round(cb_upscaled.cpu().data.numpy().squeeze(0) * 255)
            cb_upscaled = np.transpose(np.clip(cb_upscaled, 0, 255), [1, 2, 0])

            cr_upscaled = np.round(cr_upscaled.cpu().data.numpy().squeeze(0) * 255)
            cr_upscaled = np.transpose(np.clip(cr_upscaled, 0, 255), [1, 2, 0])

            image_out = np.stack((image_out, cb_upscaled, cr_upscaled), axis=2).squeeze()

            image_out = _ycbcr2rgb(image_out)

            # Save to file
            image_out = image_out.astype(np.uint8)
            Image.fromarray(image_out).save(
                save_path + '{}_x{}.png'.format(file.split('/')[-1], np.prod(SCALE)))

        print(np.array(times).mean())
