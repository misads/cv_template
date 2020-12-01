# encoding: utf-8
import torch
import ipdb
import cv2
import numpy as np
from options import opt
# from dataloader import paired_dataset
from mscv.summary import create_summary_writer, write_image
from mscv.image import tensor2im

from dataloader.dataloaders import train_dataloader, val_dataloader

import misc_utils as utils

import random

"""
source domain 是clear的
"""
writer = create_summary_writer('logs/preview')

"""
这个改成需要预览的数据集
"""
previewed = val_dataloader  # train_dataloader, val_dataloader


for i, sample in enumerate(previewed):
    # if i > 30:
    #     break
    utils.progress_bar(i, len(previewed), 'Handling...')
    if opt.debug:
        ipdb.set_trace()

    image = sample['input'][0].detach().cpu().numpy().transpose([1,2,0])
    image = (image.copy()*255).astype(np.uint8)

    label = sample['label'][0].detach().cpu().numpy().transpose([1,2,0])
    label = (label.copy()*255).astype(np.uint8)

    write_image(writer, f'preview_{opt.dataset}/{i}', '0_input', image, 0, 'HWC')
    write_image(writer, f'preview_{opt.dataset}/{i}', '1_label', label, 0, 'HWC')

writer.flush()