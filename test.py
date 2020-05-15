# encoding=utf-8
"""
  python3 test.py --tag complex_256 --dataset complex --load checkpoints/complex/500_checkpoint.pt

"""

import os, sys
import pdb

# from dataloader.image_folder import get_data_loader_folder
from torch_template.dataloader.tta import OverlapTTA

import dataloader as dl
from network import get_model
from options import opt
from misc_utils import get_file_name

sys.path.insert(1, '../')
import torch
import torchvision
import numpy as np
from PIL import Image

import misc_utils as utils

if not opt.load:
    raise Exception('Checkpoint must be specified at test phase, try --load <checkpoint_dir>')


result_dir = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
utils.try_make_dir(result_dir)

Model = get_model(opt.model)
model = Model(opt)

model = model.cuda(device=opt.device)
model.eval()


for i, data in enumerate(dl.test_dataloader):
    print('Testing image %d' % i)
    img, paths = data['input'], data['path']
    """
    Test Codes
    """
