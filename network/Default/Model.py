import pdb

import numpy as np
import torch
import os

from .FFA import FFA

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from loss import get_default_loss


import misc_utils as utils


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.cleaner = FFA().to(device=opt.device)
        #####################
        #    Init weights
        #####################
        # normal_init(self.cleaner)

        print_network(self.cleaner)

        self.g_optimizer = get_optimizer(opt, self.cleaner)
        self.scheduler = get_scheduler(opt, self.g_optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, x, y):

        # L1 & SSIM loss
        cleaned = self.cleaner(x)

        loss = get_default_loss(cleaned, y, self.avg_meters)  # or altered with your custom loss

        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return {'recovered': cleaned}

    def forward(self, x):
        return self.cleaner(x)

    def inference(self, x, image=None):
        pass

    def load(self, ckpt_path):
        load_dict = {
            'cleaner': self.cleaner,
        }

        if opt.resume:
            load_dict.update({
                'optimizer': self.g_optimizer,
                'scheduler': self.scheduler,
            })
            utils.color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            utils.color_print('Load checkpoint from %s.' % ckpt_path, 3)

        ckpt_info = load_checkpoint(load_dict, ckpt_path, map_location=opt.device)
        epoch = ckpt_info.get('epoch', 0)

        return epoch

    def save(self, which_epoch):
        save_filename = f'{which_epoch}_{opt.model}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {
            'cleaner': self.cleaner,
            'optimizer': self.g_optimizer,
            'scheduler': self.scheduler,
            'epoch': which_epoch
        }

        save_checkpoint(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)



