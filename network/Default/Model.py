import pdb

import numpy as np
import torch
import os

from .FFA import FFA

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, save_checkpoint
from loss import criterionL1, criterionSSIM


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.cleaner = FFA().to(device=opt.device)
        #####################
        #    Init weights
        #####################
        # self.cleaner.apply(weights_init)

        print_network(self.cleaner)

        self.g_optimizer = get_optimizer(opt, self.cleaner)
        self.scheduler = get_scheduler(opt, self.g_optimizer)

        # load networks
        if opt.load:
            ckpt_path = opt.load
            self.load(ckpt_path)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, x, y):

        # L1 & SSIM loss
        cleaned = self.cleaner(x)
        ssim = - criterionSSIM(cleaned, y)
        ssim_loss = ssim * opt.weight_ssim

        # Compute L1 loss (not used)
        l1_loss = criterionL1(cleaned, y)
        l1_loss = l1_loss * opt.weight_l1

        loss = ssim_loss + l1_loss

        # record losses
        self.avg_meters.update({'ssim': -ssim.item(), 'L1': l1_loss.item()})

        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return cleaned

    def forward(self, x):
        return self.cleaner(x)

    def inference(self, x, image=None):
        pass

    def load(self, ckpt_path):
        pass

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



