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

        self.optimizer = get_optimizer(opt, self.cleaner)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

    def update(self, sample):
        y = sample['label'].to(opt.device)

        output = self.forward(sample)

        loss = get_default_loss(output, y, self.avg_meters)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'output': output}

    def forward(self, sample):
        x = sample['input'].to(opt.device)
        return self.cleaner(x)

    def write_train_summary(self, update_return):
        pass

    def step_scheduler(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_lr()[0]

    def load(self, ckpt_path):
        load_dict = {
            'cleaner': self.cleaner,
        }

        if opt.resume:
            load_dict.update({
                'optimizer': self.optimizer,
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
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'epoch': which_epoch
        }

        save_checkpoint(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)



