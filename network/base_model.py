import os
import torch.nn as nn

from abc import abstractmethod

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, ckpt_path):
        pass

    @abstractmethod
    def save(self, which_epoch):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def write_train_summary(self, update_return):
        pass

    def step_scheduler(self):
        pass

    def get_lr(self):
        pass
