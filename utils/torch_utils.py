# encoding=utf-8
"""Misc PyTorch utils

Usage:
    >>> from torch_template import torch_utils
    >>> torch_utils.func_name()  # to call functions in this file

"""
from datetime import datetime
import math
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


##############################
#    Functional utils
##############################
from misc_utils import format_num


def tensor2im(x: torch.Tensor, norm=False, dtype='float32'):
    """Convert tensor to image.

    Args:
        x(torch.Tensor): input tensor, [n, c, h, w] float32 type.
        norm(bool): if the tensor should be denormed first
        dtype(str): not used yet.

    Returns:
        an image in shape of [h, w, c].

    """
    if norm:
        x = (x + 1) / 2
    x[x > 1] = 1
    x[x < 0] = 0
    return x.detach().cpu().data[0]


##############################
#    Network utils
##############################
def print_network(net: nn.Module, print_size=False):
    """Print network structure and number of parameters.

    Args:
        net(nn.Module): network model.
        print_size(bool): print parameter num of each layer.

    Example
        >>> import torchvision as tv
        >>> from torch_template import torch_utils
        >>>
        >>> vgg16 = tv.models.vgg16()
        >>> torch_utils.print_network(vgg16)
        >>> '''
        >>> features.0.weight [3, 64, 3, 3]
        >>> features.2.weight [64, 64, 3, 3]
        >>> features.5.weight [64, 128, 3, 3]
        >>> features.7.weight [128, 128, 3, 3]
        >>> features.10.weight [128, 256, 3, 3]
        >>> features.12.weight [256, 256, 3, 3]
        >>> features.14.weight [256, 256, 3, 3]
        >>> features.17.weight [256, 512, 3, 3]
        >>> features.19.weight [512, 512, 3, 3]
        >>> features.21.weight [512, 512, 3, 3]
        >>> features.24.weight [512, 512, 3, 3]
        >>> features.26.weight [512, 512, 3, 3]
        >>> features.28.weight [512, 512, 3, 3]
        >>> classifier.0.weight [25088, 4096]
        >>> classifier.3.weight [4096, 4096]
        >>> classifier.6.weight [4096, 1000]
        >>> Total number of parameters: 138,357,544
        >>> '''
    """
    num_params = 0
    print(net)
    for name, param in net.named_parameters():
        num_params += param.numel()
        size = list(param.size())
        if len(size) > 1:
            if print_size:
                print(name, size[1:2]+size[:1]+size[2:], format_num(param.numel()))
            else:
                print(name, size[1:2] + size[:1] + size[2:])
    print('Total number of parameters: %s' % format_num(num_params))


##############################
#            Meters
##############################


class Meters(object):
    def __init__(self):
        pass

    def update(self, new_dic):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def items(self):
        return self.dic.items()


class AverageMeters(Meters):
    """AverageMeter class

    Example
        >>> avg_meters = AverageMeters()
        >>> for i in range(100):
        >>>     avg_meters.update({'f': i})
        >>>     print(str(avg_meters))

    """

    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


class ExponentialMovingAverage(Meters):
    """EMA class

    Example
        >>> ema_meters = ExponentialMovingAverage(0.98)
        >>> for i in range(100):
        >>>     ema_meters.update({'f': i})
        >>>     print(str(ema_meters))

    """

    def __init__(self, decay=0.9, dic=None, total_num=None):
        self.decay = decay
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        decay = self.decay
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = (1 - decay) * new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] = decay * self.dic[key] + (1 - decay) * new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key]  # / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


"""
    TensorBoard
    Example:
        writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
        write_meters_loss(writer, 'train', avg_meters, iteration)
        write_loss(writer, 'train', 'F1', 0.78, iteration)
        write_image(writer, 'train', 'input', img, iteration)
        # shell
        tensorboard --logdir {base_path}/logs

"""


def create_summary_writer(log_dir):
    """Create a tensorboard summary writer.

    Args:
        log_dir: log directory.

    Returns:
        (SummaryWriter): a summary writer.

    Example
        >>> writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
        >>> write_meters_loss(writer, 'train', avg_meters, iteration)
        >>> write_loss(writer, 'train', 'F1', 0.78, iteration)
        >>> write_image(writer, 'train', 'input', img, iteration)
        >>> # shell
        >>> tensorboard --logdir {base_path}/logs

    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%m-%d_%H-%M-%S'))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir, max_queue=3, flush_secs=10)

    return writer


def write_loss(writer: SummaryWriter, prefix, loss_name: str, value: float, iteration):
    """Write loss into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        prefix(str): any string, e.g. 'train'.
        loss_name(str): loss name.
        value(float): loss value.
        iteration(int): epochs or iterations.

    Example
        >>> write_loss(writer, 'train', 'F1', 0.78, iteration)

    """
    writer.add_scalar(
        os.path.join(prefix, loss_name), value, iteration)


def write_graph(writer: SummaryWriter, model, inputs_to_model=None):
    """Write net graph into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        model(nn.Module): model.
        inputs_to_model(tuple or list): forward inputs.

    Example
        >>> from tensorboardX import SummaryWriter
        >>> input_data = Variable(torch.rand(16, 3, 224, 224))
        >>> vgg16 = torchvision.models.vgg16()
        >>>
        >>> writer = SummaryWriter(log_dir='logs')
        >>> write_graph(vgg16, (input_data,))

    """
    with writer:
        writer.add_graph(model, inputs_to_model)


def write_image(writer: SummaryWriter, prefix, image_name: str, img, iteration, dataformats='CHW'):
    """Write images into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        prefix(str): any string, e.g. 'train'.
        image_name(str): image name.
        img: image tensor in [C, H, W] shape.
        iteration(int): epochs or iterations.
        dataformats(str): 'CHW' or 'HWC' or 'NCHW'.

    Example
        >>> write_image(writer, 'train', 'input', img, iteration)

    """
    writer.add_image(
        os.path.join(prefix, image_name), img, iteration, dataformats=dataformats)


def write_meters_loss(writer: SummaryWriter, prefix, avg_meters: Meters, iteration):
    """Write all losses in a meter class into writer.

    Args:
        writer(SummaryWriter): writer created by create_summary_writer()
        prefix(str): any string, e.g. 'train'.
        avg_meters(AverageMeters or ExponentialMovingAverage): meters.
        iteration(int): epochs or iterations.

    Example
        >>> writer = create_summary_writer(os.path.join(self.basedir, 'logs'))
        >>> ema_meters = ExponentialMovingAverage(0.98)
        >>> for i in range(100):
        >>>     ema_meters.update({'f1': i, 'f2': i*0.5})
        >>>     write_meters_loss(writer, 'train', ema_meters, i)

    """
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)


