# encoding=utf-8
"""
TensorBoard Summary
"""
from datetime import datetime
import math
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from .meters import Meters

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


