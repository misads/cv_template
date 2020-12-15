# encoding = utf-8
"""
    Author: xuhaoyu@tju.edu.cn
    Github: https://github.com/misads
    License: MIT

"""
import os
import pdb
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

import dataloader as dl
from network import get_model
from eval import evaluate

from options import opt

from utils import init_log
from mscv.summary import create_summary_writer, write_meters_loss

import misc_utils as utils

# 初始化
with torch.no_grad():
    # 初始化路径
    save_root = os.path.join(opt.checkpoint_dir, opt.tag)
    log_root = os.path.join(opt.log_dir, opt.tag)

    utils.try_make_dir(save_root)
    utils.try_make_dir(log_root)

    # Dataloader
    train_dataloader = dl.train_dataloader
    val_dataloader = dl.val_dataloader

    # 初始化日志
    logger = init_log(training=True)

    # 初始化模型
    Model = get_model(opt.model)
    model = Model(opt)

    # 暂时还不支持多GPU
    # if len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    model = model.to(device=opt.device)

    # 加载预训练模型，恢复中断的训练
    if opt.load:
        load_epoch = model.load(opt.load)
        start_epoch = load_epoch + 1 if opt.resume else 1
    else:
        start_epoch = 1

    # 开始训练
    model.train()

    # 计算开始和总共的step
    print('Start training...')
    start_step = (start_epoch - 1) * len(train_dataloader)
    global_step = start_step
    total_steps = opt.epochs * len(train_dataloader)
    start = time.time()

    # Tensorboard初始化
    writer = create_summary_writer(log_root)

    start_time = time.time()

    # 在日志记录transforms
    logger.info('train_trasforms: ' +str(train_dataloader.dataset.transforms))
    logger.info('===========================================')
    if val_dataloader is not None:
        logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
    logger.info('===========================================')


try:
    # 训练循环
    for epoch in range(start_epoch, opt.epochs + 1):
        for iteration, sample in enumerate(train_dataloader):
            global_step += 1
            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            # 更新模型参数
            update_return = model.update(sample)

            # 获取当前学习率
            lr = model.get_lr()
            lr = lr if lr is not None else opt.lr

            # 显示进度条
            msg = f'lr:{round(lr, 6) : .6f} (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            utils.progress_bar(iteration, len(train_dataloader), 'Epoch:%d' % epoch, msg)

            # 训练时每1000个step记录一下summary
            if global_step % 1000 == 0:
                write_meters_loss(writer, 'train', model.avg_meters, global_step)
                model.write_train_summary(update_return)

        # 每个epoch结束后的显示信息
        logger.info(f'Train epoch: {epoch}, lr: {round(lr, 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            model.save(epoch)

        # 训练中验证
        if epoch % opt.eval_freq == 0:

            model.eval()
            eval_result = evaluate(model, val_dataloader, epoch, writer, logger)
            model.train()

        model.step_scheduler()

except Exception as e:

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的错误信息
