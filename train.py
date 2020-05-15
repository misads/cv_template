# encoding = utf-8
"""
    一个图像复原或分割的Baseline。

    如何添加新的模型：

    ① 复制network目录下的Default文件夹，改成另外一个名字(比如MyNet)。

    ② 在network/__init__.py中import你的Model并且在models = {}中添加它。
        from MyNet.Model import Model as MyNet
        models = {
            'default': Default,
            'MyNet': MyNet,
        }

    ③ 尝试 python train.py --model MyNet 看能否成功运行


    File Structure:
    cv_template
        ├── train.py                :Train and evaluation loop, errors and outputs visualization (Powered by TensorBoard)
        ├── eval.py                 :Evaluation and test (with visualization)
        ├── test.py                 :Test
        │
        ├── clear.py                :Clear cache, be CAREFUL to use it
        │
        ├── run_log.txt             :Record your command logs (except --tag cache)
        │
        ├── network
        │     ├── __init__.py       :Declare all models here so that `--model` can work properly
        │     ├── Default
        │     │      ├── Model.py   :Define default model, losses and parameter updating procedure
        │     │      └── FFA.py
        │     └── MyNet
        │            ├── Model.py   :Define your model, losses and parameter updating procedure
        │            └── mynet.py
        ├── options
        │     └── options.py        :Define options
        │
        │
        ├── dataloader/             :Define Dataloaders
        │     ├── __init__.py       :imports all dataloaders in dataloaders.py
        │     ├── dataloaders.py    :Define all dataloaders here
        │     └── my_dataset.py     :Custom Dataset
        │
        ├── checkpoints/<tag>       :Trained checkpoints
        ├── logs/<tag>              :Logs and TensorBoard event files
        └── results/<tag>           :Test results


    Datasets:

        datasets
           ├── train
           │     ├── 00001
           │     ├── 00002
           │     └── .....
           ├──  val
           │     ├── 00001
           │     ├── 00002
           │     └── .....
           ├── train.txt
           └── val.txt

    Usage:

    #### Train

        python train.py --tag train_1 --epochs 500 -b 8 --gpu 1

    #### Resume Training

        python train.py --load checkpoints/train_1/500_checkpoint.pt --resume

    #### Evaluation

        python eval.py --tag eval_1 --model MyNet --load checkpoints/train_1/500_checkpoint.pt

    #### Test

        python test.py --tag test_1

    #### Clear

        python clear.py [--tag cache]  # (DO NOT use this command unless you know what you are doing.)


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
# from utils.send_sms import send_notification

import misc_utils as utils

######################
#       Paths
######################
save_root = os.path.join(opt.checkpoint_dir, opt.tag)
log_root = os.path.join(opt.log_dir, opt.tag)

utils.try_make_dir(save_root)
utils.try_make_dir(log_root)


######################
#      DataLoaders
######################
train_dataloader = dl.train_dataloader
val_dataloader = dl.val_dataloader
# init log
logger = init_log(training=True)

######################
#     Init model
######################
Model = get_model(opt.model)
model = Model(opt)

# 暂时还不支持多GPU
# if len(opt.gpu_ids):
#     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model = model.to(device=opt.device)

if opt.load:
    start_epoch = model.load(opt.load) + 1
else:
    start_epoch = 1

model.train()

# Start training
print('Start training...')
start_step = (start_epoch - 1) * len(train_dataloader)
global_step = start_step
total_steps = opt.epochs * len(train_dataloader)
start = time.time()

#####################
#   定义scheduler
#####################

scheduler = model.scheduler

######################
#    Summary_writer
######################
writer = create_summary_writer(log_root)

start_time = time.time()
######################
#     Train loop
######################
try:
    eval_result = ''

    for epoch in range(start_epoch, opt.epochs + 1):
        for iteration, data in enumerate(train_dataloader):
            global_step += 1
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            img, label = data['input'], data['label']  # ['label'], data['image']  #

            img_var = Variable(img, requires_grad=False).to(device=opt.device)
            label_var = Variable(label, requires_grad=False).to(device=opt.device)

            ##############################
            #       Update parameters
            ##############################
            update = model.update(img_var, label_var)
            restored = update.get('restored')

            pre_msg = 'Epoch:%d' % epoch

            msg = f'lr:{round(scheduler.get_lr()[0], 6) : .6f} (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            utils.progress_bar(iteration, len(train_dataloader), pre_msg, msg)
            # print(pre_msg, msg)

            if global_step % 1000 == 0:
                write_meters_loss(writer, 'train', model.avg_meters, global_step)

        logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            model.save(epoch)

        ####################
        #     Validation
        ####################
        if epoch % opt.eval_freq == 0:

            model.eval()
            # eval_result = evaluate(model, val_dataloader, epoch, writer, logger)
            model.train()

        if scheduler is not None:
            scheduler.step()

    # send_notification([opt.tag[:12], '', '', eval_result])

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Accuracy:' + eval_result + '\n')

except Exception as e:

    # if not opt.debug:  # debug模式不会发短信 12是短信模板字数限制
    #     send_notification([opt.tag[:12], str(e)[:12]], template='error')

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    # print(e)
    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的trace back信息
