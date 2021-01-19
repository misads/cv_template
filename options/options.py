import argparse
import sys
import os

import torch

import misc_utils as utils

"""
    Arg parse
    opt = parse_args()
"""


def parse_args():
    # experiment specifics
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str, default='cache',
                        help='folder name to save the outputs')
    parser.add_argument('--gpu_ids', '--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    # dirs (NOT often Changed)
    parser.add_argument('--data_root', type=str, default='./datasets/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')
    #######################

    parser.add_argument('--model', type=str, default=None, help='which model to use')
    parser.add_argument('--norm', type=str, choices=['batch', 'instance', None], default=None,
                        help='[instance] normalization or [batch] normalization')

    # batch size
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', '-w', type=int, default=4, help='dataloader workers')

    # optimizer and scheduler
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'radam', 'lookahead', 'ranger'], default='adam')
    parser.add_argument('--scheduler', choices=['cos', 'step', 'exp', 'cyclic', 'lambda', 'None'], default='cos')

    # data augmentation
    parser.add_argument('--aug', action='store_true', help='Randomly scale, jitter, change hue, saturation and brightness')
    parser.add_argument('--norm-input', action='store_true')
    parser.add_argument('--random-erase', action='store_true', help='debug mode')

    # scale
    parser.add_argument('--scale', type=int, default=None, help='scale images to this size')
    parser.add_argument('--crop', type=int, default=256, help='then crop to this size')

    # for datasets
    parser.add_argument('--dataset', default='', help='training dataset')
    parser.add_argument('--transform', default='crop256', help='transform')
    parser.add_argument('--val_set', type=str, default=None)
    parser.add_argument('--test_set', type=str, default=None)

    # init weights
    parser.add_argument('--init', type=str, default=None, help='{normal, xavier, kaiming, orthogonal}')

    # loss weight
    parser.add_argument('--weight_ssim', type=float, default=1.1)  # SSIM loss
    parser.add_argument('--weight_l1', type=float, default=0.75)  # l1 loss
    parser.add_argument('--weight_vgg', type=float, default=0.)  # content loss(vgg loss)
    parser.add_argument('--weight_grad', type=float, default=0.)  # gradient loss

    # training options
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--load', type=str, default=None, help='load checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training, only used when --load')
    parser.add_argument('--reset', action='store_true', help='reset training, only used when --load')

    parser.add_argument('--epochs', '--max_epoch', type=int, default=50, help='epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

    parser.add_argument('--save_freq', type=int, default=10, help='freq to save models')
    parser.add_argument('--eval_freq', '--val_freq', type=int, default=10, help='freq to eval models')
    parser.add_argument('--log_freq', type=int, default=1, help='freq to vis in tensorboard')

    # test options
    parser.add_argument('--tta', action='store_true', help='test with augmentation')
    parser.add_argument('--tta-x8', action='store_true', help='test with augmentation x8')

    return parser.parse_args()


opt = parse_args()

opt.device = 'cuda:' + opt.gpu_ids if torch.cuda.is_available() and opt.gpu_ids != '-1' else 'cpu'

if opt.debug:
    opt.save_freq = 1
    opt.eval_freq = 1
    opt.log_freq = 1


def get_command_run():
    args = sys.argv.copy()
    args[0] = args[0].split('/')[-1]

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
        command = f'CUDA_VISIBLE_DEVICES={gpu_id} '
    else:
        command = ''

    if sys.version[0] == '3':
        command += 'python3'
    else:
        command += 'python'

    for i in args:
        command += ' ' + i
    return command


if opt.tag != 'cache':
    pid = f'[PID:{os.getpid()}]'
    with open('run_log.txt', 'a') as f:
        f.writelines(utils.get_time_str(fmt="%Y-%m-%d %H:%M:%S") + ' ' + pid + ' ' + get_command_run() + '\n')

# utils.print_args(opt)
