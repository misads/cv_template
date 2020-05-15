# encoding=utf-8

from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim  # deprecated
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ski_ssim

import pdb

import dataloader as dl
from options import opt
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im

import torch
import numpy as np
from torch.autograd import Variable

from PIL import Image
from utils import *

import misc_utils as utils


def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):

    save_root = os.path.join(opt.result_dir, opt.tag, str(epoch), data_name)

    utils.try_make_dir(save_root)

    total_psnr = 0.0
    total_ssim = 0.0
    ct_num = 0
    # print('Start testing ' + tag + '...')
    for i, data in enumerate(dataloader):
        if data_name == 'val':
            input, label, path = data['input'], data['label'], data['path']
            utils.progress_bar(i, len(dataloader), 'Eva... ')

            ct_num += 1

            with torch.no_grad():
                img_var = Variable(input, requires_grad=False).to(device=opt.device)

                predicted = model(img_var)
                if isinstance(predicted, tuple) or isinstance(predicted, list):
                    predicted = predicted[0]

                label = tensor2im(label)
                predicted = tensor2im(predicted)

                total_psnr += psnr(predicted, label, data_range=255)
                total_ssim += ski_ssim(predicted, label, data_range=255, multichannel=True)

                save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
                Image.fromarray(predicted).save(save_dst)

        elif data_name == 'test':
            input, path = data['input'], data['path']
            utils.progress_bar(i, len(dataloader), 'Eva... ')
            # ct_num += 1
            with torch.no_grad():
                img_var = Variable(input, requires_grad=False).to(device=opt.device)

                predicted = model(img_var)
                if isinstance(predicted, tuple) or isinstance(predicted, list):
                    predicted = predicted[0]

                label = tensor2im(label)
                predicted = tensor2im(predicted)

                total_psnr += psnr(predicted, label, data_range=255)
                total_ssim += ski_ssim(predicted, label, data_range=255, multichannel=True)

                save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
                Image.fromarray(predicted).save(save_dst)

        else:
            raise Exception('Unknown dataset name: %s.' % data_name)

    if data_name == 'val':
        ave_psnr = total_psnr / float(ct_num)
        ave_ssim = total_ssim / float(ct_num)
        # write_loss(writer, f'val/{data_name}', 'psnr', total_psnr / float(ct_num), epochs)

        logger.info(f'Eva({data_name}) epoch {epoch} , psnr: {ave_psnr : .6f}.')
        logger.info(f'Eva({data_name}) epoch {epoch} , ssim: {ave_ssim : .6f}.')
        
        return f'{ave_ssim: .3f}'
    else:
        return ''


if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)

    opt.which_epoch = model.load(opt.load)

    model.eval()

    log_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
    utils.try_make_dir(log_root)

    writer = create_summary_writer(log_root)

    logger = init_log(training=False)
    evaluate(model, dl.val_dataloader, opt.which_epoch, writer, logger, 'val')

