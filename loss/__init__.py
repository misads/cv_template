import torch.nn as nn
from loss.gradient import grad_loss
from loss.vggloss import vgg_loss
from loss.ssim import ssim_loss as criterionSSIM

criterionCAE = nn.L1Loss()
criterionL1 = criterionCAE
criterionBCE = nn.BCELoss()
criterionMSE = nn.MSELoss()
