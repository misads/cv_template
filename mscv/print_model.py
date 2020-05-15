import torch.nn as nn
from misc_utils import format_num


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
