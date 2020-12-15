import albumentations as A
from dataloader.transforms import custom_transform as C
from albumentations.pytorch.transforms import ToTensorV2

class Padding8(object):
    width = height = 256

    divisor = 8 
    train_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),  # TTA×8
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        additional_targets = {'gt': 'image'}
    )

    divisor = 8  # padding成8的倍数
    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        additional_targets = {'gt': 'image'}
    )
