import albumentations as A
from dataloader.transforms import custom_transform as C
from albumentations.pytorch.transforms import ToTensorV2

class No_Transform(object):
    width = height = 1000

    train_transform = A.Compose(  # FRCNN
        [
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        ),
        additional_targets = {'gt': 'image'}
    )

    val_transform = train_transform

