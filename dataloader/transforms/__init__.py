from .no_transform import No_Transform
from .crop256 import Crop256
from .padding8 import Padding8


transforms = {
    'crop256': Crop256,
    'padding8': Padding8,
    'none': No_Transform,
}


def get_transform(transform: str):
    if transform in transforms:
        return transforms[transform]
    else:
        raise Exception('No such transform: "%s", available: {%s}.' % (transform, '|'.join(transforms.keys())))

