# encoding=utf-8
from dataloader.image_list import ListTrainValDataset, ListTestDataset
from dataloader.transforms import get_transform
from torch.utils.data import DataLoader
from options import opt
import pdb
import os
###################

TEST_DATASET_HAS_OPEN = False  # 有没有开放测试集

###################

train_list = os.path.join('datasets', opt.dataset, 'train.txt')
val_list = os.path.join('datasets', opt.dataset, 'val.txt')

max_size = 128 if opt.debug else None

# transforms
transform = get_transform(opt.transform)
train_transform = transform.train_transform
val_transform = transform.val_transform

# datasets和dataloaders
train_dataset = ListTrainValDataset(train_list, transforms=train_transform, max_size=max_size)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)

val_dataset = ListTrainValDataset(val_list, transforms=val_transform, max_size=max_size)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=opt.workers//2)

if TEST_DATASET_HAS_OPEN:
    test_list = os.path.join('datasets', opt.dataset, 'test.txt') # 还没有

    test_dataset = ListTestDataset(test_list, scale=opt.scale, max_size=max_size, norm=opt.norm_input)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

else:
    test_dataloader = None
