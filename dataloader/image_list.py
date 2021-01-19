# encoding=utf-8
import pdb
import os
import torch.utils.data.dataset as dataset
import misc_utils as utils
import random
import numpy as np
import cv2

from dataloader.transforms.custom_transform import read_image


class ListTrainValDataset(dataset.Dataset):
    """ImageDataset for training.

    Args:
        file_list(str): dataset list, input and label should be split by ','
        aug(bool): data argument (×8)
        norm(bool): normalization

    Example:
        train_dataset = ImageDataset('train.txt', aug=False)
        for i, data in enumerate(train_dataset):
            input, label = data['input']. data['label']

    """

    def __init__(self, file_list, transforms, max_size=None):
        self.im_names = []
        self.labels = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                img, label = line.split()
                img, label = img.strip(), label.strip()
                self.im_names.append(img)
                self.labels.append(label)

        self.transforms = transforms
        self.max_size = max_size

    def __getitem__(self, index):
        """Get indexs by index

        Args:
            index(int): index

        Returns:
            {
                'input': input,
                'label': label,
                'path': path,
            }

        """

        input = read_image(self.im_names[index])
        gt = read_image(self.labels[index]) 

        sample = self.transforms(**{
            'image': input,
            'gt': gt,
        })

        sample = {
            'input': sample['image'],
            'label': sample['gt'],
            'path': self.im_names[index],
        }

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)


class ListTestDataset(dataset.Dataset):
    """ImageDataset for test.

    Args:
        file_list(str): dataset path'
        norm(bool): normalization

    Example:
        test_dataset = ImageDataset('test', crop=256)
        for i, data in enumerate(test_dataset):
            input, file_name = data

    """
    def __init__(self, file_list, transforms, max_size=None):
        self.im_names = []
        with open(file_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n')
                img = line
                self.im_names.append(img)

        self.transforms = transforms
        self.max_size = max_size

    def __getitem__(self, index):

        input = read_image(self.im_names[index])

        sample = self.transforms(**{
            'image': input,
            'gt': input,
        })

        sample = {
            'input': sample['image'],
            'path': self.im_names[index],
        }

        return sample

    def __len__(self):
        if self.max_size is not None:
            return min(self.max_size, len(self.im_names))

        return len(self.im_names)

