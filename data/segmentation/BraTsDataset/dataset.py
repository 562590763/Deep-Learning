import os
import random
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from utils.register import register_dataset


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


@register_dataset
class BraTsDataset(Dataset):
    def __init__(self, data_dir, list_dir="", mode="train", split_n=0.9,
                 image_size=224, self_transform=True, norm_transform=False,
                 transform=None, debug=True, debug_ratio=1, seed=42):
        self.data_dir = data_dir
        self.mode = mode
        self.list_dir = os.path.join(list_dir, '{}.txt'.format("test" if mode == 'test' else 'train'))
        self.split_n = split_n
        self.image_size = image_size
        self.self_transform = self_transform
        self.norm_transform = norm_transform
        self.transform = transform
        self.debug = debug
        self.debug_ratio = debug_ratio
        self.seed = seed
        self._get_img_info()

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception(
                "\ndata_dir:{} is a empty dir! Please chekout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def __getitem__(self, item):

        path = self.data_info[item]
        if self.transform is not None:
            if self.self_transform:
                if self.mode == 'train':
                    image, label = pkload(path + 'data_f32b0.pkl')
                    sample = {'image': image, 'label': label}
                    sample = self.transform(sample)
                    return sample['image'], sample['label']
                elif self.mode == 'valid':
                    image, label = pkload(path + 'data_f32b0.pkl')
                    sample = {'image': image, 'label': label}
                    sample = self.transform(sample)
                    return sample['image'], sample['label']
                else:
                    image = pkload(path + 'data_f32b0.pkl')
                    image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
                    image = np.ascontiguousarray(image.transpose((3, 0, 1, 2)))
                    image = torch.from_numpy(image).float()
                    return image, None
            else:
                raise Exception("This dataset does not support automatic data augmentation.")
        else:
            raise Exception("This dataset can't be without transform method.")

    @staticmethod
    def collate(batch):
        return [torch.cat(v) for v in zip(*batch)]

    def _get_img_info(self):

        self.data_info = []
        with open(self.list_dir) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                path = os.path.join(self.data_dir, line, name + '_')
                self.data_info.append(path)

        random.seed(self.seed)
        random.shuffle(self.data_info)
        if self.debug:
            self.data_info = self.data_info[:len(self.data_info) // self.debug_ratio]
        split_idx = int(len(self.data_info) * self.split_n)
        if self.mode == "train":
            self.data_info = self.data_info[:split_idx]
        elif self.mode == "valid":
            self.data_info = self.data_info[split_idx:]
        elif self.mode == "test":
            pass
        else:
            raise Exception("self.mode is not recognized, only (train, valid, test) is supported.")
