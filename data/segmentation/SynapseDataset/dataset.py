import os
import random
import h5py
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.register import register_dataset
from scipy.ndimage.interpolation import zoom
from utils.augment import augment, random_transform2


@register_dataset
class SynapseDataset(Dataset):
    def __init__(self, data_dir, list_dir="", mode="train", split_n=0.9,
                 image_size=224, self_transform=True, norm_transform=False,
                 transform=None, debug=True, debug_ratio=1, seed=42):
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.mode = mode
        self.split_n = split_n
        self.image_size = image_size
        self.self_transform = self_transform
        self.norm_transform = norm_transform
        self.path = os.path.join(list_dir, '{}.txt'.format("test" if self.mode == 'test' else 'train'))
        self.sample_list = open(self.path).readlines()
        self.transform = transform
        self.debug = debug
        self.debug_ratio = debug_ratio
        self.seed = seed
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()
        self._get_img_info()

    def __len__(self):
        if len(self.sample_list) == 0:
            raise Exception(
                "\ndata_dir:{} is a empty dir! Please chekout your path to images!".format(self.data_dir))
        return len(self.sample_list)

    def __getitem__(self, item):
        if self.mode != "test":
            slice_name = self.sample_list[item].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[item].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        if self.mode == 'test':
            # image_list = []
            # label_list = []
            # for ind in range(image.shape[0]):
            #     slice_image = image[ind, :, :]
            #     slice_label = label[ind, :, :]
            #     image_list.append(self.transform(slice_image).unsqueeze(0))
            #     label_list.append(self.transform(slice_label))
            # image = torch.cat(image_list, dim=0)  # N, C, H, W
            # label = torch.cat(label_list, dim=0)  # N, H, W
            pass
        elif self.transform is not None:
            if self.self_transform:
                if self.mode == 'train':
                    sample = {'image': image, 'label': label}
                    sample = self.transform(sample)
                    image, label = sample['image'], sample['label']
                elif self.mode == 'valid':
                    image = self.transform(image)  # C, H, W
                    label = self.transform(label)
                    label = label.squeeze(0)  # H, W
            else:
                image, label = augment(random_transform2(), image, label)
                x, y = image.shape
                if x != self.image_size or y != self.image_size:
                    image = zoom(image, (self.image_size / x, self.image_size / y), order=3)
                    label = zoom(label, (self.image_size / x, self.image_size / y), order=0)
                image, label = self.totensor(image), self.totensor(label).squeeze(0)
        else:
            trans = transforms.Compose([self.topil,
                                        transforms.Resize((self.image_size, self.image_size)), self.totensor])
            image, label = trans(image).unsqueeze(0), trans(label)

        if self.mode != 'test' and self.norm_transform:
            norm_transforms = transforms.Normalize(0.5, 0.5)
            image = norm_transforms(image)
        return image, label, self.sample_list[item].strip('\n')

    def _get_img_info(self):

        random.seed(self.seed)
        random.shuffle(self.sample_list)
        if self.debug:
            self.sample_list = self.sample_list[:len(self.sample_list) // self.debug_ratio]
        split_idx = int(len(self.sample_list) * self.split_n)
        if self.mode == "train":
            self.sample_list = self.sample_list[:split_idx]
        elif self.mode == "valid":
            self.sample_list = self.sample_list[split_idx:]
        elif self.mode == "test":
            pass
        else:
            raise Exception("self.mode is not recognized, only (train, valid, test) is supported.")
