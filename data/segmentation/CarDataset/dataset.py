import os
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utils.register import register_dataset


@register_dataset
class CarDataset(Dataset):
    def __init__(self, data_dir, list_dir="", mode="train", split_n=0.9,
                 image_size=224, self_transform=True, norm_transform=False,
                 transform=None, debug=True, debug_ratio=1, seed=42):
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.mode = mode
        self.image_size = image_size
        self.split_n = split_n
        self.self_transform = self_transform
        self.norm_transform = norm_transform
        self.transform = transform
        self.debug = debug
        self.debug_ratio = debug_ratio
        self.seed = seed
        self.topil = transforms.ToPILImage()
        self.totensor = transforms.ToTensor()
        self.trans = transforms.Compose([self.topil,
                                         transforms.Resize((self.image_size, self.image_size)), self.totensor])
        self.data_info = self._get_img_info()

    def __getitem__(self, item):
        image_name = self.data_info[item]
        image_path = self.data_dir + '/img/' + image_name + '.jpg'
        label_path = self.data_dir + '/mask/' + image_name + '_mask.png'

        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path)).astype(np.float32)

        if self.transform is not None:
            if self.self_transform:
                if self.mode == 'train':
                    sample = {'image': image, 'label': label}
                    sample = self.transform(sample)
                    image, label = sample['image'], sample['label']
                else:
                    image = self.transform(image)
                    label = self.transform(label)
                    label = label.squeeze(0)
            else:
                raise Exception("This dataset does not support automatic data augmentation.")
        else:
            image, label = self.trans(image), self.trans(label).squeeze(0)

        return image, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please chekout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):

        img_names = os.listdir(self.data_dir + '/img')
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
        img_names = [x[:-4] for x in img_names]
        random.seed(self.seed)
        random.shuffle(img_names)
        if self.debug:
            img_names = img_names[:len(img_names) // self.debug_ratio]
        split_idx = int(len(img_names) * self.split_n)
        if self.mode == "train":
            img_names = img_names[:split_idx]
        elif self.mode == "valid":
            img_names = img_names[split_idx:]
        elif self.mode == "test":
            pass
        else:
            raise Exception("self.mode is not recognized, only (train, valid, test) is supported.")

        return img_names
