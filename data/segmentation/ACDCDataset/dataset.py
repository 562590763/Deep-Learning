import os
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.register import register_dataset
from scipy.ndimage.interpolation import zoom
from utils.augment import augment, random_transform2


@register_dataset
class ACDCDataset(Dataset):
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
        self.transform = transform
        self.debug = debug
        self.debug_ratio = debug_ratio
        self.seed = seed
        self.topil = transforms.ToPILImage()
        self.togray = transforms.Grayscale()
        self.totensor = transforms.ToTensor()
        self._get_img_info()

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception(
                "\ndata_dir:{} is a empty dir! Please chekout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def __getitem__(self, item):

        data_name = self.data_info[item]
        data_path = self.data_dir + "/" + data_name
        data = np.load(data_path, allow_pickle=True)
        image, label = data[0], data[1]

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
                # image, label = self.topil(image), self.topil(image)
                # image = self.transform['image'](image)
                # label = self.transform['label'](label)
                # label = self.togray(label).squeeze(0)
                # raise Exception("This dataset does not support automatic data augmentation.")
                image, label = augment(random_transform2(), image, label)
                x, y = image.shape
                if x != self.image_size or y != self.image_size:
                    image = zoom(image, (self.image_size / x, self.image_size / y), order=3)
                    label = zoom(label, (self.image_size / x, self.image_size / y), order=0)
                image, label = self.totensor(image), self.totensor(label).squeeze(0)
        else:
            trans = transforms.Compose([self.topil,
                                        transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
            image, label = trans(image).unsqueeze(0), trans(label)

        return image, label  # C H W, H W

    def _get_img_info(self):

        if self.mode == 'test':
            self.data_dir = self.data_dir[:-4] + 'train'
        self.data_info = os.listdir(self.data_dir)
        self.data_info = list(filter(lambda x: x.endswith('.npy'), self.data_info))
        random.seed(self.seed)
        random.shuffle(self.data_info)
        if self.debug:
            self.data_info = self.data_info[:len(self.data_info) // self.debug_ratio]
        split_num = len(self.data_info)
        split_left = int(split_num * 0.9 * self.split_n)
        split_right = int(split_num * 0.9)
        if self.mode == "train":
            # 数据量:1540
            self.data_info = self.data_info[:split_left]
        elif self.mode == "valid":
            # 数据量:171
            self.data_info = self.data_info[split_left:split_right]
        elif self.mode == "test":
            # 数据量:
            self.data_info = self.data_info[split_right:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid, test)")
