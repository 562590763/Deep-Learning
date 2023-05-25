import os
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utils.register import register_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


@register_dataset
class ImageNet200(Dataset):
    def __init__(self, data_dir, list_dir="", mode="train", split_n=0.9,
                 image_size=224, self_transform=True, norm_transform=False,
                 transform=None, debug=True, debug_ratio=1, seed=42):
        self.data_dir = data_dir
        self.list_dir = list_dir
        self.mode = mode
        self.image_size = image_size
        self.label_map = {}
        self.split_n = split_n
        self.self_transform = self_transform
        self.norm_transform = norm_transform
        self.transform = transform
        self.debug = debug
        self.debug_ratio = debug_ratio
        self.seed = seed
        self.data_info = self._get_img_info()

    def __getitem__(self, item):
        image_path, label = self.data_info[item]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        else:
            trans = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(),
                                        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
            image = trans(image)

        return image, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please chekout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):

        label_dirs = os.listdir(self.data_dir)
        for label in range(len(label_dirs)):
            label_dir = label_dirs[label]
            self.label_map[label_dir] = label
        if self.mode == "train":
            img_names = []
            img_labels = []
            for label in range(len(label_dirs)):
                label_dir = label_dirs[label]
                img_names_local = os.listdir(self.data_dir + '/' + label_dir + '/images')
                img_names_local = list(filter(lambda x: x.endswith('.JPEG'), img_names_local))
                img_names_local = [self.data_dir + '/' + label_dir + '/images/' + x for x in img_names_local]
                img_labels_local = [label for _ in range(len(img_names_local))]
                img_names.extend(img_names_local)
                img_labels.extend(img_labels_local)
        elif self.mode == "valid":
            img_names = os.listdir(self.data_dir[:-5] + 'valid/images')
            img_names = list(filter(lambda x: x.endswith('.JPEG'), img_names))
            img_names = [self.data_dir[:-5] + 'valid/images/' + x for x in img_names]
            label_path = self.data_dir[:-5] + 'valid/val_annotations.txt'
            with open(label_path) as f:
                data = f.readlines()
                img_labels = [self.label_map[line.split()[1]] for line in data]
        elif self.mode == "test":
            img_names = os.listdir(self.data_dir + '/images')
            img_names = list(filter(lambda x: x.endswith('.JPEG'), img_names))
            img_names = [self.data_dir + '/images/' + x for x in img_names]
            img_labels = [0 for _ in range(len(img_names))]
        else:
            raise Exception("self.mode is not recognized, only (train, valid, test) is supported.")

        data_info = [(n, l) for n, l in zip(img_names, img_labels)]
        random.seed(self.seed)
        random.shuffle(data_info)
        if self.debug:
            data_info = data_info[:len(data_info) // self.debug_ratio]

        return data_info
