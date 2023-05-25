import os
import random
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utils.register import register_dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


@register_dataset
class CatDogDataset(Dataset):
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
        self.data_info = self._get_img_info()

    def __getitem__(self, item):
        image_path, label = self.data_info[item]
        image = Image.open(image_path).convert('RGB')  # 0~255

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

        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        random.seed(self.seed)
        random.shuffle(img_names)
        if self.debug:
            img_names = img_names[:len(img_names) // self.debug_ratio]

        img_labels = [1 if n.startswith('dog') else 0 for n in img_names]

        split_idx = int(len(img_names) * self.split_n)  # 25000 * 0.9 = 22500
        # split_idx = int(100 * self.split_n)
        if self.mode == "train":
            img_names = img_names[:split_idx]
            img_labels = img_labels[:split_idx]
        elif self.mode == "valid":
            img_names = img_names[split_idx:]
            img_labels = img_labels[split_idx:]
        elif self.mode == "test":
            pass
        else:
            raise Exception("self.mode is not recognized, only (train, valid, test) is supported.")

        path_img_set = [os.path.join(self.data_dir, n) for n in img_names]
        data_info = [(n, l) for n, l in zip(path_img_set, img_labels)]

        return data_info
