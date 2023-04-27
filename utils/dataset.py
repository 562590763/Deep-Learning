import torchvision.datasets as datasets
from utils.register import get_dataset, get_dataset_type
from data.classification.CatDogDataset import dataset
from data.classification.ImageNet.classes200 import dataset
from data.segmentation.CarDataset import dataset
from data.segmentation.MelanomaDataset import dataset
from data.segmentation.MembraneDataset import dataset
from data.segmentation.BraTsDataset import dataset
from data.segmentation.SynapseDataset import dataset
from data.segmentation.ACDCDataset import dataset


def getdata(dataset, data_dir, list_dir="", debug=True, debug_ratio=1, mode="train", split_n=0.9,
            image_size=224, self_transform=True, norm_transform=False, transform=None, seed=42):
    dataset_type = get_dataset_type()
    if dataset == 'CIFAR10':
        path = "./data/classification/CIFAR10/{}".format('train' if mode == 'train' else 'test')
        return datasets.CIFAR10(path, train=mode == 'train',
                                transform=transform, download=False)
    if dataset == 'MNIST':
        path = "./data/classification/MNIST/{}".format('train' if mode == 'train' else 'test')
        return datasets.MNIST(path, train=mode == 'train',
                              transform=transform, download=False)
    elif dataset in dataset_type:
        return get_dataset(dataset)(data_dir=data_dir, list_dir=list_dir, mode=mode, split_n=split_n,
                                    image_size=image_size, self_transform=self_transform,
                                    norm_transform=norm_transform, transform=transform,
                                    debug=debug, debug_ratio=debug_ratio, seed=seed)
    else:
        raise Exception("未导入{}数据集".format(dataset))
