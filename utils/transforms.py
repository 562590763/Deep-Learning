import os
import sys
from torchvision import transforms
from utils.augment import TenCropLambda
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from data.classification.CatDogDataset.transforms import catdogdataset_transform
from data.classification.CIFAR10.transforms import cifar10dataset_transform
from data.classification.MNIST.transforms import mnistdataset_transform
from data.classification.ImageNet.classes200.transforms import imagenetdataset_transform
from data.segmentation.CarDataset.transforms import cardataset_transform
from data.segmentation.MelanomaDataset.transforms import melanomadataset_transform
from data.segmentation.MembraneDataset.transforms import membranedataset_transform
from data.segmentation.BraTsDataset.transforms import bratsdataset_transform
from data.segmentation.SynapseDataset.transforms import synapsedataset_transform
from data.segmentation.ACDCDataset.transforms import acdcdataset_transform
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(BASE_DIR)


def build_transform(args):
    resize_im = args.img_size > 32
    # this should always dispatch to transforms_imagenet_train
    train_transform_image = create_transform(
        input_size=args.img_size,
        is_training=True,
        use_prefetcher=args.no_prefetcher,
        no_aug=args.no_aug,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_num_splits=args.re_num_splits)
    train_transform_image.transforms.insert(0, transforms.Grayscale(3))
    if not resize_im:
        # replace RandomResizedCropAndInterpolation with RandomCrop
        train_transform_image.transforms[0] = transforms.RandomCrop(
            args.img_size, padding=4)
    train_transform = {'image': train_transform_image, 'label': train_transform_image}

    t = [transforms.Grayscale(3)]
    if resize_im:
        size = int((256 / 224) * args.img_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.img_size))

    if args.task == 'classification':
        t.append(transforms.TenCrop(args.img_size, vertical_flip=False))
        t.append(TenCropLambda())
        tempt = t[:]
    else:
        t.append(transforms.ToTensor())
        tempt = t[:]
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    valid_transform_image = transforms.Compose(t)
    valid_transform_label = transforms.Compose(tempt)
    valid_transform = {'image': valid_transform_image, 'label': valid_transform_label}

    transform_classification = {
        'train': train_transform_image,
        'valid': valid_transform_image,
        'test': valid_transform_image
    }
    transform_segmentation = {
        'train': train_transform,
        'valid': valid_transform,
        'test': valid_transform
    }

    if args.task == 'classification':
        return transform_classification
    else:
        return transform_segmentation
