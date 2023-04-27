import torch
import random
import numpy as np
import albumentations
import imgaug as ia
import imgaug.augmenters as iaa
from scipy import ndimage
from torchvision import transforms
from scipy.ndimage.interpolation import zoom
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def normalizes():
    # norm_mean = [0.485, 0.456, 0.406]
    # norm_std = [0.229, 0.224, 0.225]
    return transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)


class TenCropLambda(object):
    def __init__(self):
        self.normalizes = normalizes()

    def __call__(self, crops):
        return torch.stack([self.normalizes(transforms.ToTensor()(crop)) for crop in crops])


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        h, w = image.shape[:2]
        if h != self.output_size[0] or w != self.output_size[1]:
            if len(image.shape) > 2:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w, 1), order=3)
            else:
                image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        if len(image.shape) > 2:
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        else:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


def random_transform1(image_size):
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # res = self.transform(image=image)
    # image = res['image']
    # image = self.totensor(image)

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375),
                              max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    transform = {'train': transforms_train, 'valid': transforms_val}

    return transform


def mask_to_onehot(mask):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask, -1)
    for colour in range(9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map


def random_transform2():
    return iaa.SomeOf((0, 4), [
        iaa.Flipud(0.5, name="Flipud"),  # 50%的概率水平翻转（影响关键点）
        iaa.Fliplr(0.5, name="Fliplr"),  # 50%的概率垂直翻转（影响关键点）
        iaa.AdditiveGaussianNoise(scale=0.005 * 255),  # 添加高斯噪声（不影响关键点）
        iaa.GaussianBlur(sigma=1.0),  # 高斯模糊，使用高斯核的sigma取值范围在（0，3）之间，sigma的随机取值符合均匀分布（不影响关键点）
        iaa.LinearContrast((0.5, 1.5), per_channel=0.5),  # 增强或削弱图片的对比度（不影响关键点）
        iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),  # 缩放变换（影响关键点）
        iaa.Affine(rotate=(-40, 40)),  # 旋转（影响关键点）
        iaa.Affine(shear=(-16, 16)),  # 剪切（影响关键点）
        iaa.PiecewiseAffine(scale=(0.008, 0.03)),  # 扭曲（影响关键点）
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})  # 平移变换（影响关键点）
    ], random_order=True)


def augment(aug_transform, image, mask):
    mask = mask_to_onehot(mask)
    aug_det = aug_transform.to_deterministic()
    image_aug = aug_det.augment_image(image)
    segmap = ia.SegmentationMapOnImage(mask, nb_classes=np.max(mask) + 1, shape=image.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug, segmap_aug
