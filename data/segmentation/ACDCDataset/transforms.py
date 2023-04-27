import torchvision.transforms as transforms
from utils.augment import RandomGenerator


def acdcdataset_transform(args):
    train_transform = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])

    # t = [transforms.ToPILImage()]
    # size = int((256 / 224) * args.img_size)
    # t.append(transforms.Resize((size, size), interpolation=3))
    # t.append(transforms.CenterCrop(args.img_size))
    # t.append(transforms.ToTensor())
    # temp = t[:]
    # valid_transform_label = transforms.Compose(temp)
    # t.append(transforms.Normalize(sum(IMAGENET_DEFAULT_MEAN) / 3, sum(IMAGENET_DEFAULT_STD) / 3))
    # valid_transform_image = transforms.Compose(t)
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    transform = {
        'train': train_transform,
        'valid': valid_transform,
        'test': valid_transform
    }
    return transform
