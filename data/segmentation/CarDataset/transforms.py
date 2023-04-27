import torchvision.transforms as transforms
from utils.augment import RandomGenerator


def cardataset_transform(args):
    train_transform = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])

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
