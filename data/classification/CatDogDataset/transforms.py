import torchvision.transforms as transforms
from utils.augment import normalizes, TenCropLambda


def catdogdataset_transform(img_size):

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalizes(),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(img_size, vertical_flip=False),
        TenCropLambda()
        # transforms.Lambda(lambda crops: torch.stack([normalizes()(transforms.ToTensor()(crop)) for crop in crops]))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    transform = {
        'train': train_transform,
        'valid': valid_transform,
        'test': test_transform
    }

    return transform
