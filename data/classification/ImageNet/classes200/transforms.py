import torchvision.transforms as transforms
from utils.augment import normalizes, TenCropLambda


def imagenetdataset_transform(img_size):

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalizes(),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(img_size, vertical_flip=False),
        TenCropLambda()
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
