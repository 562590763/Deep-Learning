def get_config():
    model_config = {

        # classification
        'AlexNet': {
            'model_dir': './model/AlexNet',
            'pretrain': True,
            'path_state_dict': ['./model/AlexNet/path_state_dict/alexnet-owt-4df8aa71.pth'],
            'epochs': 90,
            'batch_size': 128,
            'lr': 0.01,  # 0.001
            'img_size': 224,
            'momentum': 0.9,
            'freeze': 0,
            'weight_decay': 0.0005,
            'decay_epochs': 3,
            'gamma': 0.1,
        },
        'PVT': {
            'model_dir': './model/PVT',
            'epochs': 300,
            'num_workers': 10,
            'batch_size': 128,
            'lr': 0.01,  # 5e-4
            'decay_epochs': 30,
            'patch_size': 4,
            'img_size': 224,
            'weight_decay': 0.05,
            'drop_path': 0.1,
            'train_interpolation': 'bicubic',
            'opt_eps': 1e-8,
            'version': ['pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large']
        },
        'TNT': {
            'model_dir': './model/TNT',
            'path_state_dict': ['./model/TNT/path_state_dict/' + i + '.pth.tar'
                                for i in ['tnt_s_patch16_224', 'tnt_b_patch16_224']],
            'epochs': 200,
            'patch_size': 16,
            'batch_size': 32,
            'opt': 'sgd',
            'num_workers': 4,
            'lr': 0.01,
            'img_size': 224,
            'momentum': 0.9,
            'warmup_lr': 1e-4,
            'decay_epochs': 30,
            'warmup_epochs': 3,
            'reprob': 0,
            'remode': 'const',
            'mixup': 0.,
            'cutmix': 0.,
            'dist_bn': '',
            'version': ['tnt_s_patch16_224', 'tnt_b_patch16_224']
        },
        'PyramidTNT': {
            'model_dir': './model/TNT',
            'path_state_dict': ['./model/TNT/path_state_dict/' + i + '.pth.tar'
                                for i in ['ptnt_ti_patch16_192', 'ptnt_s_patch16_256',
                                          'ptnt_m_patch16_256', 'ptnt_b_patch16_256']],
            'epochs': 200,
            'patch_size': 16,
            'batch_size': 32,
            'opt': 'sgd',
            'num_workers': 4,
            'lr': 0.01,
            'img_size': [192, 256, 256, 256],
            'momentum': 0.9,
            'warmup_lr': 1e-4,
            'decay_epochs': 30,
            'warmup_epochs': 3,
            'reprob': 0,
            'remode': 'const',
            'mixup': 0.,
            'cutmix': 0.,
            'dist_bn': '',
            'version': ['ptnt_ti_patch16_192', 'ptnt_s_patch16_256', 'ptnt_m_patch16_256', 'ptnt_b_patch16_256']
        },
        'PoolFormer': {
            'model_dir': './model/PoolFormer',
            'pretrain': True,
            'path_state_dict': ['./model/PoolFormer/path_state_dict/' + i + '.pth.tar'
                                for i in ['poolformer_s12', 'poolformer_s24', 'poolformer_s36',
                                          'poolformer_m36', 'poolformer_m48']],
            'epochs': 300,
            'batch_size': 128,
            'num_workers': 8,
            'lr': 0.01,  # 0.001
            'img_size': 224,
            'momentum': 0.9,
            'weight_decay': 0.05,
            'version': [
                'poolformer_s12', 'poolformer_s24', 'poolformer_s36',
                'poolformer_m36', 'poolformer_m48',
            ]
        },

        # segmentation
        'UNet': {
            'model_dir': './model/UNet',
            'epochs': 150,
            'batch_size': 24,  # 2
            'num_workers': 8,
            'lr': 0.01,
            'img_size': 224,  # 512
            'momentum': 0.99,
            'train_interpolation': 'bicubic',
            'weight_decay': 0.001,
            'self_transform': False,
            'norm_transform': True,
            'version': ['unet_base']
        },
        'VNet': {
            'model_dir': './model/VNet',
            'in_dim': 3,
            'd_size': 64,
            'epochs': 300,
            'batch_size': 10,
            'opt': 'adam',
            'lr': 0.01,  # 0.0001
            'img_size': 128,
            'momentum': 0.99,
            'weight_decay': 1e-8,
            'version': ['vnet_base']
        },
        'AttentionUNet': {
            'model_dir': './model/AttentionUNet',
            'epochs': 150,
            'batch_size': 24,  # 2
            'self_optimizer': True,
            'opt': 'sgd',  # adam
            'self_scheduler': True,
            'num_workers': 8,
            'lr': 0.01,
            'img_size': 224,  # 512
            'weight_decay': 0.0001,  # 0.001
            'version': ['attention_unet_base']
        },
        'FTN': {
            'model_dir': './model/FTN',
            'epochs': 150,  # 100
            'batch_size': 24,  # 20
            'num_workers': 8,
            'lr': 0.01,  # 0.0002
            'img_size': 224,  # 384
            'version': ['FTN_4', 'FTN_8', 'FTN_12']
        },
        'PoolUNet': {
            'model_dir': './model/PoolUNet',
            'epochs': 150,
            'batch_size': 24,
            'num_workers': 8,
            'pool_size': 3,
            'self_optimizer': True,
            'opt': 'sgd',
            'self_scheduler': True,
            'lr': 0.01,
            'img_size': 224,
            'n_skip': 3,
            'patch_size': 16,
            'drop_rate': 0.1,
            'drop_path': 0.1,
            'version': ['poolunet_base', 'poolunet_large']
        },
        'TransUNet': {
            'model_dir': './model/TransUNet',
            'pretrain': True,
            'path_state_dict': ['', '', '', '', '', './model/TransUNet/path_state_dict/R50+ViT-B_16.npz', '', ''],
            'epochs': 150,
            'batch_size': 24,
            'num_workers': 8,
            'self_optimizer': True,
            'opt': 'sgd',
            'self_scheduler': True,
            'lr': 0.01,
            'img_size': 224,
            'n_skip': 3,
            'patch_size': 16,
            'drop_rate': 0.1,
            'version': [
                'ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32',
                'ViT-H_14', 'R50-ViT-B_16', 'R50-ViT-L_16', 'testing'
            ]
        },
        'SwinUNet': {
            'model_dir': './model/SwinUNet',
            'pretrain': True,
            'path_state_dict': ['./model/SwinUNet/path_state_dict/swin_tiny_patch4_window7_224.pth'],
            'epochs': 300,
            'batch_size': 48,  # 24
            'num_workers': 8,
            'opt': 'sgd',
            'lr': 0.01,  # 5e-4
            'img_size': 224,
            'smoothing': 0.1,
            'decay_epochs': 30,
            'warmup_lr': 5e-7,
            'warmup_epochs': 20,
            'train_interpolation': 'bicubic',
            'drop_rate': 0,
            'drop_path': 0.1,
            'opt_eps': 1e-8,
            # 'opt_betas': 0.9,
            'patch_size': 4,
            'version': ['swin_tiny_patch4_window7_224']
        },
        'MISSFormer': {
            'model_dir': './model/MISSFormer',
            'epochs': 400,
            'batch_size': 24,
            'num_workers': 8,
            'self_optimizer': True,
            'opt': 'sgd',
            'self_scheduler': True,
            'lr': 0.05,
            'img_size': 224,
            'dice_loss_weight': 0.6,
            'version': ['missformer_base']
        },
        'DAEFormer': {
            'model_dir': './model/DAEFormer',
            'epochs': 400,
            'batch_size': 24,
            'num_workers': 8,
            'self_optimizer': True,
            'opt': 'sgd',
            'self_scheduler': True,
            'lr': 0.05,
            'img_size': 224,
            'dice_loss_weight': 0.6,
            'version': ['daeformer_base']
        },
    }

    dataset_config = {

        # classification
        'CatDogDataset': {
            'task': 'classification',
            'data_dir': './data/classification/CatDogDataset',
            'n_channels': 3,
            'num_classes': 2,
            'debug_ratio': 100,
            'transform': 'catdogdataset_transform(args.img_size)',
        },
        'CIFAR10': {
            'task': 'classification',
            'data_dir': './data/classification/CIFAR10',
            'n_channels': 3,
            'num_classes': 10,
            'zip': True,
            'transform': 'cifar10dataset_transform(args.img_size)',
        },
        'MNIST': {
            'task': 'classification',
            'data_dir': './data/classification/MNIST',
            'n_channels': 1,
            'num_classes': 10,
            'zip': True,
            'transform': 'mnistdataset_transform(args.img_size)',
        },
        'ImageNet200': {
            'task': 'classification',
            'data_dir': './data/classification/ImageNet/classes200',
            'n_channels': 3,
            'num_classes': 200,
            'debug_ratio': 500,
            'transform': 'imagenetdataset_transform(args.img_size)',
        },

        # segmentation
        'CarDataset': {
            'task': 'segmentation',
            'data_dir': './data/segmentation/CarDataset',
            'n_channels': 3,
            'num_classes': 2,
            'transform': 'cardataset_transform(args)',
        },
        'MelanomaDataset': {
            'task': 'segmentation',
            'data_dir': './data/segmentation/MelanomaDataset',
            'n_channels': 3,
            'num_classes': 2,
            'debug_ratio': 50,
            'transform': 'melanomadataset_transform(args)',
        },
        'MembraneDataset': {
            'task': 'segmentation',
            'data_dir': './data/segmentation/MembraneDataset',
            'n_channels': 3,
            'num_classes': 2,
            'transform': 'membranedataset_transform(args)',
        },
        'BraTsDataset': {
            'task': 'segmentation',
            'data_dir': './data/segmentation/BraTsDataset',
            'list_dir': './data/segmentation/BraTsDataset/lists',
            'dimension': 3,
            'n_channels': 1,
            'num_classes': 4,
            'debug_ratio': 10,
            'zip': True,
            'transform': 'bratsdataset_transform(args)',
        },
        'SynapseDataset': {
            'task': 'segmentation',
            'data_dir': './data/segmentation/SynapseDataset',
            'list_dir': './data/segmentation/SynapseDataset/lists',
            'test_batch': False,
            'n_channels': 1,
            'num_classes': 9,
            'debug_ratio': 40,
            'zip': True,
            'transform': 'synapsedataset_transform(args)',
        },
        'ACDCDataset': {
            'task': 'segmentation',
            'data_dir': './data/segmentation/ACDCDataset',
            'n_channels': 1,
            'num_classes': 4,
            'debug_ratio': 30,
            'zip': True,
            'transform': 'acdcdataset_transform(args)',
        },
    }
    return model_config, dataset_config
