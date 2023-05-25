import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from torchsummaryX import summary
from model.UNet.networks.unet import *
from model.VNet.networks.vnet import *
from model.PVT.networks.pvt import *
from model.FTN.networks.ftn import *
from model.TNT.networks.tnt import *
from model.TNT.networks.pyramid_tnt import *
from model.PoolFormer.networks.poolformer import *
from model.AttentionUNet.networks.attention_unet import *
from model.PoolUNet.networks.poolunet import *
from model.DAEFormer.networks.daeformer import *
from model.MISSFormer.networks.missformer import *
from model.SwinUNet.networks.vision_transformer import SwinUnet
from model.TransUNet.networks.vit_seg_modeling import VisionTransformer
from model.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils.register import get_models, get_model_type
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.models import create_model, load_checkpoint, convert_splitbn_model

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


def get_model(args):
    """
    Create models and load parameters
    :param args:
    :return:
    """
    model_type = get_model_type()
    if args.mode == 'train' and args.pretrain:
        print("pretrained_path:{}".format(args.path_state_dict))
        print("---start load pretrained model of {}---".format(args.model_type))
    elif args.mode == 'test':
        if not os.path.exists(args.finetune_state_dict):
            raise NotImplementedError("{} is not finetuned.".format(args.model_type))
        else:
            print("finetune_path:{}".format(args.finetune_state_dict))
    if args.model_type == 'AlexNet':
        model = models.alexnet()
        if args.mode == 'train' and args.pretrain:
            pretrained_state_dict = torch.load(args.path_state_dict)
            model.load_state_dict(pretrained_state_dict)
        if args.num_classes != 1000:
            num_ftrs = model.classifier._modules["6"].in_features
            model.classifier._modules["6"] = nn.Linear(num_ftrs, args.num_classes)
        if args.n_channels != 3:
            model.features._modules["0"] = nn.Conv2d(args.n_channels, 64,
                                                     kernel_size=(11, 11), stride=(4, 4), padding=2)
        if args.mode == 'test':
            best_state_dict = torch.load(args.finetune_state_dict, map_location="cpu")
            model.load_state_dict(best_state_dict)
    elif args.model_type == 'TransUNet':
        config_vit = CONFIGS_ViT_seg[args.version]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.version.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.patch_size), int(args.img_size / args.patch_size))
        model = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
        if args.mode == 'train' and args.pretrain:
            model.load_from(weights=np.load(args.path_state_dict))
        elif args.mode == 'test':
            best_state_dict = torch.load(args.finetune_state_dict, map_location="cpu")
            model.load_state_dict(best_state_dict)
    elif args.model_type == 'SwinUNet':
        model = SwinUnet(args)
        if args.mode == 'train' and args.pretrain:
            model.load_from(args.path_state_dict)
        elif args.mode == 'test':
            best_state_dict = torch.load(args.finetune_state_dict, map_location="cpu")
            model.load_state_dict(best_state_dict)
    elif args.model_type in model_type:
        # model = create_model(
        #     args.version,
        #     pretrained=False,
        #     num_classes=args.num_classes,
        #     drop_rate=args.drop_rate,
        #     drop_connect_rate=args.drop_connect,
        #     drop_path_rate=args.drop_path,
        #     drop_block_rate=args.drop_block,
        #     global_pool=args.gp,
        #     bn_tf=args.bn_tf,
        #     bn_momentum=args.bn_momentum,
        #     bn_eps=args.bn_eps,
        #     scriptable=args.torchscript,
        #     checkpoint_path=args.path_state_dict)
        model = get_models(args.version)(args)
        if args.mode == 'train' and args.pretrain:
            if not os.path.exists(args.path_state_dict):
                raise NotImplementedError("{} is not pretrained.".format(args.model_type))
            checkpoint = torch.load(args.path_state_dict, map_location="cpu")
            model.load_state_dict(checkpoint)
        if args.num_classes != 1000:
            assert hasattr(model, 'reset_classifier'), \
                'Model must have `num_classes` attr if not set on cmd line/config.'
            model.reset_classifier(args.num_classes)
        if args.mode == 'test':
            best_state_dict = torch.load(args.finetune_state_dict, map_location="cpu")
            model.load_state_dict(best_state_dict)
    else:
        raise NotImplementedError("{} is not imported".format(args.model_type))

    if args.debug and args.visual:
        # It cannot be turned on during training, otherwise the model parameter import will be affected.
        if args.in_dim == 2:
            inputs = torch.randn(1, args.n_channels, args.img_size, args.img_size)
        else:
            inputs = torch.randn(1, args.n_channels, args.img_size, args.img_size, args.d_size)
        summary(model, inputs)
        flops, params = profile(model, inputs=(inputs,))
        print('Use {}: FLOPs = {:.3f}G, Params = {:.3f}M'.
              format(args.model_type, flops / 1000 ** 3, params / 1000 ** 2))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        logging.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.n_gpu > 1:
        if use_amp == 'apex':
            logging.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.n_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    elif args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    model = model.to(args.device)

    # setup synchronized BatchNorm for distributed training
    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                logging.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # if args.torchscript:
    #     assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
    #     assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
    #     model = torch.jit.script(model)

    # setup exponential moving average of model weights, SWA could be used here too
    # model_ema = None
    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEmaV2(
    #         model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
    #     if args.resume:
    #         load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    # if args.distributed:
    #     if has_apex and use_amp == 'apex':
    #         # Apex DDP preferred unless native amp is activated
    #         if args.local_rank == 0:
    #             _logger.info("Using NVIDIA APEX DistributedDataParallel.")
    #         model = ApexDDP(model, delay_allreduce=True)
    #     else:
    #         if args.local_rank == 0:
    #             _logger.info("Using native Torch DistributedDataParallel.")
    #         model = NativeDDP(model, device_ids=[args.local_rank])
    #     # NOTE: EMA model does not need to be wrapped by DDP

    # teacher_model = None
    # if args.distillation_type != 'none':
    #     assert args.teacher_path, 'need to specify teacher-path when using distillation'
    #     print(f"Creating teacher model: {args.teacher_model}")
    #     teacher_model = create_model(
    #         args.teacher_model,
    #         pretrained=False,
    #         num_classes=args.num_classes,
    #         global_pool='avg',
    #     )
    #     if args.teacher_path.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.teacher_path, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.teacher_path, map_location='cpu')
    #     teacher_model.load_state_dict(checkpoint['model'])
    #     teacher_model.to(device)
    #     teacher_model.eval()

    return model
