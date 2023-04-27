import os
import yaml
import torch
import logging
import argparse
from torch.backends import cudnn


def parse_args():

    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='Training with PyTorch')

    # Model parameters
    parser.add_argument('--model_type', type=str, metavar='NAME', default='AlexNet', help='model name')
    parser.add_argument('--model_dir', type=str, metavar='DIR', default='./model/AlexNet', help='root dir for model')
    parser.add_argument('--model_index', type=int, default=0, help='')
    parser.add_argument('--pretrain', action='store_true', help='')
    parser.add_argument('--path_state_dict', type=str, default='', help='')
    parser.add_argument('--finetune_state_dict', type=str, default='', help='')
    parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no_resume_opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--freeze', action='store_true', help='whether to freeze convolution layer')
    parser.add_argument('--version', type=str, default='', help='')
    parser.add_argument('--visual', action='store_false', help='')

    # Structure parameters
    parser.add_argument('--stages', type=int, default=4, help='')
    parser.add_argument('--in_dim', type=int, default=2, help='')
    parser.add_argument('--img_size', type=int, metavar='N', default=224, help='size of network input')
    parser.add_argument('--d_size', type=int, default=0, help='')
    parser.add_argument('--ape', action='store_true', help='whether to use absolute position embedding')
    parser.add_argument('--n_skip', type=int, default=0, help='using number of skip-connect')
    parser.add_argument('--patch_size', type=int, default=0, help='vit_patches_size')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--pool_size', default=0, type=int, help='')
    parser.add_argument('--use_layer_scale', action='store_false', help='')
    parser.add_argument('--layer_scale_init_value', default=1e-5, type=float, help='')
    parser.add_argument('--embed_dim', type=int, default=96, help='')
    parser.add_argument('--window_size', type=int, default=7, help='')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='')
    parser.add_argument('--attn_ratio', type=float, default=1., help='attention ratio')

    # Dataset parameters
    parser.add_argument('--task', type=str, metavar='NAME', default='classification', help='')
    parser.add_argument('--dataset', type=str, metavar='NAME', default='CatDogDataset', help='dataset name')
    parser.add_argument('--data_dir', type=str, metavar='DIR',
                        default='./data/classification/CatDogDataset', help='root dir for data')
    parser.add_argument('--list_dir', type=str, metavar='DIR', default='', help='root dir for list')
    parser.add_argument('--test_batch', action='store_false', help='')
    parser.add_argument('--dimension', type=int, metavar='N', default=2, help='')
    parser.add_argument('--split_n', type=float, default=0.9, help='')
    parser.add_argument('--n_channels', type=int, metavar='N', default=3, help='input channel of network')
    parser.add_argument('--num_classes', type=int, metavar='N', default=2, help='output channel of network')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--z_spacing', type=int, metavar='N', default=1, help='')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--tag', help='tag of experiment')

    # Training parameters
    parser.add_argument('--epochs', type=int, metavar='N', default=300, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, metavar='N', default=128, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, metavar='N', default=1, help='number of subprocesses')
    parser.add_argument('--deterministic', action='store_true', help='whether use deterministic training')
    parser.add_argument('--device', type=str, default='cpu', help='')

    # Optimizer parameters
    parser.add_argument('--self_optimizer', action='store_true', help='')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')

    # Learning rate schedule parameters
    parser.add_argument('--self_scheduler', action='store_true', help='')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr_cycle_mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr_cycle_decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr_cycle_limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay_epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')

    # Augmentation parameters
    # parser.add_argument('--transform', type=str, default='', help='transform function')
    parser.add_argument('--self_transform', action='store_false', help='')
    parser.add_argument('--norm_transform', action='store_true', help='')
    parser.add_argument('--no_aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--aug_repeats', type=int, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug_splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--re_num_splits', type=int, default=0,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup_off_epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')

    # Loss parameters
    parser.add_argument('--dice_loss_weight', type=float, default=0.5, help='')
    parser.add_argument('--jsd_loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce_loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce_target_thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--focal_loss', action='store_true', default=False, help='')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='')

    # Regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_connect', type=float, default=0.0, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.)')
    parser.add_argument('--drop_block', type=float, default=0.0, metavar='PCT',
                        help='Drop block rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT', help='')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn_tf', action='store_true', default=False,
                        help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
    parser.add_argument('--bn_momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn_eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync_bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist_bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split_bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model exponential moving average
    parser.add_argument('--model_ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model_ema_decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Distillation parameters
    parser.add_argument('--teacher_model', default='ViT', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "ViT"')
    parser.add_argument('--teacher_path', type=str, default='')
    parser.add_argument('--distillation_type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation_alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation_tau', default=1.0, type=float, help="")

    # Misc
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex_amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native_amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--channels_last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--pin_mem', action='store_false', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--eval_metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--rank", default=0, type=int, help='global rank')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--use_multi_epochs_loader', action='store_true', default=False,
                        help='use the multi_epochs_loader to save time at the beginning of every epoch')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')
    parser.add_argument('--log_wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')

    # Distributed training parameters
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # For huawei cloud
    parser.add_argument("--init_method", default='env://', type=str)
    parser.add_argument("--train_url", type=str)

    # Other parameters
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_interval', type=int, default=1, help='')
    parser.add_argument('--val_interval', type=int, default=5, help='')
    parser.add_argument('--write_interval', type=int, default=50, help='')
    parser.add_argument('--check_interval', type=int, default=5, help='')
    parser.add_argument('--save_interval', type=int, default=20, help='')
    parser.add_argument('--debug', action='store_false', help='')
    parser.add_argument('--debug_ratio', type=int, default=1, help='')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    return args, args_text


def get_snapshot_path(args):
    snapshot_path = './snapshot/'
    if args.debug:
        snapshot_path += 'tmp/'
    snapshot_path = snapshot_path + '{}'.format(args.model_type)
    snapshot_path = snapshot_path + "/{}V{}/{}".format('pretrained_' if args.pretrain else '',
                                                       args.model_index, args.dataset)
    snapshot_path = snapshot_path + '/epo' + str(args.epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed)
    snapshot_path = snapshot_path + '/{}'.format(args.mode)

    return snapshot_path


def deploy(args):
    if args.mode == 'train':
        mode_text = 'Training'
    else:
        mode_text = 'Testing'
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.n_gpu > 1:
            logging.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.n_gpu = 1

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if args.distributed:
        args.n_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        torch.distributed.init_process_group(backend='nccl', init_method=args.init_method,
                                             rank=args.rank, world_size=args.world_size)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        logging.info('{} in distributed mode with multiple processes, 1 GPU per process. Process {}, total {}.'.format(
                     mode_text, args.rank, args.world_size))
    elif args.device == 'cuda':
        logging.info('{} with a single process on {} GPUs.'.format(mode_text, args.n_gpu))
    else:
        logging.info('{} with a single process on CPU.'.format(mode_text))
    assert args.rank >= 0

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
