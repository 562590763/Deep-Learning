# Import base library
import logging
import random
import warnings
import torch
import numpy as np
# Import function
from utils.transforms import *
from utils.config import get_config
from utils.args import parse_args, deploy, get_snapshot_path
from trainer import classification_trainer, segmentation_trainer

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    args, _ = parse_args()
    args.model_type = 'PoolUNet'
    args.model_index = 0
    args.dataset = 'SynapseDataset'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config, dataset_config = get_config()

    for key, value in model_config[args.model_type].items():
        if isinstance(value, int) or isinstance(value, float):
            exec("args.{}={}".format(key, value))
        elif isinstance(value, str):
            exec("args.{}='{}'".format(key, value))
        elif isinstance(value, list):
            if isinstance(value[0], str):
                exec("args.{}='{}'".format(key, value[args.model_index]))
            elif isinstance(value[0], int):
                exec("args.{}={}".format(key, value[args.model_index]))
    for key, value in dataset_config[args.dataset].items():
        if isinstance(value, int) or isinstance(value, float) or key == 'transform':
            exec("args.{}={}".format(key, value))
        else:
            exec("args.{}='{}'".format(key, value))

    # TODO(FFX) 修改设置
    args.debug = True
    args.num_workers = 1
    if args.debug:
        args.batch_size = 2
        args.epochs = 1
        args.val_interval = 1
    if not args.self_transform:
        args.transform = build_transform(args)

    snapshot_path = get_snapshot_path(args)
    args.initial_checkpoint = snapshot_path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    deploy(args)
    set_seed(args.seed)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.n_gpu)

    trainer = {'classification': classification_trainer, 'segmentation': segmentation_trainer, }
    trainer[args.task](args, snapshot_path)
