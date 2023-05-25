from train import *
from inference import classification_inference, segmentation_inference

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

    # TODO
    args.debug = False
    args.mode = 'test'
    args.num_workers = 1
    args.visual = False
    if not args.self_transform:
        args.transform = build_transform(args)
    if not args.test_batch:
        args.batch_size = 1

    snapshot_path = get_snapshot_path(args)
    args.initial_checkpoint = snapshot_path
    args.finetune_state_dict = './model/{}/path_state_dict/{}_V{}.pth'.format(
        args.model_type, args.dataset, args.model_index)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    deploy(args)
    set_seed(args.seed)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.n_gpu)

    inference = {'classification': classification_inference, 'segmentation': segmentation_inference, }
    inference[args.task](args, snapshot_path)
