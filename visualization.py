import SimpleITK as sitk
from train import *
from scipy.ndimage import zoom
from utils.model import get_model
from utils.dataset import getdata
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.tools import show_seg, show_seg3d

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
    args.debug = False
    args.mode = 'test'
    args.batch_size = 1
    args.num_workers = 1
    args.visual = False
    args.self_transform = True
    if not args.self_transform:
        args.transform = build_transform(args)

    snapshot_path = get_snapshot_path(args)
    args.initial_checkpoint = snapshot_path
    # args.finetune_state_dict = './model/{}/path_state_dict/{}_V{}.pth'.format(
    #     args.model_type, args.dataset, args.model_index)
    args.finetune_state_dict = './model/PoolUNet/path_state_dict/SynapseDataset_V0.pth'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    deploy(args)
    set_seed(args.seed)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.n_gpu)
    device = args.device

    writer = SummaryWriter(snapshot_path + '/visual')
    test_dir = args.data_dir + '/test'
    save_path = snapshot_path + '/result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + '/Image')
        os.makedirs(save_path + '/GroundTruth')
        os.makedirs(save_path + '/{}'.format(args.model_type))

    test_transform = args.transform[args.mode]
    model = get_model(args)
    test_data = getdata(args.dataset, data_dir=test_dir, list_dir=args.list_dir, debug=args.debug,
                        debug_ratio=args.debug_ratio, mode=args.mode, split_n=args.split_n, image_size=args.img_size,
                        self_transform=args.self_transform, transform=test_transform, seed=args.seed)
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=1, pin_memory=args.pin_mem)

    for i_batch, sampled_batch in enumerate(testloader):
        image_batch, label_batch = sampled_batch[:2]
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        if args.channels_last:
            image_batch = image_batch.contiguous(memory_format=torch.channels_last)

        if not args.test_batch:
            image_batch, label_batch = image_batch.squeeze(0), label_batch.squeeze(0)
            index = image_batch.size()[0] // 3 * 2
            image, label = image_batch[index].cpu().detach().numpy(), label_batch[index].unsqueeze(0)  # H, W / C, H, W
            input = image.copy()
            x, y = input.shape[0], input.shape[1]
            if x != args.img_size or y != args.img_size:
                input = zoom(input, (args.img_size / x, args.img_size / y), order=3)
            input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).float().cuda()  # 1, C, H, W
        else:
            index = image_batch.size()[0] // 2
            image, label = image_batch[index].unsqueeze(0), label_batch[index].unsqueeze(0)  # 1, C, H, W / C, H, W
            input = image.copy()
        outputs = model(input)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        outputs = outputs.cpu().detach().numpy()
        if not args.test_batch and x != args.img_size or y != args.img_size:
            outputs = zoom(outputs, (x / args.img_size, y / args.img_size), order=0)
        outputs = torch.from_numpy(outputs).unsqueeze(0).float().cuda()  # C, H, W

        # Image
        img_itk = sitk.GetImageFromArray(image[np.newaxis, :, :].astype(np.float32))
        img_itk = sitk.Cast(sitk.RescaleIntensity(img_itk, ), sitk.sitkUInt8)
        show_seg3d(img_itk, save_path + '/Image/{}.png'.format(i_batch))
        # Ground Truth
        lab_itk = sitk.GetImageFromArray(label.cpu().detach().numpy())
        show_seg(sitk.LabelOverlay(img_itk, lab_itk), save_path +
                 '/GroundTruth/{}.png'.format(i_batch), "GroundTruth")
        # Prediction
        prd_itk = sitk.GetImageFromArray(outputs.to(torch.uint8).cpu().detach().numpy())
        show_seg(sitk.LabelOverlay(img_itk, prd_itk), save_path +
                 '/{}/{}.png'.format(args.model_type, i_batch), args.model_type)

        input = (input - input.min()) / (input.max() - input.min())
        writer.add_image('visual/Image', input.squeeze(0), i_batch)
        writer.add_image('visual/Prediction', outputs / args.num_classes, i_batch)
        writer.add_image('visual/GroundTruth', label / args.num_classes, i_batch)
