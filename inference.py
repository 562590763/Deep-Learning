# Import base library
import time
import logging
import torch
import numpy as np
import SimpleITK as sitk
# Import function
from scipy.ndimage import zoom
from utils.model import get_model
from utils.dataset import getdata
from utils.tools import write_results
from torch.utils.data import DataLoader
from timm.models.layers import to_2tuple
from timm.utils import accuracy, AverageMeter
from torchvision.transforms import transforms
from utils.metric import calculate_metric_percase


def classification_inference(args, snapshot_path):
    logging.info('\nSnapshot Path: {}.'.format(snapshot_path))
    # ================================= step 0/5 Config =================================
    logging.info('{0:*^98}'.format("  Args  "))
    logging.info(str(args))
    test_dir = args.data_dir + '/test'
    test_transform = args.transform[args.mode]
    log_interval = args.log_interval
    device = args.device

    # ================================== step 1/5 Model =================================
    logging.info('\n{0:*^98}'.format("  " + args.model_type + "  "))
    model = get_model(args)

    # ================================== step 2/5 Data ==================================
    # Build an instance of Dataset
    logging.info('\n{0:*^98}'.format("  " + args.dataset + "  "))
    test_data = getdata(args.dataset, data_dir=test_dir, list_dir=args.list_dir, debug=args.debug,
                        debug_ratio=args.debug_ratio, mode=args.mode, split_n=args.split_n,
                        image_size=args.img_size, self_transform=args.self_transform,
                        norm_transform=args.norm_transform, transform=test_transform, seed=args.seed)
    logging.info("The length of the test set is: {}.".format(len(test_data)))

    # Build dataloader
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem)

    # ================================== step 3/5 Test ===================================
    logging.info('\n{0:*^98}'.format("  Test  "))
    logging.info("{} iterations per epoch, use device:{}.\n".format(len(testloader), device))
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():

        start = time.time()
        for i_batch, sampled_batch in enumerate(testloader):

            # predict
            image_batch, label_batch = sampled_batch[:2]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            if args.channels_last:
                image_batch = image_batch.contiguous(memory_format=torch.channels_last)
            outputs = model(image_batch)

            # measure accuracy
            acc1, acc5 = accuracy(outputs.detach(), label_batch, topk=(1, 5))
            top1.update(acc1.item(), image_batch.size(0))
            top5.update(acc5.item(), image_batch.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if i_batch % log_interval == 0:
                logging.info(
                    'Testing:Iteration[{:0>3}/{:0>3}]  '
                    'Time:{batch_time.val:5.3f}s  '
                    'Acc@1:{top1.val:>6.2f}({top1.avg:>5.2f})  '
                    'Acc@5:{top5.val:>6.2f}({top5.avg:>5.2f}).'.format(
                        i_batch + 1, len(testloader), batch_time=batch_time,
                        rate_avg=image_batch.size(0) / batch_time.avg,
                        top1=top1, top5=top5))

    # top1a, top5a = top1.avg, top5.avg
    # results = OrderedDict(
    #     top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
    #     top5=round(top5a, 4), top5_err=round(100 - top5a, 4))
    # logging.info(' * Acc@1:{:.3f}({:.3f})  Acc@5:{:.3f}({:.3f}).'.format(
    #     results['top1'], results['top1_err'], results['top5'], results['top5_err']))
    # write_results(snapshot_path, results)

    return "Testing Finished!"


def segmentation_inference(args, snapshot_path):
    logging.info('\nSnapshot Path: {}.'.format(snapshot_path))
    # ================================= step 0/5 Config =================================
    logging.info('{0:*^98}'.format("  Args  "))
    logging.info(str(args))
    test_dir = args.data_dir + '/test'
    test_transform = args.transform[args.mode]
    log_interval = args.log_interval
    device = args.device

    # ================================== step 1/5 Model =================================
    logging.info('\n{0:*^98}'.format("  " + args.model_type + "  "))
    model = get_model(args)

    # ================================== step 2/5 Data ==================================
    # Build an instance of Dataset
    logging.info('\n{0:*^98}'.format("  " + args.dataset + "  "))
    test_data = getdata(args.dataset, data_dir=test_dir, list_dir=args.list_dir, debug=args.debug,
                        debug_ratio=args.debug_ratio, mode=args.mode, split_n=args.split_n,
                        image_size=args.img_size, self_transform=args.self_transform,
                        norm_transform=args.norm_transform, transform=test_transform, seed=args.seed)
    logging.info("The length of the test set is: {}.".format(len(test_data)))

    # Build dataloader
    testloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem)

    # ================================== step 3/5 Test ===================================
    logging.info('\n{0:*^98}'.format("  Test  "))
    logging.info("{} iterations per epoch, use device:{}.\n".format(len(testloader), device))
    batch_time = AverageMeter()
    dice = AverageMeter()
    hausdorff = AverageMeter()
    metric_list = 0.0

    model.eval()
    with torch.no_grad():

        start = time.time()
        for i_batch, sampled_batch in enumerate(testloader):

            # predict
            image_batch, label_batch = sampled_batch[:2]
            if len(sampled_batch) == 3:
                case_name = sampled_batch[2][0]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            if args.channels_last:
                image_batch = image_batch.contiguous(memory_format=torch.channels_last)

            # measure dice and hausdorff distance
            if not args.test_batch:
                metric_i = test_single_volume(image_batch, label_batch, model, classes=args.num_classes,
                                              image_size=args.img_size, norm_transform=args.norm_transform,
                                              test_save_path=snapshot_path, case=case_name, z_spacing=args.z_spacing)
            else:
                metric_i = test_batch_data(image_batch, label_batch, model,
                                           classes=args.num_classes, norm_transform=args.norm_transform)
            metric_list += np.array(metric_i) * image_batch.size(0)
            dice.update(np.mean(metric_i, axis=0)[0], image_batch.size(0))
            hausdorff.update(np.mean(metric_i, axis=0)[1], image_batch.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if i_batch % log_interval == 0:
                logging.info(
                    'Testing:Iteration[{:0>3}/{:0>3}]  '
                    'Time:{batch_time.val:5.3f}s  '
                    'Dice:{dice.val:>5.3f}({dice.avg:>5.3f})  '
                    'Hausdorff:{hausdorff.val:>5.2f}({hausdorff.avg:>5.2f}).'.format(
                        i_batch + 1, len(testloader), batch_time=batch_time,
                        rate_avg=image_batch.size(0) / batch_time.avg,
                        dice=dice, hausdorff=hausdorff))

    metric_list = metric_list / len(test_data)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f.' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f.' % (performance, mean_hd95))

    return "Testing Finished!"


def test_batch_data(image, label, model, classes, norm_transform=False):
    label = label.cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :, :].unsqueeze(0)
        output = model(slice)
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)
        pred = out.cpu().detach().numpy()
        prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list


def test_single_volume(image, label, model, classes, image_size=224,
                       norm_transform=False, test_save_path=None, case=None, z_spacing=1):
    image_size = to_2tuple(image_size)
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != image_size[0] or y != image_size[1]:
            slice = zoom(slice, (image_size[0] / x, image_size[1] / y), order=3)  # previous using 0
        if norm_transform:
            image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
            input = image_transforms(slice).unsqueeze(0).float().cuda()
        else:
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        outputs = model(input)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != image_size[0] or y != image_size[1]:
            pred = zoom(out, (x / image_size[0], y / image_size[1]), order=0)
        else:
            pred = out
        prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # if test_save_path is not None and case is not None:
    #     img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    #     prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    #     lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #     img_itk.SetSpacing((1, 1, z_spacing))
    #     prd_itk.SetSpacing((1, 1, z_spacing))
    #     lab_itk.SetSpacing((1, 1, z_spacing))
    #     sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
    #     sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
    #     sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")

    return metric_list
