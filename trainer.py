# Import base library
import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Import function
from utils.model import get_model
from utils.dataset import getdata
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from timm.optim import create_optimizer
from utils.loss import DiceLoss, FocalLoss
from timm.scheduler import create_scheduler
from timm.data import Mixup, FastCollateMixup
from timm.models import convert_splitbn_model
from timm.loss import LabelSmoothingCrossEntropy, \
    SoftTargetCrossEntropy, JsdCrossEntropy
from utils.metric import calculate_segmentation_metric


def log(text, has_checkpoint=True):
    if not has_checkpoint:
        logging.info(text)


def classification_trainer(args, snapshot_path):
    checkpoint_path = snapshot_path + '/checkpoint.pth'
    has_checkpoint = False
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        has_checkpoint = True
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    log('\nSnapshot Path: {}.'.format(snapshot_path), has_checkpoint)
    # ================================= step 0/5 Config =================================
    log('{0:*^98}'.format("  Args  "), has_checkpoint)
    log(str(args), has_checkpoint)
    max_epoch = args.epochs
    lr = args.lr
    batch_size = args.batch_size * args.n_gpu
    num_workers = args.num_workers
    momentum = args.momentum
    train_dir = args.data_dir + '/train'
    if not os.path.exists(train_dir):
        logging.error('Training folder does not exist at: {}.'.format(train_dir))
        exit(1)
    train_transform = args.transform[args.mode]
    valid_transform = args.transform['valid']
    val_interval = args.val_interval
    log_interval = args.log_interval
    check_interval = args.check_interval
    save_interval = args.save_interval  # epochs / 6
    device = args.device
    writer = SummaryWriter(snapshot_path + '/log')

    # ================================== step 1/5 Model =================================
    log('\n{0:*^98}'.format("  " + args.model_type + "  "), has_checkpoint)
    model = get_model(args)
    if has_checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    # ================================== step 2/5 Data ==================================
    # Build an instance of Dataset
    log('\n{0:*^98}'.format("  " + args.dataset + "  "), has_checkpoint)
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense.'
        num_aug_splits = args.aug_splits
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if not args.no_prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    train_data = getdata(args.dataset, data_dir=train_dir, list_dir=args.list_dir, debug=args.debug,
                         debug_ratio=args.debug_ratio, mode=args.mode, split_n=args.split_n,
                         image_size=args.img_size, self_transform=args.self_transform,
                         norm_transform=args.norm_transform, transform=train_transform, seed=args.seed)
    valid_data = getdata(args.dataset, data_dir=train_dir, list_dir=args.list_dir, debug=args.debug,
                         debug_ratio=args.debug_ratio, mode="valid", split_n=args.split_n,
                         image_size=args.img_size, self_transform=args.self_transform,
                         norm_transform=args.norm_transform, transform=valid_transform, seed=args.seed)
    log("The length of train set and val set is: {} and {}.".format(len(train_data), len(valid_data)), has_checkpoint)

    # Build dataloader
    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=args.pin_mem, collate_fn=collate_fn)
    validloader = DataLoader(valid_data, batch_size=batch_size, num_workers=0,
                             pin_memory=args.pin_mem, collate_fn=collate_fn)

    # ================================== step 3/5 Loss ==================================
    log('\n{0:*^98}'.format("  Loss  "), has_checkpoint)
    # ce_loss = nn.CrossEntropyLoss()
    # ce_loss = ce_loss.to(device)
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
        log("Use JsdCrossEntropy loss function for training.", has_checkpoint)
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy()
        log("Use SoftTargetCrossEntropy loss function for training.", has_checkpoint)
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        log("Use LabelSmoothingCrossEntropy loss function for training.", has_checkpoint)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
        log("Use CrossEntropyLoss loss function for training.", has_checkpoint)
    validate_loss_fn = nn.CrossEntropyLoss()
    log("Use CrossEntropyLoss loss function for validation.", has_checkpoint)
    train_loss_fn, validate_loss_fn = train_loss_fn.to(device), validate_loss_fn.to(device)

    # ================================= step 4/5 Optimizer ===============================
    if args.freeze:
        fc_params_id = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': lr * 0.1},  # 0
            {'params': model.classifier.parameters(), 'lr': lr}], momentum=momentum)
    elif args.self_optimizer:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=args.weight_decay)
    else:
        optimizer = create_optimizer(args, model)

    if args.self_scheduler:
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epochs, gamma=args.gamma)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=5)
        pass
    else:
        scheduler, num_epochs = create_scheduler(args, optimizer)

    if has_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if not args.self_scheduler:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # ================================= step 5/5 Train ===================================
    log('\n{0:*^98}'.format("  Train  "), has_checkpoint)
    iter_num = checkpoint['iter_num'] if has_checkpoint else 0
    iterator = range(checkpoint['epoch'] if has_checkpoint else 0, max_epoch)
    max_iteration = max_epoch * len(trainloader)  # max_epoch = max_iteration // len(trainloader) + 1
    log("{} iterations per epoch, {} max iterations, use device:{}.\n".
        format(len(trainloader), max_iteration, device), has_checkpoint)
    for epoch_num in iterator:

        total_loss, total_correct = (0, 0)
        local_samples, local_loss, local_correct = (0, 0, 0)
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):

            # forward
            image_batch, label_batch = sampled_batch[:2]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            if mixup_fn is not None:
                image_batch, label_batch = mixup_fn(image_batch, label_batch)
            assert 0 <= label_batch.max() <= args.num_classes, \
                "label error max{} min{}.".format(label_batch.max(), label_batch.min())
            if args.channels_last:
                image_batch = image_batch.contiguous(memory_format=torch.channels_last)
            outputs = model(image_batch)

            # backward
            optimizer.zero_grad()
            loss = train_loss_fn(outputs, label_batch)
            if math.isnan(loss.item()):
                raise Exception('Loss becomes Nan.')
            loss.backward()

            # update weights
            optimizer.step()
            if args.self_scheduler:
                lr_ = lr * (1.0 - iter_num / max_iteration) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            _, predicted = torch.max(outputs.data, 1)
            local_samples += label_batch.size(0)
            local_loss += loss.item()
            local_correct += (predicted == label_batch).squeeze().cpu().sum().numpy()

            # writer.add_scalar('info/lr', lr_, iter_num)
            iter_num = iter_num + 1
            if (i_batch + 1) % log_interval == 0:
                logging.info("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}.".format(
                    epoch_num + 1, max_epoch, i_batch + 1, len(trainloader),
                    local_loss / log_interval, local_correct / local_samples))
                total_loss += local_loss
                total_correct += local_correct
                local_samples, local_loss, local_correct = (0, 0, 0)

        writer.add_scalar('info/loss', total_loss / len(trainloader), epoch_num)
        writer.add_scalar('info/accuracy', total_correct / len(train_data), epoch_num)

        if not args.self_scheduler:
            scheduler.step(epoch_num + 1, local_loss)

        # validate the model
        if (epoch_num + 1) % val_interval == 0:
            total_val, correct_val, loss_val = (0, 0, 0)
            model.eval()
            with torch.no_grad():
                for j_batch, sampled_batch in enumerate(validloader):
                    image_batch, label_batch = sampled_batch[:2]
                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                    assert 0 <= label_batch.max() <= args.num_classes, \
                        "label error max{} min{}.".format(label_batch.max(), label_batch.min())
                    if args.channels_last:
                        image_batch = image_batch.contiguous(memory_format=torch.channels_last)

                    bs, ncrops, c, h, w = image_batch.size()
                    outputs = model(image_batch.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = validate_loss_fn(outputs_avg, label_batch)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += label_batch.size(0)
                    correct_val += (predicted == label_batch).squeeze().cpu().sum().numpy()
                    loss_val += loss.item()

                loss_val_mean = loss_val / len(validloader)
                logging.info("Valid:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.2%}.".format(
                    epoch_num + 1, max_epoch, j_batch + 1, len(validloader), loss_val_mean, correct_val / total_val))
                writer.add_scalar('info/valid_acc', correct_val / total_val, (epoch_num + 1) // val_interval)

        if epoch_num + 1 >= max_epoch // 2 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, str(args.version) + '_ep' + str(epoch_num + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}.".format(save_mode_path))

        if epoch_num + 1 >= max_epoch:
            best_mode_path = os.path.join(snapshot_path, '{}_V{}.pth'.format(args.dataset, args.model_index))
            torch.save(model.state_dict(), best_mode_path)

        if (epoch_num + 1) % check_interval == 0 or epoch_num + 1 >= max_epoch:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          'lr_scheduler': scheduler.state_dict() if not args.self_scheduler else 0,
                          "epoch": epoch_num + 1,
                          "iter_num": iter_num}
            torch.save(checkpoint, checkpoint_path)
            logging.info("save checkpoint to {}.".format(checkpoint_path))
            if epoch_num + 1 >= max_epoch:
                break

    writer.close()
    return "Training Finished!"


def segmentation_trainer(args, snapshot_path):
    checkpoint_path = snapshot_path + '/checkpoint.pth'
    has_checkpoint = False
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        has_checkpoint = True
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    log('\nSnapshot Path: {}.'.format(snapshot_path), has_checkpoint)
    # ================================= step 0/5 Config =================================
    log('{0:*^98}'.format("  Args  "), has_checkpoint)
    log(str(args), has_checkpoint)
    max_epoch = args.epochs
    lr = args.lr
    batch_size = args.batch_size * args.n_gpu
    num_workers = args.num_workers
    momentum = args.momentum
    train_dir = args.data_dir + '/train'
    if not os.path.exists(train_dir):
        logging.error('Training folder does not exist at: {}.'.format(train_dir))
        exit(1)
    train_transform = args.transform[args.mode]
    valid_transform = args.transform['valid']
    val_interval = args.val_interval
    log_interval = args.log_interval
    check_interval = args.check_interval
    save_interval = args.save_interval
    device = args.device
    writer = SummaryWriter(snapshot_path + '/log')

    # ================================== step 1/5 Model =================================
    log('\n{0:*^98}'.format("  " + args.model_type + "  "), has_checkpoint)
    model = get_model(args)
    if has_checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    # ================================== step 2/5 Data ==================================
    # Build an instance of Dataset
    log('\n{0:*^98}'.format("  " + args.dataset + "  "), has_checkpoint)
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense.'
        num_aug_splits = args.aug_splits
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # Build an instance of Dataset
    train_data = getdata(args.dataset, data_dir=train_dir, list_dir=args.list_dir, debug=args.debug,
                         debug_ratio=args.debug_ratio, mode=args.mode, split_n=args.split_n,
                         image_size=args.img_size, self_transform=args.self_transform,
                         norm_transform=args.norm_transform, transform=train_transform, seed=args.seed)
    valid_data = getdata(args.dataset, data_dir=train_dir, list_dir=args.list_dir, debug=args.debug,
                         debug_ratio=args.debug_ratio, mode="valid", split_n=args.split_n,
                         image_size=args.img_size, self_transform=args.self_transform,
                         norm_transform=args.norm_transform, transform=valid_transform, seed=args.seed)
    log("The length of train set and val set is: {} and {}.".format(len(train_data), len(valid_data)), has_checkpoint)

    # Build dataloader
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=args.pin_mem)
    validloader = DataLoader(valid_data, batch_size=batch_size, num_workers=0, pin_memory=args.pin_mem)

    # ================================== step 3/5 Loss ==================================
    log('\n{0:*^98}'.format("  Loss  "), has_checkpoint)
    dice_loss_weight = args.dice_loss_weight
    dice_loss = DiceLoss(args.num_classes)
    dice_loss = dice_loss.to(device)
    if args.focal_loss:
        train_loss_fn = FocalLoss(args.num_classes, alpha=args.focal_alpha, gamma=args.focal_gamma)
        train_loss_fn = train_loss_fn.to(device)
        log("Use FocalLoss and DiceLoss for training and validation.", has_checkpoint)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device)
        log("Use CrossEntropyLoss and DiceLoss for training and validation.", has_checkpoint)

    # ================================= step 4/5 Optimizer ===============================
    if args.freeze:
        fc_params_id = list(map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': lr * 0.1},  # 0
            {'params': model.classifier.parameters(), 'lr': lr}], momentum=momentum)
    elif args.self_optimizer:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=args.weight_decay)
    else:
        optimizer = create_optimizer(args, model)

    if args.self_scheduler:
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epochs, gamma=args.gamma)
        pass
    else:
        scheduler, num_epochs = create_scheduler(args, optimizer)

    if has_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if not args.self_scheduler:
            scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # ================================= step 5/5 Train ===================================
    log('\n{0:*^98}'.format("  Train  "), has_checkpoint)
    iter_num = checkpoint['iter_num'] if has_checkpoint else 0
    iter_num_val = checkpoint['iter_num_val'] if has_checkpoint else 0
    iterator = range(checkpoint['epoch'] if has_checkpoint else 0, max_epoch)
    max_iteration = max_epoch * len(trainloader)
    write_interval = args.write_interval = max(1, len(trainloader) // 4)
    val_write_interval = max(1, write_interval // 10)
    log("{} iterations per epoch, {} max iterations, use device:{}.\n".
        format(len(trainloader), max_iteration, device), has_checkpoint)
    for epoch_num in iterator:

        total_loss_part, total_loss_dice, total_loss, total_dice = (0, 0, 0, 0)
        local_loss_part, local_loss_dice, local_loss, local_dice = (0, 0, 0, 0)
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):

            # forward
            image_batch, label_batch = sampled_batch[:2]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            assert 0 <= label_batch.max() <= args.num_classes, \
                "label error max{} min{}.".format(label_batch.max(), label_batch.min())
            if args.channels_last:
                image_batch = image_batch.contiguous(memory_format=torch.channels_last)
            outputs = model(image_batch)

            # backward
            optimizer.zero_grad()
            loss_part = train_loss_fn(outputs, label_batch.long())
            loss_dice, class_wise_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss_dice, _ = dice_loss(outputs, label_batch, softmax=True)
            # mean_dice = calculate_segmentation_metric(outputs, label_batch, args.num_classes)[0]
            loss = (1 - dice_loss_weight) * loss_part + dice_loss_weight * loss_dice
            if math.isnan(loss.item()):
                raise Exception('Loss becomes Nan.')
            loss.backward()

            # update weights
            optimizer.step()
            if args.self_scheduler:
                lr_ = lr * (1.0 - iter_num / max_iteration) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            local_loss_part += loss_part.item()
            local_loss_dice += loss_dice.item()
            local_loss += loss.item()
            local_dice += np.mean(class_wise_dice[1:])
            # local_dice += mean_dice

            iter_num = iter_num + 1
            # writer.add_scalar('info/lr', lr_, iter_num)
            if (i_batch + 1) % log_interval == 0:
                logging.info("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] {}:{:.4f} Loss_dice:{:.4f} "
                             "Loss:{:.4f} Dice:{:.4f}.".format(epoch_num + 1, max_epoch, i_batch + 1, len(trainloader),
                                                               'Loss_focal' if args.focal_loss else 'Loss_ce',
                                                               local_loss_part / log_interval,
                                                               local_loss_dice / log_interval,
                                                               local_loss / log_interval, local_dice / log_interval))
                total_loss_part += local_loss_part
                total_loss_dice += local_loss_dice
                total_loss += local_loss
                total_dice += local_dice
                local_loss_part, local_loss_dice, local_loss, local_dice = (0, 0, 0, 0)

            if iter_num % write_interval == 0:
                step = iter_num // write_interval
                image = image_batch[0, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, step)
                outputs = torch.argmax(torch.softmax(outputs[0, ...], dim=0), dim=0, keepdim=True)
                writer.add_image('train/Prediction', outputs / args.num_classes, step)
                labs = label_batch[0, ...].unsqueeze(0)
                writer.add_image('train/GroundTruth', labs / args.num_classes, step)

        writer.add_scalar('info/{}'.format('loss_focal' if args.focal_loss else 'loss_ce'),
                          total_loss_part / len(trainloader), epoch_num)
        writer.add_scalar('info/loss_dice', total_loss_dice / len(trainloader), epoch_num)
        writer.add_scalar('info/loss', total_loss / len(trainloader), epoch_num)
        writer.add_scalar('info/dice', total_dice / len(trainloader), epoch_num)

        if not args.self_scheduler:
            scheduler.step(epoch_num + 1, local_loss)

        # validate the model
        if (epoch_num + 1) % val_interval == 0:
            loss_part_val, loss_dice_val, loss_val, dice_val = (0, 0, 0, 0)
            model.eval()
            with torch.no_grad():
                for j_batch, sampled_batch in enumerate(validloader):
                    image_batch, label_batch = sampled_batch[:2]
                    image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                    assert 0 <= label_batch.max() <= args.num_classes, \
                        "label error max{} min{}.".format(label_batch.max(), label_batch.min())
                    if args.channels_last:
                        image_batch = image_batch.contiguous(memory_format=torch.channels_last)

                    outputs = model(image_batch)

                    loss_part = train_loss_fn(outputs, label_batch.long())
                    loss_dice, class_wise_dice = dice_loss(outputs, label_batch, softmax=True)
                    # loss_dice, _ = dice_loss(outputs, label_batch, softmax=True)
                    # mean_dice = calculate_segmentation_metric(outputs, label_batch, args.num_classes)[0]
                    loss = (1 - dice_loss_weight) * loss_part + dice_loss_weight * loss_dice

                    loss_part_val += loss_part.item()
                    loss_dice_val += loss_dice.item()
                    loss_val += loss.item()
                    dice_val += np.mean(class_wise_dice[1:])
                    # dice_val += mean_dice

                    iter_num_val = iter_num_val + 1
                    if iter_num_val % val_write_interval == 0:
                        step = iter_num_val // val_write_interval
                        image = image_batch[0, :, :, :]
                        image = (image - image.min()) / (image.max() - image.min())
                        writer.add_image('valid/Image', image, step)
                        outputs = torch.argmax(torch.softmax(outputs[0, ...], dim=0), dim=0, keepdim=True)
                        writer.add_image('valid/Prediction', outputs / args.num_classes, step)
                        labs = label_batch[0, ...].unsqueeze(0)
                        writer.add_image('valid/GroundTruth', labs / args.num_classes, step)

                loss_part_val_mean = loss_part_val / len(validloader)
                loss_dice_val_mean = loss_dice_val / len(validloader)
                loss_val_mean = loss_val / len(validloader)
                dice_val_mean = dice_val / len(validloader)
                logging.info("Valid:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] {}:{:.4f} Loss_dice:{:.4f} "
                             "Loss:{:.4f} Dice:{:.4f}.".format(epoch_num + 1, max_epoch,
                                                               len(validloader), len(validloader),
                                                               'Loss_focal' if args.focal_loss else 'Loss_ce',
                                                               loss_part_val_mean, loss_dice_val_mean,
                                                               loss_val_mean, dice_val_mean))
                writer.add_scalar('info/valid_dice', dice_val_mean, (epoch_num + 1) // val_interval)

        if epoch_num + 1 >= max_epoch // 2 and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, str(args.version) + '_ep' + str(epoch_num + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}.".format(save_mode_path))

        if epoch_num + 1 >= max_epoch:
            best_mode_path = os.path.join(snapshot_path, '{}_V{}.pth'.format(args.dataset, args.model_index))
            torch.save(model.state_dict(), best_mode_path)

        if (epoch_num + 1) % check_interval == 0 or epoch_num + 1 >= max_epoch:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          'lr_scheduler': scheduler.state_dict() if not args.self_scheduler else 0,
                          "epoch": epoch_num + 1,
                          "iter_num": iter_num,
                          "iter_num_val": iter_num_val}
            torch.save(checkpoint, checkpoint_path)
            logging.info("save checkpoint to {}.".format(checkpoint_path))
            if epoch_num + 1 >= max_epoch:
                break

    writer.close()
    logging.info("Training Finished!")
    return True
