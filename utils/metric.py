import torch
import numpy as np
from medpy import metric


def calculate_metric_percase(pred, mask):
    pred[pred > 0] = 1
    mask[mask > 0] = 1
    if pred.sum() > 0 and mask.sum() > 0:
        dice = metric.binary.dc(pred, mask)
        hd95 = metric.binary.hd95(pred, mask)
        return dice, hd95
    elif pred.sum() > 0 and mask.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def calculate_segmentation_metric(outputs, label, classes):
    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(outputs == i, label == i))

    return np.mean(metric_list, axis=0)
