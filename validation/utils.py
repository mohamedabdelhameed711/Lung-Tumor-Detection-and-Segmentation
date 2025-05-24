import torch
import numpy as np

def iou_score(pred_mask, true_mask, threshold=0.5):
    pred_mask = pred_mask > threshold
    true_mask = true_mask > 0.5
    intersection = (pred_mask & true_mask).float().sum()
    union = (pred_mask | true_mask).float().sum()
    return (intersection / union).item() if union > 0 else 0

def dice_score(pred_mask, true_mask, threshold=0.5):
    pred_mask = pred_mask > threshold
    true_mask = true_mask > 0.5
    intersection = (pred_mask & true_mask).float().sum()
    return (2. * intersection / (pred_mask.sum() + true_mask.sum())).item()

def bbox_iou(box1, box2):
    """ box = [xmin, ymin, xmax, ymax] """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0
