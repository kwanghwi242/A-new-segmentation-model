import torch
import torch.nn.functional as F
import torch.nn as nn

def dice(y_true, y_pred, smooth = 1e-4):
    y_pred = torch.argmax(y_pred, dim=1, keepdim = True)
    intersection = torch.sum(y_true * y_pred, dim=[1,2,3])
    cardinality  = torch.sum(y_true + y_pred , dim=[1,2,3])
    return torch.mean((2. * intersection + smooth) / (cardinality + smooth), dim=0)

def jaccard(y_true, y_pred, smooth = 1e-4):
    y_pred = torch.argmax(y_pred, dim=1, keepdim = True)
    intersection = torch.sum(y_true * y_pred, dim=[1,2,3])
    union = torch.sum(y_true + y_pred , dim=[1,2,3]) - intersection
    return torch.mean((intersection + smooth) / (union + smooth), dim=0)

def precision(y_true, y_pred, smooth=1e-4):
    y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
    TP = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    FP = torch.sum((1 - y_true) * y_pred, dim=[1, 2, 3])
    return torch.mean((TP + smooth) / (TP + FP + smooth), dim=0)

def recall(y_true, y_pred, smooth=1e-4):
    y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
    TP = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    FN = torch.sum(y_true * (1 - y_pred), dim=[1, 2, 3])
    return torch.mean((TP + smooth) / (TP + FN + smooth), dim=0)

