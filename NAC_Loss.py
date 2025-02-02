import torch
import torch.nn.functional as F
import torch.nn as nn

NUM_CLASS=2
class NAC_LossS(nn.Module):
    def __init__(self, device, NUM_CLASS=2, m=2):
        super().__init__()
        self.device = device
#        self.class_weight = torch.tensor(class_weight, device=device)
        self.m = m

    def RegionTerm(self, y_true, y_pred, kernel_size=7):
        dim = (1,2,3)
        yTrueOnehot = torch.zeros(y_true.size(0), NUM_CLASS, y_true.size(2), y_true.size(3), device=self.device)
        yTrueOnehot = torch.scatter(yTrueOnehot, 1, y_true, 1)[:,1:]
        y_pred = y_pred[:,1:]

        active = yTrueOnehot** self.m * (1 - y_pred)  + (1 - yTrueOnehot)** self.m * y_pred
        loss = torch.sum(active, dim = dim) / torch.sum(yTrueOnehot*yTrueOnehot + y_pred*y_pred - yTrueOnehot * y_pred +smooth, dim = dim)
        return torch.mean(loss)

    def GradientLoss(self, y_pred, penalty="l1"):
        dH = torch.abs(y_pred[..., 1:] - y_pred[..., :-1])
        dW = torch.abs(y_pred[:, :, 1:] - y_pred[:, :, :-1])
        if penalty == "l2":
            dH = dH * dH
            dW = dW * dW
        loss = torch.sum(dH) + torch.sum(dW)
        return loss

    def forward(self, y_true, y_pred, m=1e-4):
        self.m = m
        region = self.RegionTerm(y_true, y_pred)
        length = self.GradientLoss(y_pred)
        return region + self.m *(1/(192*256))* length
