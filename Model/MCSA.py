import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class AxialDW2(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)
        self.dw_f = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding='same', groups = dim, dilation = dilation)
    def forward(self, x):
        x = x + self.dw_f(self.dw_h(self.dw_w(x)))
        return x
    
class mcsa_module(torch.nn.Module):
    def __init__(self, channels , e_lambda = 1e-4):
        super(mcsa_module, self).__init__()
        self.dw1 = AxialDW2(channels, mixer_kernel = (3, 3), dilation = 1)
        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1)

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
    def forward(self, x):
        x1=self.pw1(self.dw1(x))

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x1 * self.activaton(y)