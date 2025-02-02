import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.ResMEVSS import ResMambaBlock
from Model.Selective_Kernel import *
from Model.CAD_Mamba import *
from Model.RABM import *
from Model.MCSA import *

class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class EncoderBlock_Mamba(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c,mixer_kernel=(3,3)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = mixer_kernel )
        self.pw= nn.Conv2d(in_c, out_c, kernel_size=1, padding = 'same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.resmamba = ResMambaBlock(in_c)
        self.down = nn.MaxPool2d((2,2))

    def forward(self, x):
        x = self.resmamba(x)
        skip=self.dw(x)
        x= self.act(self.bn(self.pw(skip)))
        x = self.down(x)

        return x, skip
    
class EncoderBlock_Axial(nn.Module):
      def __init__(self, in_c, out_c, mixer_kernel = (7,7)):
          super().__init__()
          self.dw = AxialDW(in_c, mixer_kernel = mixer_kernel )
          self.bn = nn.BatchNorm2d(in_c)
          self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
          self.down = nn.MaxPool2d((2,2))
          self.act = nn.ReLU()

      def forward(self, x):
          skip = self.bn(self.dw(x))
          skip = self.pw(skip)
          x = self.act(self.down(skip))

          return x
      
class EncoderBlock_Mamba2(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c,mixer_kernel=(3,3)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = mixer_kernel )
        self.pw= nn.Conv2d(in_c, out_c, kernel_size=1, padding = 'same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.resmamba = CADmamba(in_c//2)
        self.down = nn.MaxPool2d((2,2))

    def forward(self, x):
        skip=x
        x1, x2 = torch.chunk(x,2, dim=1)
        x1 = self.resmamba(x1)
        x2 = self.resmamba(x2)

        x = torch.cat([x1, x2], dim=1)
        skip=self.dw(x)
        x= self.act(self.bn(self.pw(skip)))
        x = self.down(x)

        return x, skip
    
class DecoderBlock_Mamba(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (3, 3))
        self.conv2d = nn.Conv2d(in_c, in_c, kernel_size=1, padding = 'same')
        self.bn2 = nn.BatchNorm2d(in_c)
        self.resmamba = ResMambaBlock(in_c)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.act(self.bn2(self.conv2d((self.dw(x)))))
        x1=x
        x1 = self.resmamba(x1)
        return x1
    
class DecoderBlock_Mamba2(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (3, 3))
        self.conv2d = nn.Conv2d(in_c, in_c, kernel_size=1, padding = 'same')
        self.bn2 = nn.BatchNorm2d(in_c)
        self.resmamba = CADmamba(in_c//2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn2(self.conv2d((self.dw(x)))))
        xs=x
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.resmamba(x1)
        x2 = self.resmamba(x2)
        x = torch.cat([x1, x2], dim=1)
        return x
    
class litemamba_bound(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsize = nn.Upsample(scale_factor = 2)
        self.pw_in = nn.Conv2d(3,16, kernel_size=1)
        self.sk_in = SKConv_7(16, M=2, G=16, r=4, stride=1 ,L=32)
        self.conv_end = nn.Conv2d(256, 128, kernel_size=1, padding=0, groups=1)
        self.pw1 = nn.Conv2d(512, 1, kernel_size=1, padding=0, groups=1)
        self.pw2 = nn.Conv2d(256, 1, kernel_size=1, padding=0, groups=1)
        '''En'''
        self.e1 = EncoderBlock_Mamba(16, 32)
        self.e2 = EncoderBlock_Mamba(32, 64)
        self.e3 = EncoderBlock_Mamba(64, 128)

        self.e11 = EncoderBlock_Axial(16, 32)
        self.e12 = EncoderBlock_Axial(32, 64)
        self.e13 = EncoderBlock_Axial(64, 128)

        self.e4 = EncoderBlock_Mamba2(128, 256)
        self.e5 = EncoderBlock_Mamba2(256, 512)

        '''Skip Connection'''

        self.c1=CSAG(16,32)
        self.c2=CSAG(32,64)
        self.c3=CSAG(64,128)
        self.c4=RABM(128,1)
        self.c5=RABM(256,1)
        """Bottle Neck"""
        self.b52= mcsa_module(512)
        """Decoder"""
        self.d5 = DecoderBlock_Mamba2( 256)
        self.d4 = DecoderBlock_Mamba2( 128)
        self.d3 = DecoderBlock_Mamba( 64)
        self.d2 = DecoderBlock_Mamba(32)
        self.d1 = DecoderBlock_Mamba(16)
        self.conv_out = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=1,stride=1,padding=0),
            nn.Softmax(dim=1)
            )
    def forward(self, x):
        """Encoder"""
        x = self.pw_in(x)
        x = self.sk_in(x)
        x1=x
        x, skip1 = self.e1(x)
        x1=self.e11(x1)
        x, skip2 = self.e2(x)
        x1=self.e12(x1)
        x, skip3 = self.e3(x)
        x1=self.e13(x1)
        x=torch.cat([x, x1], dim=1)
        x= self.conv_end(x)
        x, skip4 = self.e4(x)

        x, skip5 = self.e5(x)
        """BottleNeck"""
        x=self.b52(x)
        x=self.pw1(x)
        """Skip5+Decoder5"""
        x=self.upsize(x)
        xd1=x
        skip5 = self.c5(skip5,x)
        x = self.d5(skip5)
        x=self.pw2(x)
        x=xd1+x
        """Skip4+Decoder4"""
        x=self.upsize(x)
        skip4 = self.c4(skip4,x)
        x = self.d4(skip4)
        """Skip3+Decoder3"""
        x=self.upsize(x)
        skip3 = self.c3(skip3,x)
        x = self.d3(skip3)

        """Skip2+Decoder2"""
        x=self.upsize(x)
        skip2 = self.c2(skip2,x)
        x = self.d2(skip2)

        """Skip1+Decoder1"""
        x=self.upsize(x)
        skip1 = self.c1(skip1,x)
        x = self.d1(skip1)
        
        x = self.conv_out(x)
        return x