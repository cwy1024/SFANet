# 完全体
import os
import time
import math
import torch
import joblib
import random
import warnings
import argparse
import numpy as np
import torchvision
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import sklearn.externals
import torch.optim as optim
# from dataset import Dataset
from datetime import datetime
from skimage.io import imread
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms


def SeedSed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SeedSed(seed=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


# MADConv: Multi-scale Asymmetric Dilated Convolution​
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthwiseSeparableConv, self).__init__()
        # 多核逐通道卷积
        self.depth_conv1_3_1 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(1, 3),
                                         stride=1,
                                         padding=(0, 1),
                                         groups=in_channel)
        self.depth_conv3_1_1 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(3, 1),
                                         stride=1,
                                         padding=(1, 0),
                                         groups=in_channel)
        self.depth_conv1_3_4 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(1, 3),
                                         dilation=(1, 4),
                                         stride=1,
                                         padding=(0, 4),
                                         groups=in_channel)
        self.depth_conv3_1_4 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(3, 1),
                                         dilation=(4, 1),
                                         stride=1,
                                         padding=(4, 0),
                                         groups=in_channel)
        self.depth_conv1_3_5 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(1, 3),
                                         dilation=(1, 5),
                                         stride=1,
                                         padding=(0, 5),
                                         groups=in_channel)
        self.depth_conv3_1_5 = nn.Conv2d(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=(3, 1),
                                         dilation=(5, 1),
                                         stride=1,
                                         padding=(5, 0),
                                         groups=in_channel)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out1 = self.depth_conv3_1_1(self.depth_conv1_3_1(input))
        out2 = self.depth_conv3_1_4(self.depth_conv1_3_4(input))
        out3 = self.depth_conv3_1_5(self.depth_conv1_3_5(input))
        out = out1 + out2 + out3
        out = self.point_conv(out)
        return out


# Statistical Multi-feature Adaptive spatial Recalibration Attention(SASA)
class SASA(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels):
        super(SASA, self).__init__()

        self.weight = nn.Parameter(torch.ones(4))

        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()  # 增强非线性
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_std = torch.std(x, dim=1, keepdim=True)
        x_energy = torch.mean(x ** 2, dim=1, keepdim=True)
        x_cat = torch.cat([x_avg, x_max, x_std, x_energy], dim=1)

        weights = F.softmax(self.weight, dim=0)  # 归一化权重
        x_weighted = (x_cat * weights.view(1, 4, 1, 1)).sum(dim=1, keepdim=True)  # 加权融合

        x_attention = torch.sigmoid(self.conv(x_weighted))

        return x * x_attention


# Statistical Multi-feature Adaptive Channel Recalibration Attention(SACA)
class saca(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()

        # 多特征权重学习
        self.weights = nn.Parameter(torch.ones(4))  # 均值/最大值/标准差/能量
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        b, c, h, w = x.shape

        # 多维度通道统计特征 --------------------------------
        # 均值池化
        x_avg = x.mean(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # 最大值池化
        x_max = x.amax(dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # 标准差池化
        x_std = torch.std(x, dim=[2, 3], keepdim=True)  # [B,C,1,1]
        # 能量池化 (L2 Norm)
        x_energy = (x ** 2).mean(dim=[2, 3], keepdim=True)  # [B,C,1,1]

        # 自适应特征融合 --------------------------------
        weights = F.softmax(self.weights, dim=0)  # 归一化权重
        fused = (weights[0] * x_avg + weights[1] * x_max +
                 weights[2] * x_std + weights[3] * x_energy)  # [B,C,1,1]

        x_c = fused.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        x_c = self.conv(x_c)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        x_c = torch.sigmoid(x_c)  # 应用Sigmoid函数激活，得到最终的注意力权重
        x_c = x_c.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * x_c.expand_as(x)


class Downsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, self_is=True):
        super(Downsample_block, self).__init__()
        self.is_down = self_is

        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        self.sa = SASA(out_channels)
        self.ca = saca(out_channels)

    def forward(self, x):

        residual = self.conv1x1(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.sa(x)
        x = self.ca(x)
        y = F.relu(self.bn2(self.conv2(x)))
        y = y + residual
        if self.is_down:
            x = F.max_pool2d(y, 2, stride=2)
            return x, y
        else:
            return y


class Light_Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Light_Up, self).__init__()
        # 多核逐通道卷积
        self.depth_conv1 = DepthwiseSeparableConv((in_channels // 4) + 1, out_channels // 4)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.depth_conv2 = DepthwiseSeparableConv((in_channels // 4) + 1, out_channels // 4)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.depth_conv3 = DepthwiseSeparableConv((in_channels // 4) + 1, out_channels // 4)
        self.bn3 = nn.BatchNorm2d(out_channels // 4)
        self.depth_conv4 = DepthwiseSeparableConv((in_channels // 4) + 1, out_channels // 4)
        self.bn4 = nn.BatchNorm2d(out_channels // 4)

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # reshape
        # b, c, h, w =======>  b, g, c_per, h, w
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x

    def forward(self, x, mask):
        chunks = torch.chunk(x, chunks=4, dim=1)
        group1 = torch.cat([chunks[0], mask], dim=1)
        group2 = torch.cat([chunks[1], mask], dim=1)
        group3 = torch.cat([chunks[2], mask], dim=1)
        group4 = torch.cat([chunks[3], mask], dim=1)

        x1 = F.relu(self.bn1(self.depth_conv1(group1)))
        x2 = F.relu(self.bn2(self.depth_conv2(group2)))
        x3 = F.relu(self.bn3(self.depth_conv3(group3)))
        x4 = F.relu(self.bn4(self.depth_conv4(group4)))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.channel_shuffle(x, 4)

        return x


# Group mask enhanced upsampling block
class Upsample_block(nn.Module):
    SeedSed(seed=10)

    def __init__(self, in_channels, out_channels, i=0):
        super(Upsample_block, self).__init__()

        self.conv1 = Light_Up(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_supervised = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, y):
        B, C, H, W = y.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = x + y

        # Supervised
        mask = self.conv_supervised(x)
        gt_pred = F.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)

        residual = F.relu(self.conv1x1(x))
        x = self.conv1(x, mask)
        x = x + residual
        return x, gt_pred


class sfanet(nn.Module):
    SeedSed(seed=10)

    def __init__(self):
        in_chan = 3
        out_chan = 1
        super(sfanet, self).__init__()

        # in=12
        # self.down1 = Downsample_block(in_chan, 12)
        # self.down2 = Downsample_block(12, 20)
        # self.down3 = Downsample_block(20, 28)
        # self.down4 = Downsample_block(28, 36)
        # self.bottle1 = Downsample_block(36, 44, self_is=False)
        # self.bottle2 = Downsample_block(44, 36, self_is=False)
        # self.up4 = Upsample_block(36, 28)
        # self.up3 = Upsample_block(28, 20)
        # self.up2 = Upsample_block(20, 12)
        # self.up1 = Upsample_block(12, 12)
        # self.outconv = nn.Conv2d(12, out_chan, 1)

        # in=8
        # self.down1 = Downsample_block(in_chan, 8)
        # self.down2 = Downsample_block(8, 16)
        # self.down3 = Downsample_block(16, 24)
        # self.down4 = Downsample_block(24, 32)
        # self.bottle1 = Downsample_block(32, 40, self_is=False)
        # self.bottle2 = Downsample_block(40, 32, self_is=False)
        # self.up4 = Upsample_block(32, 24)
        # self.up3 = Upsample_block(24, 16)
        # self.up2 = Upsample_block(16, 8)
        # self.up1 = Upsample_block(8, 8)
        # self.outconv = nn.Conv2d(8, out_chan, 1)

        self.down1 = Downsample_block(in_chan, 16)
        self.down2 = Downsample_block(16, 24)
        self.down3 = Downsample_block(24, 32)
        self.down4 = Downsample_block(32, 40)
        self.bottle1 = Downsample_block(40, 48, self_is=False)
        self.bottle2 = Downsample_block(48, 40, self_is=False)
        self.up4 = Upsample_block(40, 32)
        self.up3 = Upsample_block(32, 24)
        self.up2 = Upsample_block(24, 16)
        self.up1 = Upsample_block(16, 16)
        self.outconv = nn.Conv2d(16, out_chan, 1)

    def forward(self, x):
        x1, y1 = self.down1(x)  # y1=12x256x256
        x2, y2 = self.down2(x1)  # y2=24x128x128
        x3, y3 = self.down3(x2)  # y3=36x64x64
        x4, y4 = self.down4(x3)  # y4=48x32x32
        x5 = self.bottle1(x4)  # x5=60x16x16
        x5 = self.bottle2(x5)  # x5=48x16x16
        x6, gt4 = self.up4(x5, y4)  # x6=36x32x32
        x7, gt3 = self.up3(x6, y3)  # x7=24x64x64
        x8, gt2 = self.up2(x7, y2)  # x8=12x128x128
        x9, gt1 = self.up1(x8, y1)  # x9=12x256x256
        out = self.outconv(x9)

        return (torch.sigmoid(gt1), torch.sigmoid(gt2), torch.sigmoid(gt3), torch.sigmoid(gt4)), torch.sigmoid(out)


if __name__ == '__main__':
    SeedSed(seed=10)
    input = torch.randn((1, 3, 256, 256)).to(device)
    model = sfanet().to(device)
    out = model(input)
    print(out.shape)
    print(out)
