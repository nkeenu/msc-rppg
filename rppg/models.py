import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from .modules import *


class RegressionResNetPretrained(nn.Module):

    def __init__(self):
        super(RegressionResNetPretrained, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove last classification layer
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Freeze all layers except last classification layer
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels=32, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, self.in_channels, layers[0])
        self.layer2 = self._make_layer(block, self.in_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_channels * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_channels * 2, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, num_classes)
        
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def regression_resnet18(in_channels=32):
    return ResNet(ResBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=1)

def regression_resnet18_cbam(in_channels=32):
    return ResNet(ResBlockCBAM, [2, 2, 2, 2], in_channels=in_channels, num_classes=1)


class rPPGModelConv(nn.Module):
    """
    rPPG model using 2+1D ResNet encoder and transposed convolutional decoder.
    :param block: Residual block class
    :param layers: List of number of residual blocks in each layer
    :param in_channels: Number of input channels
    :param clip_length: Number of frames in each clip
    """

    def __init__(self, block, layers, in_channels=32, clip_length=64):
        super(rPPGModelConv, self).__init__()

        self.in_channels = in_channels
        self.clip_length = clip_length

        # 2+1D ResNet encoder layers
        self.conv1 = Conv2plus1d(3, in_channels, kernel_size=[3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        self.layer1 = self._make_layer(block, self.in_channels, layers[0])
        self.layer2 = self._make_layer(block, self.in_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_channels * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_channels, layers[3], stride=2)  # Not doubling channels due to computational constraints

        # Global average pooling on spatial dimensions
        self.spatial_avgpool = nn.AdaptiveAvgPool3d([self.clip_length, 1, 1])

        # Transposed convolutional decoder layers
        self.decode_layer1 = nn.Sequential(
            nn.ConvTranspose3d(self.in_channels, self.in_channels, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(self.in_channels),
            nn.ELU(inplace=True)
        )
        self.decode_layer2 = nn.Sequential(
            nn.ConvTranspose3d(self.in_channels, self.in_channels, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(self.in_channels),
            nn.ELU(inplace=True)
        )
        self.decode_layer3 = nn.Sequential(
            nn.ConvTranspose3d(self.in_channels, self.in_channels, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(self.in_channels),
            nn.ELU(inplace=True)
        )

        self.conv_out = nn.Conv3d(self.in_channels, 1, kernel_size=1, stride=1, padding=0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        self.clip_length //= stride
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):  # (3, clip_length, 128, 128)
        # Encoder forward pass
        x = self.conv1(x)  # (32, clip_length, 64, 64)

        x = self.layer1(x)  # (32, clip_length, 64, 64)
        x = self.layer2(x)  # (64, clip_length / 2, 32, 32)
        x = self.layer3(x)  # (128, clip_length / 4, 16, 16)
        x = self.layer4(x)  # (256, clip_length / 8, 8, 8)

        x = self.spatial_avgpool(x)  # (256, clip_length / 8, 1, 1)

        # Decoder forward pass
        x = self.decode_layer1(x)  # (256, clip_length / 4, 1, 1)
        x = self.decode_layer2(x)  # (256, clip_length / 2, 1, 1)
        x = self.decode_layer3(x)  # (256, clip_length, 1, 1)

        x = self.conv_out(x)  # (1, clip_length, 1, 1)
        rppg = x.squeeze(1, 3, 4)  # (clip_length)

        return rppg


def rppg_model_conv(in_channels=32, clip_length=64):
    return rPPGModelConv(ResBlock2plus1d, [2, 2, 2, 2], in_channels=in_channels, clip_length=clip_length)

def rppg_model_conv_cbam(in_channels=32, clip_length=64):
    return rPPGModelConv(ResBlock2plus1dCBAM, [2, 2, 2, 2], in_channels=in_channels, clip_length=clip_length)

def rppg_model_conv_cbam3d(in_channels=32, clip_length=64):
    return rPPGModelConv(ResBlock2plus1dCBAM3D, [2, 2, 2, 2], in_channels=in_channels, clip_length=clip_length)


class rPPGModelLSTM(nn.Module):
    """
    rPPG model using 2+1D ResNet encoder and LSTM decoder.
    :param block: Residual block class
    :param layers: List of number of residual blocks in each layer
    :param in_channels: Number of input channels
    :param clip_length: Number of frames in each clip
    :param hidden_size: Number of hidden units in LSTM
    """

    def __init__(self, block, layers, in_channels=32, clip_length=64, hidden_size=128):
        super(rPPGModelLSTM, self).__init__()

        self.in_channels = in_channels
        self.clip_length = clip_length

        # 2+1D ResNet encoder layers
        self.conv1 = Conv2plus1d(3, in_channels, kernel_size=[3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        
        self.layer1 = self._make_layer(block, self.in_channels, layers[0])
        self.layer2 = self._make_layer(block, self.in_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_channels * 2, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_channels, layers[3], stride=2)  # Not doubling channels due to computational constraints

        # Global average pooling on spatial dimensions and upsample temporal dimensions
        self.spatial_pool = nn.AdaptiveAvgPool3d([self.clip_length, 1, 1])

        # BiLSTM decoder layers
        self.temporal_upsample = nn.Upsample(size=(clip_length, 1, 1), mode='nearest')
        self.lstm = nn.LSTM(self.in_channels, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.in_channels * 2, 1)  # Multiply by 2 due to bidirectional LST

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        self.clip_length //= stride
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):  # (3, clip_length, 128, 128)
        # Encoder forward pass
        x = self.conv1(x)  # (32, clip_length, 64, 64)

        x = self.layer1(x)  # (32, clip_length, 64, 64)
        x = self.layer2(x)  # (64, clip_length / 2, 32, 32)
        x = self.layer3(x)  # (128, clip_length / 4, 16, 16)
        x = self.layer4(x)  # (256, clip_length / 8, 8, 8)

        x = self.spatial_pool(x)  # (256, clip_length / 8, 1, 1)
        x = self.temporal_upsample(x)  # (256, clip_length, 1, 1)
        x = x.squeeze(3, 4).permute(0, 2, 1)  # (clip_length, 256)

        # Decoder forward pass
        lstm_out, _ = self.lstm(x)  # (clip_length, 512)

        x = self.fc(lstm_out)  # (clip_length, 1)
        rppg = x.squeeze(2)  # (clip_length)

        return rppg


def rppg_model_lstm(in_channels=32, clip_length=64, hidden_size=128):
    return rPPGModelLSTM(ResBlock2plus1d, [2, 2, 2, 2], in_channels=in_channels, clip_length=clip_length, hidden_size=hidden_size)

def rppg_model_lstm_cbam(in_channels=32, clip_length=64, hidden_size=128):
    return rPPGModelLSTM(ResBlock2plus1dCBAM, [2, 2, 2, 2], in_channels=in_channels, clip_length=clip_length, hidden_size=hidden_size)

def rppg_model_lstm_cbam3d(in_channels=32, clip_length=64, hidden_size=128):
    return rPPGModelLSTM(ResBlock2plus1dCBAM3D, [2, 2, 2, 2], in_channels=in_channels, clip_length=clip_length, hidden_size=hidden_size)