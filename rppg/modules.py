import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class ResBlock(nn.Module):
    """
    Residual block for ResNet.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


class ChannelAttention(nn.Module):
    """
    Channel attention module as described in the paper "CBAM: Convolutional Block Attention Module".
    :param in_channels: Number of input channels
    :param ratio: Ratio of number of input channels to number of channels in the intermediate mapping
    """
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module as described in the paper "CBAM: Convolutional Block Attention Module".
    :param kernel_size: Size of the convolving kernel
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResBlockCBAM(nn.Module):
    """
    Residual block for ResNet with CBAM.
    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param stride: Stride of the first convolutional layer
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x




class Conv2plus1d(nn.Module):
    """
    2+1D convolution as described in the paper "A Closer Look at Spatiotemporal Convolutions for Action Recognition".
    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param kernel_size: Size of the convolving kernel
    :param stride: Stride of the convolving kernel
    :param padding: Zero-padding added to both sides of the input
    :param bias: If True, adds a learnable bias to the output
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(Conv2plus1d, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class ResBlock2plus1d(nn.Module):
    """
    Residual block for ResNet with 2+1D convolutions.
    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param stride: Stride of the first convolutional layer
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock2plus1d, self).__init__()

        self.conv1 = Conv2plus1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2plus1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2plus1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x


class ChannelAttention2plus1d(nn.Module):
    """
    Channel attention module for 2+1D convolutions.
    :param in_channels: Number of input channels
    :param ratio: Ratio of number of input channels to number of channels in the intermediate mapping
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention2plus1d, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(nn.Conv3d(in_channels, in_channels // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention2plus1d(nn.Module):
    """
    Spatial attention module for 2+1D convolutions.
    :param kernel_size: Size of the convolving kernel
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention2plus1d, self).__init__()

        kernel_size = _triple(kernel_size)

        self.conv1 = nn.Conv3d(2, 1, [1, kernel_size[1], kernel_size[2]], padding=[0, kernel_size[1] // 2, kernel_size[2] // 2], bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=(1, 2), keepdim=True)
        max_out = torch.amax(x, dim=(1, 2), keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResBlock2plus1dCBAM(nn.Module):
    """
    Residual block for ResNet with 2+1D convolutions and CBAM.
    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param stride: Stride of the first convolutional layer
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock2plus1dCBAM, self).__init__()

        self.conv1 = Conv2plus1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2plus1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.ca = ChannelAttention2plus1d(out_channels)
        self.sa = SpatialAttention2plus1d()

        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2plus1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.ca(x) * x
        x = self.sa(x) * x

        x += residual
        x = self.relu(x)

        return x


class TemporalAttention2plus1d(nn.Module):
    """
    Temporal attention module for 2+1D convolutions.
    :param kernel_size: Size of the convolving kernel
    """

    def __init__(self, kernel_size=7):
        super(TemporalAttention2plus1d, self).__init__()

        kernel_size = _triple(kernel_size)

        self.conv1 = nn.Conv3d(2, 1, [kernel_size[0], 1, 1], padding=[kernel_size[0] // 2, 0, 0], bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = torch.mean(x, dim=(1, 3, 4), keepdim=True)
        max_out = torch.amax(x, dim=(1, 3, 4), keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResBlock2plus1dCBAM3D(nn.Module):
    """
    Residual block for ResNet with 2+1D convolutions and CBAM.
    :param in_channels: Number of input channels
    :param out_channels: Number of output channels
    :param stride: Stride of the first convolutional layer
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock2plus1dCBAM3D, self).__init__()

        self.conv1 = Conv2plus1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2plus1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.ca = ChannelAttention2plus1d(out_channels)
        self.sa = SpatialAttention2plus1d()
        self.ta = TemporalAttention2plus1d()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv2plus1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.ta(x) * x

        x += residual
        x = self.relu(x)

        return x