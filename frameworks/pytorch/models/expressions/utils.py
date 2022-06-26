
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, 
                                                            groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1_bn_relu(in_planes, out_planes, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv3x3_bn_relu(in_planes, out_planes, stride=1, padding=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv1x1_bn(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_planes)
    )

def conv3x3_bn(in_planes, out_planes, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes)
    )

def conv1x1_relu(in_planes, out_planes, stride=1, leaky=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv3x3_relu(in_planes, out_planes, stride=1, padding=1, leaky=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

class PyConv2d(nn.Module):
    """ Mostly copied from pyconvsegnet code """
    def __init__(self, in_channels, out_channels, *, pyconv_kernels, pyconv_padding, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups) == len(pyconv_padding)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_padding[i], groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out, dim=1)

