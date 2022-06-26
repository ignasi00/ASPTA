
import torch
import torch.nn as nn

from .utils import conv3x3_bn_relu


def _conv3x3_bn_relu_leaky(in_planes, out_planes, stride=1, padding=1):
    # stride=1 & padding=1 with kernel=3 means HxW is mantained
    leaky = 0.1 if out_planes <= 64 else 0
    return conv3x3_bn_relu(in_planes, out_planes, stride=stride, padding=padding, leaky=leaky)

class SSH_Context(nn.Module):
    # Small context around each point as a (by default) 5x5 and a 7x7 windows with a shared 3x3 "seed".

    def __init__(self, in_planes, out_planes, conv_module=None):
        super(SSH_Context, self).__init__()

        conv_module = conv_module or _conv3x3_bn_relu_leaky

        hidden_planes = out_planes // 2
        oddity = out_planes % 2

        self.conv1 = conv_module(in_planes, hidden_planes)
        self.conv2a = conv_module(hidden_planes, hidden_planes + oddity)
        self.conv2b = nn.Sequential(conv_module(hidden_planes, hidden_planes), conv_module(hidden_planes, hidden_planes))
    
    def forward(self, x):
        x = self.conv1(x)
        x_a = self.conv2a(x)
        x_b = self.conv2b(x)
        x = torch.cat([x_a, x_b], dim=1)
        return x

class SSH_Residual_Context(nn.Module):
    # Point features enchanced with contextual features.

    def __init__(self, in_planes, out_planes, conv_module=None, context_module=None):
        super(SSH_Residual_Context, self).__init__()

        conv_module = conv_module or _conv3x3_bn_relu_leaky
        context_module = context_module or SSH_Context

        hidden_planes = out_planes // 2
        oddity = out_planes % 2

        self.conv = conv_module(in_planes, hidden_planes + oddity)
        self.context = context_module(in_planes, hidden_planes)

    def forward(self, x):
        x_a = self.conv(x)
        x_b = self.context(x)
        x = torch.cat([x_a, x_b], dim=1)
        return x

class SSH(nn.Module):

    def __init__(self, *args, **kargs):
        super(SSH, self).__init__()

        raise NotImplementedError("I did not need it yet")

