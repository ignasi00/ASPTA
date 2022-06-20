
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv1x1_bn


class DeconvolutionModule(nn.Module):

    def __init__(self, *args, **kargs):
        super(DeconvolutionModule, self).__init__()

        raise NotImplementedError("I did not need it yet")

class DSSD_ProcessingModule(nn.Module):
    # Processing just before Classification and Regresion Heads

    def __init__(self, in_planes, out_planes, hidden_planes=None, conv_module=None, process_residual=False, relu=True):
        super(DSSD_ProcessingModule, self).__init__()

        conv_module = conv_module or conv1x1_bn
        hidden_planes = hidden_planes or out_planes // 4
        
        if relu:
            self.conv1a = nn.Sequential(conv_module(in_planes, hidden_planes), nn.ReLU(inplace=True))
            self.conv1b = nn.Sequential(conv_module(hidden_planes, hidden_planes), nn.ReLU(inplace=True))
        else:
            self.conv1a = conv_module(in_planes, hidden_planes)
            self.conv1b = conv_module(hidden_planes, hidden_planes)
        self.conv1c = conv_module(hidden_planes, out_planes)
        
        if (in_planes != out_planes) or (process_residual):
            self.conv2 = conv_module(in_planes, out_planes)
        else:
            self.conv2 = lambda x : x.clone()
    
    def forward(self, x):
        x_b = self.conv2(x)
        
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.conv1c(x)

        return x + x_b

class DSSD(nn.Module):

    def __init__(self, *args, **kargs):
        super(DSSD, self).__init__()

        raise NotImplementedError("I did not need it yet")

