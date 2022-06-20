
import torch
import torch.nn as nn

from .expressions.utils import conv1x1_bn_relu


class HH_ContextSensitivePrediction(nn.Module):

    def __init__(self, preprocessing_module, contextual_module, contextual_out_planes, out_planes, activation=None, conv_module=None):
        super(HH_ContextSensitivePrediction, self).__init__()

        self.preprocessing_module = preprocessing_module
        self.activation = activation or nn.ReLu(inplace=True)
        self.contextual_module = contextual_module
        conv_module = conv_module or conv1x1_bn_relu
        self.conv = conv_module(contextual_out_planes, out_planes)
    
    def forward(self, x):
        x  =self.preprocessing_module(x)
        x = self.activation(x)
        x = self.contextual_module(x)
        x = self.conv(x)
        return x

class HeadHunter(nn.Module):

    def __init__(self):
        super(HeadHunter, self).__init__()
