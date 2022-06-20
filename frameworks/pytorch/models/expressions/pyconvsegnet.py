""" PyConv network for semantic segmentation, presented in:
    Duta et al. "Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"
    https://arxiv.org/pdf/2006.11538.pdf
"""

import torch
from torch import nn
import torch.nn.functional as F

from .utils import PyConv2d


def PyConv4(inplans, planes, *, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
    out_channels = [planes // 4] * 4
    pyconv_padding = [x // 2 for x in pyconv_kernels]

    return PyConv2d(inplans, out_channels=out_channels, pyconv_kernels=pyconv_kernels, pyconv_padding=pyconv_padding, stride=stride, pyconv_groups=pyconv_groups)

class GlobalPyConvBlock(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(GlobalPyConvBlock, self).__init__()
        self.features = nn.Sequential(
                nn.AdaptiveAvgPool2d(bins),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True),
                PyConv4(reduction_dim, reduction_dim),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        x = F.interpolate(self.features(x), x_size[2:], mode='bilinear', align_corners=True)
        return x

class LocalPyConvBlock(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, reduction=4):
        super(LocalPyConvBlock, self).__init__()

        hidden_planes = inplanes // reduction

        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, hidden_planes, kernel_size=1, bias=False),
            BatchNorm(hidden_planes),
            nn.ReLU(inplace=True),
            PyConv4(hidden_planes, hidden_planes),
            BatchNorm(hidden_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.layers(x)


class MergeLocalGlobal(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, kernel_size=3, padding=1, groups=1):
        super(MergeLocalGlobal, self).__init__()

        #TODO: ¿What does groups do?
        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes,  kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x

class PyConvHead(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm, out_size_local_context=512, out_size_global_context=512, local_context_reduction=4, global_context_bins=9):
        super(PyConvHead, self).__init__()

        self.local_context = LocalPyConvBlock(inplanes, out_size_local_context, BatchNorm, local_context_reduction)
        self.global_context = GlobalPyConvBlock(inplanes, out_size_global_context, global_context_bins, BatchNorm)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_context, planes, BatchNorm)

    def forward(self, x):
        x = self.merge_context(self.local_context(x), self.global_context(x))
        return x

class PyConvSegNet(nn.Module):
    def __init__(self, num_classes, backbone, context_head_class=None, backbone_output_maps=2048, out_merge_all=256, dropout=0.1, zoom_factor=8, BatchNorm=nn.BatchNorm2d, aux=False):
        super(PyConvSegNet, self).__init__()
        
        self.zoom_factor = zoom_factor
        self.backbone = backbone

        context_head_class = context_head_class or PyConvHead
        self.pyconvhead = context_head_class(backbone_output_maps, out_merge_all, BatchNorm)

        if aux:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False), # 1024 because initial design, usually not.
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )
        else:
            self.aux = None

        self.cls = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_merge_all, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # B, C, H, W

        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.backbone(x)
        if self.aux is not None:
            aux = self.aux(x[-1]) # <- The Pyramid Feature Extractor I designed returns a list and allows to sort the outputs
        if not isinstance(x, torch.Tensor):
            x = x[0] # <- If Pydamid Feature Extractor is used (to allow aux), x is a list with first element the interest one
        
        x = self.pyconvhead(x)
        x = self.cls(x)

        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            
            if self.aux is not None:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                # If somebody wants aux as an output, he should use the Pyramide Feature Extractor
                # return x, aux # x.max(1)[1]
        
        return x #, None
