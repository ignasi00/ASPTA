
import torch
import torch.nn as nn

from .expressions.utils import conv1x1_bn_relu


####

class HH_ContextSensitivePrediction(nn.Module):

    def __init__(self, preprocessing_module, contextual_module, contextual_out_planes, out_planes, activation=None, conv_module=None):
        super(HH_ContextSensitivePrediction, self).__init__()

        self.preprocessing_module = preprocessing_module
        self.activation = activation or nn.ReLU(inplace=True)
        self.contextual_module = contextual_module
        conv_module = conv_module or conv1x1_bn_relu
        self.conv = conv_module(contextual_out_planes, out_planes)
    
    def forward(self, x):
        x  =self.preprocessing_module(x)
        x = self.activation(x)
        x = self.contextual_module(x)
        x = self.conv(x)
        return x

####

class HH_FPN_ContextualBackbone(nn.Module):
    
    def __init__(self, base_fpn_backbone, context_sensitive_modules):
        super(HH_FPN_ContextualBackbone, self).__init__()
        self.fpn_backbone = base_fpn_backbone
        self.context_modules = context_sensitive_modules
    
    def forward(self, image_list):
        features_list = self.fpn_backbone(torch.stack(image_list, dim=0))
        features_list = [context_module(feat) for context_module, feat in zip(self.context_modules, features_list)]
        return features_list

class HH_RPN(nn.Module):
    
    def __init__(self, base_rpn, backbone_out_channels=256, backbone=None):
        super(HH_RPN, self).__init__()
        self.backbone = backbone
        self.rpn_upscale_feat = nn.ConvTranspose2d(backbone_out_channels, backbone_out_channels, kernel_size=(8,8), stride=2, padding=3)
        self.rpn = base_rpn

    def forward(self, image_list, features_list=None):
        if features_list is None:
            assert self.backbone is not None
            features_list = self.backbone(torch.stack(image_list, dim=0))
        features_list = [self.rpn_upscale_feat(feat) for feat in features_list]
        return rpn(image_list, features_list)


class HH_RoIHead(nn.Module):
    
    def __init__(self, base_roi_head, backbone_out_channels=256):
        super(HH_RoIHead, self).__init__()
        self.roi_upscale_feat = nn.ConvTranspose2d(backbone_out_channels, backbone_out_channels, kernel_size=(8,8), stride=2, padding=3)
        self.roi_head = base_roi_head

    def forward(self, features_list, proposals, image_list):
        features_list = [self.roi_upscale_feat(feat) for feat in features_list]
        return roi_head(features_list, proposals, image_list)

####

# The HeadHunter detector is, in fact, a Faster R-CNN with a extended backbone (backbone followed by HH_ContextSensitivePrediction) and "rpn and roi_head" with a ConvTranspose2d at the begining of each one.

####

class HeadHunter_Tracker():
    pass
