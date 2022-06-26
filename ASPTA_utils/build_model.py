
import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.misc import ConvNormActivation
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.boxes import batched_nms

from frameworks.pytorch.models.algorithms.faster_rcnn import RoIHead, FasterRCNN
from frameworks.pytorch.models.algorithms.region_proposal_network import RPNHead, Batches_RPN
from frameworks.pytorch.models.expressions.deconvolutional_single_shot_detector import DSSD_ProcessingModule
from frameworks.pytorch.models.expressions.feature_pyramid_network import FPN
from frameworks.pytorch.models.expressions.resnet import resnet50
from frameworks.pytorch.models.expressions.single_stage_headless import SSH_Residual_Context
from frameworks.pytorch.models.expressions.utils import conv1x1_relu, conv3x3_relu
from frameworks.pytorch.models.head_hunter import HH_ContextSensitivePrediction
from frameworks.pytorch.models.head_hunter import HH_FPN_ContextualBackbone, HH_RPN, HH_RoIHead
from frameworks.pytorch.models.utils.feature_extractor import FeatureExtractor

from .rpn_bbox_filter import build_batches_proposals_filter


def _conv_1X1_leaky(in_planes, out_planes, stride=1):
    leaky = 0.1 if out_planes <= 64 else 0
    return conv1x1_relu(in_planes, out_planes, stride=stride, leaky=leaky, bias=True)

def _conv_3X3_leaky(in_planes, out_planes, stride=1):
    leaky = 0.1 if out_planes <= 64 else 0
    return conv3x3_relu(in_planes, out_planes, stride=stride, leaky=leaky, bias=True)

def fpn_extra_block(x, y):
    return F.max_pool2d(x[-1], 1, 2, 0)


class Conv2dNormActivation(ConvNormActivation):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, norm_layer=nn.BatchNorm2d, activation_layer=torch.nn.ReLU, dilation=1, inplace=True, bias=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups, norm_layer, activation_layer, dilation, inplace, bias)


class ListMultiScaleRoIAlign(nn.Module): # TODO: Implement my version of MultiScaleRoIAlign
    def __init__(self, featmap_names, output_size, sampling_ratio):
        super().__init__()
        self.multiscale_RoI_align = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=output_size, sampling_ratio=sampling_ratio)

    def forward(self, features_list, proposals, image_list):
        features_dict = {i : x for i, x in enumerate(features_list)}
        image_shapes = [img.shape[1:] for img in image_list]
        return self.multiscale_RoI_align(features_dict, proposals, image_shapes)

class ListAnchorGenerator(nn.Module): # TODO: Implement my version of AnchorGenerator
    def __init__(self, scales, aspect_ratios):
        super().__init__()
        self.anchor_generator = AnchorGenerator(scales, aspect_ratios)
    
    def forward(self, image_list, features):
        images_size = [img.shape[-2:] for img in image_list]
        images = ImageList(torch.stack(image_list), images_size) # This asumes a constant input size of the images
        return self.anchor_generator(images, features)


def build_model(num_classes_backbone=1000, num_classes=2, rpn_depth=1, deconv=True):
    base_backbone = resnet50(num_classes_backbone)
    base_backbone = FeatureExtractor(base_backbone, ['layer1', 'layer2', 'layer3', 'layer4'])
    in_channels_list = [256, 512, 1024, 2048]
    base_backbone = FPN(base_backbone, in_channels_list=in_channels_list, out_channels=256, channels_funnel=_conv_1X1_leaky, interpolator=None, interpolator_filter=_conv_3X3_leaky, extra_blocks=fpn_extra_block)

    preprocessing_context = DSSD_ProcessingModule(256, 1024)
    context_processing = SSH_Residual_Context(1024, 512)
    context_sensitive_module = HH_ContextSensitivePrediction(preprocessing_context, context_processing, 512, 256)

    backbone = HH_FPN_ContextualBackbone(base_backbone, context_sensitive_module)

    scales = ((12,), (32,), (64,), (112,), (196, ), (256,), (384,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales)
    anchor_generator = ListAnchorGenerator(scales, aspect_ratios)
    rpn_processing_object = [Conv2dNormActivation(256, 256, kernel_size=3, norm_layer=None) for _ in range(rpn_depth)]
    num_anchors_per_location = len(scales[0]) * len(aspect_ratios[0])
    rpn_head = RPNHead(rpn_processing_object, 256, num_anchors_per_location)
    proposals_filter = build_batches_proposals_filter(detach=True, pre_nms_top_n=1000, min_size=1e-3, score_thresh=0.0, nms_thresh=0.7, post_nms_top_n=1000)
    rpn = Batches_RPN(anchor_generator, rpn_head, proposals_filter)
    #rpn = HH_RPN(rpn, backbone_out_channels=256)

    roi_pooling = ListMultiScaleRoIAlign(featmap_names=[0, 1, 2, 3], output_size=7, sampling_ratio=2)
    roi_processing_object = TwoMLPHead(256 * 7 ** 2, 1024)
    flatten_object = nn.Flatten(start_dim=1)
    roi_cls_layer = nn.Sequential(
        flatten_object,
        nn.Linear(1024, num_classes)
    )
    roi_bbox_layer = nn.Sequential(
        flatten_object,
        nn.Linear(1024, num_classes * 4)
    )
    roi_head = RoIHead(roi_pooling, roi_processing_object, roi_cls_layer, roi_bbox_layer)
    #roi_head = HH_RoIHead(roi_head, backbone_out_channels=256)

    model = FasterRCNN(backbone, rpn, roi_head)
    return model

def adapt_state_dict(state_dict):
    pass

# TODO: GeneralizedRCNNTransform at DataLoading

