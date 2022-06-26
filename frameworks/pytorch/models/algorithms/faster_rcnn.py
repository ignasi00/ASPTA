# Barely reach my "it is an algorithm" therdshold

import math
import torch
import torch.nn as nn


class RoIHead(nn.Module):

    def __init__(self, roi_pooling, processing_object, cls_layer, bbox_layer):
        super().__init__()

        self.roi_pooling = roi_pooling
        self.processing_object = processing_object
        self.cls_layer = cls_layer
        self.bbox_layer = bbox_layer
    
    def forward(self, features_list, proposals, image_list):
        image_shapes = [img.shape[1:] for img in image_list]
        box_features = self.roi_pooling(features_list, proposals, image_shapes)
        box_features = self.processing_object(box_features)
        class_logits = self.cls_layer(box_features)
        box_regression = self.bbox_layer(box_features)

        return class_logits, box_regression

class FasterRCNN(nn.Module):
    # When training the FasterRCNN, they usually change the proposals and match them with a target.
    # It means that the training should be done using each submodule instead of the full model, as I do not like the model having any target related logic

    def __init__(self, backbone, rpn, roi_head):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_head = roi_head

    def forward(self, image_list):
        features_list = self.backbone(image_list)

        proposals, _, _, _ = self.rpn(image_list, features_list)
        class_logits, box_regression = self.roi_head(features_list, proposals, image_list)

        return class_logits, box_regression, proposals
