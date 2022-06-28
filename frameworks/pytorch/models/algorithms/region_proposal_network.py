# Barely reach my "it is an algorithm" therdshold

import math
import torch
import torch.nn as nn
#from torchvision.ops import Conv2dNormActivation


class RPNHead(nn.Module):
    # Mostly copied from Torchvision

    def __init__(self, processing_object, hidden_channels, num_anchors_per_location, conv2d=None):
        super().__init__()
        
        self.conv = processing_object # Sequential of [Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None) for _ in range(depth)], depth=1

        if conv2d is None:
            conv2d = lambda in_channels, out_channels :  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.cls_logits = conv2d(hidden_channels, num_anchors_per_location)
        self.bbox_pred = conv2d(hidden_channels, num_anchors_per_location * 4)

    def forward(self, feature_maps):
        if isinstance(feature_maps, torch.Tensor) : feature_maps = [feature_maps]

        objectness = []
        bbox_deltas = []
        
        for feature_map in feature_maps:
            t = self.conv(feature_map)
            objectness.append(self.cls_logits(t))
            bbox_deltas.append(self.bbox_pred(t))
        
        return objectness, bbox_deltas

#######################################################

class RPN(nn.Module):
    # Mostly copied from Torchvision

    # It allows both: "a simple 1 input interface if a backbone is provided at construction" and "a 2 input interface".
    # Usually different modules uses the backbone output, it is inefficient to recompute it.
    # A solution that would allow a simple interface while not losing efficiency would be to use a object that works as a backbone cache as backbone.

    def __init__(self, anchor_generator, rpn_head, proposals_filter=None, backbone=None, bbox_xform_clip=math.log(1000.0 / 16)):
        super().__init__()

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        self.proposals_filter = proposals_filter
        self.bbox_xform_clip = bbox_xform_clip

    def _image_channels_to_bbox_predictions(self, head_output, channels):
        # head_output : lists of L tensors [1, #A * channels, H_ij, W_ij]
        # output : [H' * W' * #A, channels]

        bbox_pred = [h.view(-1, channels, H, W).permute(0, 2, 3, 1).reshape(-1, channels) for h in head_output]
        bbox_pred = torch.cat(bbox_pred, dim=1)

        return bbox_pred

    def _apply_deltas(self, anchors, bbox_deltas):
        # anchors       : list of B tensors [H'_i * W'_i * #A, 4]
        # bbox_deltas   : list of B tensors [H'_i * W'_i * #A, 4]
        # output        : list of B tensors [H'_i * W'_i * #A, 4]
        
        anchors = anchors.to(bbox_deltas[-1].dtype) # If the dtype is a cuda type, it will put the data on GPU

        proposals = []
        for image_anchors, image_deltas in zip(anchors[:, None], bbox_deltas):
            image_anchors = image_anchors.squeeze(0) # [H'_i * W'_i * #A, 4]
            #image_deltas = image_deltas.reshape(image_anchors.shape[0], -1) # [H'_i * W'_i * #A, 4]

            widths = image_anchors[:, 2] - image_anchors[:, 0] # [H'_i * W'_i * #A]
            heights = image_anchors[:, 3] - image_anchors[:, 1] # [H'_i * W'_i * #A]
            ctr_x = image_anchors[:, 0] + 0.5 * widths # [H'_i * W'_i * #A]
            ctr_y = image_anchors[:, 1] + 0.5 * heights # [H'_i * W'_i * #A]

            dx = image_deltas[:, 0] # [H'_i * W'_i * #A]
            dy = image_deltas[:, 1] # [H'_i * W'_i * #A]
            dw = image_deltas[:, 2] # [H'_i * W'_i * #A]
            dh = image_deltas[:, 3] # [H'_i * W'_i * #A]

            dw = torch.clamp(dw, max=self.bbox_xform_clip) # [H'_i * W'_i * #A]
            dh = torch.clamp(dh, max=self.bbox_xform_clip) # [H'_i * W'_i * #A]
            
            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None] # [H'_i * W'_i * #A]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None] # [H'_i * W'_i * #A]
            pred_w = torch.exp(dw) * widths[:, None] # [H'_i * W'_i * #A]
            pred_h = torch.exp(dh) * heights[:, None] # [H'_i * W'_i * #A]

            # Distance from center to box's corner.
            c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h # [H'_i * W'_i * #A]
            c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w # [H'_i * W'_i * #A]

            pred_boxes1 = pred_ctr_x - c_to_c_w # [H'_i * W'_i * #A]
            pred_boxes2 = pred_ctr_y - c_to_c_h # [H'_i * W'_i * #A]
            pred_boxes3 = pred_ctr_x + c_to_c_w # [H'_i * W'_i * #A]
            pred_boxes4 = pred_ctr_y + c_to_c_h # [H'_i * W'_i * #A]
            pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1).reshape(-1, 4) # [H'_i * W'_i * #A, 4]

            proposals.append(pred_boxes)
        return proposals # list of B tensors [H'_i * W'_i * #A, 4]


    def forward(self, image_list, features=None):
        if features is None:
            assert self.backbone is not None
            features = [] # list of B lists of L tensors [1, C, H_ij, W_ij]
            for image in image_list:
                features.append(self.backbone(image.unsqueeze(0)))
        
        batch_size = len(image_list)

        image_level_objectness_list = []
        image_level_deltas_list = []
        for image_feats in features:
            image_level_objectness, image_level_deltas = self.rpn_head(image_feats)
            image_level_objectness_list.append(image_level_objectness) # list of B lists of L tensors [1, #A * 1, H_ij, W_ij]
            image_level_deltas_list.append(image_level_deltas) # list of B lists of L tensors [1, #A * 4, H_ij, W_ij]
        
        anchors = self.anchor_generator(image_list, features) # list of B tensors [H'_i * W'_i * #A, 4]

        # len(num_anchors_per_image_and_level) = B
        # len(num_anchors_per_image_and_level[i]) = L
        # num_anchors_per_image_and_level[i][j] == #A * H_ij * W_ij
        num_anchors_per_image_and_level = [[level_objectness.numel() for level_objectness in level_objectness_list] for level_objectness_list in image_level_objectness_list]

        bbox_level_objectness_list = []
        bbox_level_deltas_list = []
        for level_objectness_list, level_deltas_list in zip(image_level_objectness_list, image_level_deltas_list): # TODO: only 1 for with all
            objectness = self._image_channels_to_bbox_predictions(level_objectness_list, 1) # [H'_i * W'_i * #A, 1]
            bbox_deltas = self._image_channels_to_bbox_predictions(level_deltas_list, 4) # [H'_i * W'_i * #A, 4]

            bbox_levels_objectness_list.append(objectness) # list of B tensors [H'_i * W'_i * #A, 1]
            bbox_levels_deltas_list.append(bbox_deltas) # list of B tensors [H'_i * W'_i * #A, 4]

        # Torchvision detach here the bbox_deltas and bbox_objectness because Faster R-CNN, I will do it at the proposal_filter step if needed
        proposals = self._apply_deltas(anchors, bbox_levels_deltas_list) # list of B tensors [H'_i * W'_i * #A, 4]

        if self.proposals_filter is not None:
            proposals, _ = self.proposals_filter(proposals, bbox_level_objectness_list, image_list, num_anchors_per_image_and_level) # list of B tensors [N_i, 4]

        anchors = anchors.reshape(-1, 4) # [B * #A * H'' * W'', 4]
        bbox_objectness = torch.cat(bbox_levels_objectness_list, dim=1).reshape(-1) # [B * H'' * W'' * #A ]
        bbox_deltas = torch.cat(bbox_levels_deltas_list, dim=1).reshape(-1, 4) # [B * H'' * W'' * #A , 4]

        return proposals, anchors, bbox_objectness, bbox_deltas

######################################

class Batches_RPN(nn.Module):
    # Mostly copied from Torchvision

    # It requiers that all batches has the same image dimensions
    # It allows both: "a simple 1 input interface if a backbone is provided at construction" and "a 2 input interface".
    # Usually different modules uses the backbone output, it is inefficient to recompute it.
    # A solution that would allow a simple interface while not losing efficiency would be to use a object that works as a backbone cache as backbone.

    def __init__(self, anchor_generator, rpn_head, proposals_filter=None, backbone=None, bbox_xform_clip=math.log(1000.0 / 16)):
        super().__init__()

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        self.proposals_filter = proposals_filter
        self.bbox_xform_clip = bbox_xform_clip

    def _image_channels_to_bbox_predictions(self, head_output, channels):
        # head_output : [B, #A * channels, H_i, W_i]
        # output : [B, H_i * W_i * #A, channels]

        B = head_output.shape[0]
        bbox_pred = head_output.view(B, -1, channels, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, channels)
        
        return bbox_pred

    def _apply_deltas(self, anchors, bbox_deltas):
        # Version oly for RPN, when #cls > 1 => it changes
        # anchors       : [B * H' * W' * #A, 4]
        # bbox_deltas   : [B * H' * W' * #A, 4]
        # output        : [B * H' * W' * #A, 4]
        
        out_shape = bbox_deltas.shape
        anchors = anchors.to(bbox_deltas[-1].dtype) # If the dtype is a cuda type, it will put the data on GPU
        bbox_deltas = bbox_deltas.reshape(anchor.shape[0], -1) # [B * H' * W' * #A, 4]

        
        widths = boxes[:, 2] - boxes[:, 0] # [B * H' * W' * #A]
        heights = boxes[:, 3] - boxes[:, 1] # [B * H' * W' * #A]
        ctr_x = boxes[:, 0] + 0.5 * widths # [B * H' * W' * #A]
        ctr_y = boxes[:, 1] + 0.5 * heights # [B * H' * W' * #A]

        dx = rel_codes[:, 0] # [B * H' * W' * #A]
        dy = rel_codes[:, 1] # [B * H' * W' * #A]
        dw = rel_codes[:, 2] # [B * H' * W' * #A]
        dh = rel_codes[:, 3] # [B * H' * W' * #A]

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip) # [B * H' * W' * #A]
        dh = torch.clamp(dh, max=self.bbox_xform_clip) # [B * H' * W' * #A]

        pred_ctr_x = dx * widths + ctr_x # [B * H' * W' * #A]
        pred_ctr_y = dy * heights + ctr_y # [B * H' * W' * #A]
        pred_w = torch.exp(dw) * widths # [B * H' * W' * #A]
        pred_h = torch.exp(dh) * heights # [B * H' * W' * #A]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h # [B * H' * W' * #A]
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w # [B * H' * W' * #A]

        pred_boxes1 = pred_ctr_x - c_to_c_w # [B * H' * W' * #A]
        pred_boxes2 = pred_ctr_y - c_to_c_h # [B * H' * W' * #A]
        pred_boxes3 = pred_ctr_x + c_to_c_w # [B * H' * W' * #A]
        pred_boxes4 = pred_ctr_y + c_to_c_h # [B * H' * W' * #A]
        proposals = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1) # [B * H' * W' * #A, 4]

        proposals.reshape(out_shape) # [B * H' * W' * #A, 4]
        return proposals


    def forward(self, image_list, features=None):
        if features is None:
            assert self.backbone is not None
            features = self.backbone(torch.stack(image_list, dim=0)) # list of L tensors [B, C, H_j, W_j]
        
        batch_size = len(image_list)

        level_image_objectness, level_image_deltas = self.rpn_head(features) # lists of L tensors [B, #A * 1, H_j, W_j], lists of L tensors [B, #A * 4, H_j, W_j]
        anchors = self.anchor_generator(image_list, features) # list of B tensors [H_i * W_i * #A, 4]

        # len(num_proposals_per_level) = L ; num_proposals_per_level[i] == #A * H_j * W_j
        num_proposals_per_level = [level_objectness[0].numel() for level_objectness in level_image_objectness]

        level_bbox_objectness = []
        level_bbox_deltas = []
        for single_level_image_objectness, single_level_image_deltas in zip(level_image_objectness, level_image_deltas):

            objectness = self._image_channels_to_bbox_predictions(single_level_image_objectness, 1) # [B, H_j * W_j * #A, 1]
            bbox_deltas = self._image_channels_to_bbox_predictions(single_level_image_deltas, 4) # [B, H_j * W_j * #A, 4]

            level_bbox_objectness.append(objectness)
            level_bbox_deltas.append(bbox_deltas)

        anchors = torch.cat(anchors, dim=0).reshape(-1, 4) # [B * H' * W' * #A, 4]
        bbox_objectness = torch.cat(level_bbox_objectness, dim=1).reshape(batch_size, -1) # [B, H' * W' * #A]
        bbox_deltas = torch.cat(level_bbox_deltas, dim=1).reshape(-1, 4) # [B * H' * W' * #A, 4]

        # Torchvision detach here the bbox_deltas and bbox_objectness because Faster R-CNN does not backpropagate through the proposals, I will do it at the Faster R-CNN or the filtering step
        proposals = self._apply_deltas(anchors, bbox_deltas) # [B * H' * W' * #A, 4]
        proposals = proposals.reshape(batch_size, -1, 4) # [B, H' * W' * #A, 4]

        if self.proposals_filter is not None:
            proposals, _ = self.proposals_filter(proposals, bbox_objectness, image_list, num_proposals_per_level) # list of B tensors [N_i, 4]
        else:
            # proposals = [p.squeeze(0) for p in proposals.split(1, dim=0)]
            proposals = [p for p in proposals] # list of B tensors [H' * W' * #A, 4]

        return proposals, anchors, bbox_objectness, bbox_deltas
