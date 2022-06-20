
import math
import torch
import torch.nn as nn
#from torchvision.ops import Conv2dNormActivation


class RPNHead(nn.Module):
    # Mostly copied from Torchvision

    def __init__(self, processing_object, hidden_channels, num_anchors, conv2d=None):
        super().__init__()
        
        self.conv = processing_object # Sequential of [Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None) for _ in range(depth)], depth=1

        if conv2d is None:
            conv2d = lambda in_channels, out_channels :  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.cls_logits = conv2d(hidden_channels, num_anchors) # pixels far away from the base anchor should generate bboxes with very small scores
        self.bbox_pred = conv2d(hidden_channels, num_anchors * 4)

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

    # It allows both: "a simple 1 input interface if a backbone is provided at construction" and "a 2 input interface".
    # Usually different modules uses the backbone output, it is inefficient to recompute it.
    # A solution that would allow a simple interface while not losing efficiency would be to use a object that works as a backbone cache as backbone.

    def __init__(self, anchor_generator, rpn_head, proposals_filter, backbone=None, bbox_xform_clip=math.log(1000.0 / 16)):
        super().__init__()

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        self.proposals_filter = proposals_filter
        self.bbox_xform_clip = bbox_xform_clip

    def _image_channels_to_bbox_predictions(self, head_output, channels):
        # head_output : lists of L tensors [1, #A * channels, H_ij, W_ij]
        # output : [#A * H' * W', channels]

        bbox_pred = [h.view(-1, channels, H, W).permute(0, 2, 3, 1).reshape(-1, channels) for h in head_output]
        bbox_pred = torch.cat(bbox_pred, dim=1)

        return bbox_pred

    def _apply_deltas(self, anchors, bbox_deltas):
        # anchors       : [B, #A, 4]
        # bbox_deltas   : list of B tensors [#A * H'_i * W'_i, 4]
        # output        : list of B tensors [#A * H'_i * W'_i, 4]
        
        anchors = anchors.to(bbox_deltas[-1].dtype) # If the dtype is a cuda type, it will put the data on GPU

        proposals = []
        for image_anchors, image_deltas in zip(anchors[:, None], bbox_deltas):
            image_anchors = image_anchors.squeeze(0) # [#A, 4]
            image_deltas = image_deltas.reshape(image_anchors.shape[0], -1)

            widths = image_anchors[:, 2] - image_anchors[:, 0] # [#A]
            heights = image_anchors[:, 3] - image_anchors[:, 1] # [#A]
            ctr_x = image_anchors[:, 0] + 0.5 * widths # [#A]
            ctr_y = image_anchors[:, 1] + 0.5 * heights # [#A]

            dx = image_deltas[:, 0::4] # [#A, H'_i * W'_i]
            dy = image_deltas[:, 1::4] # [#A, H'_i * W'_i]
            dw = image_deltas[:, 2::4] # [#A, H'_i * W'_i]
            dh = image_deltas[:, 3::4] # [#A, H'_i * W'_i]

            dw = torch.clamp(dw, max=self.bbox_xform_clip) # [#A, H'_i * W'_i]
            dh = torch.clamp(dh, max=self.bbox_xform_clip) # [#A, H'_i * W'_i]
            
            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None] # [#A, H'_i * W'_i]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None] # [#A, H'_i * W'_i]
            pred_w = torch.exp(dw) * widths[:, None] # [#A, H'_i * W'_i]
            pred_h = torch.exp(dh) * heights[:, None] # [#A, H'_i * W'_i]

            # Distance from center to box's corner.
            c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h # [#A, H'_i * W'_i]
            c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w # [#A, H'_i * W'_i]

            pred_boxes1 = pred_ctr_x - c_to_c_w # [#A, H'_i * W'_i]
            pred_boxes2 = pred_ctr_y - c_to_c_h # [#A, H'_i * W'_i]
            pred_boxes3 = pred_ctr_x + c_to_c_w # [#A, H'_i * W'_i]
            pred_boxes4 = pred_ctr_y + c_to_c_h # [#A, H'_i * W'_i]
            pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).reshape(-1, 4) # [#A * H'_i * W'_i, 4]

            proposals.append(pred_boxes)
        return proposals


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
        
        anchors = self.anchor_generator(image_list, features) # [B, #A, 4]

        # len(num_anchors_per_image_and_level) = B
        # len(num_anchors_per_image_and_level[i]) = L
        # num_anchors_per_image_and_level[i][j] == #A_j * H_ij * W_ij
        num_anchors_per_image_and_level = [[level_objectness.numel() for level_objectness in level_objectness_list] for level_objectness_list in image_level_objectness_list]

        bbox_level_objectness_list = []
        bbox_level_deltas_list = []
        for level_objectness_list, level_deltas_list in zip(image_level_objectness_list, image_level_deltas_list):
            objectness = self._image_channels_to_bbox_predictions(level_objectness_list, 1) # [#A * H'_i * W'_i, 1]
            bbox_deltas = self._image_channels_to_bbox_predictions(level_deltas_list, 4) # [#A * H'_i * W'_i, 4]

            bbox_levels_objectness_list.append(objectness) # list of B tensors [#A * H'_i * W'_i, 1]
            bbox_levels_deltas_list.append(bbox_deltas) # list of B tensors [#A * H'_i * W'_i, 4]

        # Torchvision detach here the bbox_deltas and bbox_objectness because Faster R-CNN, I will do it at the Faster R-CNN step
        proposals = self._apply_deltas(anchors, bbox_levels_deltas_list) # list of B tensors [#A * H'_i * W'_i, 4]

        proposals = self.proposals_filter(proposals, bbox_level_objectness_list, image_list, num_anchors_per_image_and_level) # list of B tensors [N, 4]

        anchors = anchors.reshape(-1, 4) # [B * #A, 4]
        bbox_objectness = torch.cat(bbox_levels_objectness_list, dim=1).reshape(-1) # [B * #A * H'' * W'']
        bbox_deltas = torch.cat(bbox_levels_deltas_list, dim=1).reshape(-1, 4) # [B * #A * H'' * W'', 4]

        return proposals, anchors, bbox_objectness, bbox_deltas

def default_proposals_filter(proposals, bbox_level_objectness_list, image_list, num_anchors_per_image_and_level):
    pass


######################################

class Batches_RPN(nn.Module):

    # It requiers that all batches has the same image dimensions
    # It allows both: "a simple 1 input interface if a backbone is provided at construction" and "a 2 input interface".
    # Usually different modules uses the backbone output, it is inefficient to recompute it.
    # A solution that would allow a simple interface while not losing efficiency would be to use a object that works as a backbone cache as backbone.

    def __init__(self, anchor_generator, rpn_head, proposals_filter, backbone=None, bbox_xform_clip=math.log(1000.0 / 16)):
        super().__init__()

        self.backbone = backbone
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        self.proposals_filter = proposals_filter
        self.bbox_xform_clip = bbox_xform_clip

    def _image_channels_to_bbox_predictions(self, head_output, channels):
        # head_output : [B, #A * channels, H_i, W_i]
        # output : [B, #A * H_i * W_i, channels]

        B = head_output.shape[0]
        bbox_pred = head_output.view(B, -1, channels, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, channels)
        
        return bbox_pred

    def _apply_deltas(self, anchors, bbox_deltas):
        # anchors       : [B * #A, 4]
        # bbox_deltas   : [B, #A * H' * W', 4]
        # output        : [B, #A * H' * W', 4]
        
        out_shape = bbox_deltas.shape
        anchors = anchors.to(bbox_deltas[-1].dtype) # If the dtype is a cuda type, it will put the data on GPU
        bbox_deltas = bbox_deltas.reshape(anchor.shape[0], -1) # [B * #A, H' * W' * 4]

        
        widths = boxes[:, 2] - boxes[:, 0] # [B * #A]
        heights = boxes[:, 3] - boxes[:, 1] # [B * #A]
        ctr_x = boxes[:, 0] + 0.5 * widths # [B * #A]
        ctr_y = boxes[:, 1] + 0.5 * heights # [B * #A]

        dx = rel_codes[:, 0::4] # [B * #A, H' * W']
        dy = rel_codes[:, 1::4] # [B * #A, H' * W']
        dw = rel_codes[:, 2::4] # [B * #A, H' * W']
        dh = rel_codes[:, 3::4] # [B * #A, H' * W']

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip) # [B * #A, H' * W']
        dh = torch.clamp(dh, max=self.bbox_xform_clip) # [B * #A, H' * W']

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None] # [B * #A, H' * W']
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None] # [B * #A, H' * W']
        pred_w = torch.exp(dw) * widths[:, None] # [B * #A, H' * W']
        pred_h = torch.exp(dh) * heights[:, None] # [B * #A, H' * W']

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h # [B * #A, H' * W']
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w # [B * #A, H' * W']

        pred_boxes1 = pred_ctr_x - c_to_c_w # [B * #A, H' * W']
        pred_boxes2 = pred_ctr_y - c_to_c_h # [B * #A, H' * W']
        pred_boxes3 = pred_ctr_x + c_to_c_w # [B * #A, H' * W']
        pred_boxes4 = pred_ctr_y + c_to_c_h # [B * #A, H' * W']
        proposals = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2) # [B * #A, H' * W', 4]

        proposals.reshape(out_shape) # [B, #A * H' * W', 4]
        return proposals


    def forward(self, image_list, features=None):
        if features is None:
            assert self.backbone is not None
            features = self.backbone(torch.stack(image_list, dim=0)) # list of L tensors [B, C, H_i, W_i]
        
        batch_size = len(image_list)

        level_image_objectness, level_image_deltas = self.rpn_head(features) # lists of L tensors [B, #A * 1, H_i, W_i], lists of L tensors [B, #A * 4, H_i, W_i]
        anchors = self.anchor_generator(image_list, features) # [B, #A, 4]

        # len(num_anchors_per_level) = L ; num_anchors_per_level[i] == #A * H_i * W_i
        num_anchors_per_level = [level_objectness[0].numel() for level_objectness in level_image_objectness]

        level_bbox_objectness = []
        level_bbox_deltas = []
        for single_level_image_objectness, single_level_image_deltas in zip(level_image_objectness, level_image_deltas)

            objectness = self._image_channels_to_bbox_predictions(single_level_image_objectness, 1) # [B, #A * H_i * W_i, 1]
            bbox_deltas = self._image_channels_to_bbox_predictions(single_level_image_deltas, 4) # [B, #A * H_i * W_i, 4]

            level_bbox_objectness.append(objectness)
            level_bbox_deltas.append(bbox_deltas)

        anchors = anchors.reshape(-1, 4) # [B * #A, 4]
        bbox_objectness = torch.cat(level_bbox_objectness, dim=1).reshape(-1) # [B * #A * H' * W']
        bbox_deltas = torch.cat(level_bbox_deltas, dim=1).reshape(-1, 4) # [B * #A * H' * W', 4]

        # Torchvision detach here the bbox_deltas and bbox_objectness because Faster R-CNN, I will do it at the Faster R-CNN step
        proposals = self._apply_deltas(anchors, bbox_deltas) # [B, #A * H' * W', 4]

        proposals = self.proposals_filter(proposals, bbox_objectness, image_list, num_anchors_per_level) # ¿?

        return proposals, anchors, bbox_objectness, bbox_deltas

def default_batches_proposals_filter(proposals, bbox_objectness, image_list, num_anchors_per_level):
    # proposals             : [B, #A * H' * W', 4]
    # bbox_objectness       : [B * #A * H' * W']
    # image_list            : list of B images [C, H, W]
    # num_anchors_per_level : list of L integers #A * H_i * W_i (sum(num_anchors_per_level) == #A * H' * W')

    
    pass
