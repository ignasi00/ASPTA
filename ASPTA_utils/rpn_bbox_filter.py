
import torch
import torch.nn as nn
import torchvision.models.detection._utils as det_utils
from torchvision.ops import boxes as box_ops


def _get_top_n_idx(objectness, num_proposals_per_level, pre_nms_top_n=None):
    r = []
    offset = 0
    for ob in objectness.split(num_proposals_per_level, 1):
        num_proposals = ob.shape[1]
        pre_nms_top_n = pre_nms_top_n or num_proposals

        pre_nms_top_n = det_utils._topk_min(ob, pre_nms_top_n, 1) # same as min(ob.shape[1], pre_nms_top_n)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1) # [B, #A * H' * W'] -> [B, pre_nms_top_n] with the higher values of dim 1 for each B
        
        r.append(top_n_idx + offset) # previous levels offset
        offset += num_proposals
    top_n_idx = torch.cat(r, dim=1)
    return top_n_idx

def _score_based_filter(objectness, levels, proposals, num_proposals_per_level, pre_nms_top_n, device):
    # Get the best "N = min(N, #A * H' * W')" objectness indexes for each level and image
    top_n_idx = _get_top_n_idx(objectness, num_proposals_per_level, pre_nms_top_n=pre_nms_top_n) # [B, N]

    image_range = torch.arange(proposals.shape[0], device=device) # [B]
    batch_idx = image_range[:, None] # [B, 1]

    objectness = objectness[batch_idx, top_n_idx] # [B, N]
    levels = levels[batch_idx, top_n_idx] # [B, N]
    proposals = proposals[batch_idx, top_n_idx] # [B, N]

    return objectness, levels, proposals

def build_nms_funct(nms_thresh):
    def nms_funct(boxes, scores, lvl):
        return box_ops.batched_nms(boxes, scores, lvl, nms_thresh)
    return nms_funct

def default_batches_proposals_filter(proposals, objectness, image_list, num_proposals_per_level, detach=True, pre_nms_top_n=None, min_size=0, score_thresh=0, nms_funct=None, post_nms_top_n=None):
    # proposals             : [B, #A * H' * W', 4]
    # bbox_objectness       : [B * #A * H' * W']
    # image_list            : list of B images [C, H, W]
    # num_proposals_per_level : list of L integers #A * H_i * W_i (sum(num_proposals_per_level) == #A * H' * W')
    
    nms_funct = nms_funct or build_nms_funct(0.5)

    if detach:
        proposals = proposals.detach()
        objectness = objectness.detach()
    
    device = proposals.device
    image_shapes = [img.shape[1:] for img in image_list]

    objectness = objectness.reshape(proposals.shape[0], -1) # [B, #A * H' * W']

    level_mask = torch.repeat_interleave(torch.as_tensor(num_proposals_per_level)) # [#A * H' * W'] with #A * H_i * W_i elemens of value i with i from 0 to L-1
    level_mask = level_mask.reshape(1, -1).expand_as(objectness) # [B, #A * H' * W']

    objectness, levels, proposals = _score_based_filter(objectness, level_mask, proposals, num_proposals_per_level, pre_nms_top_n, device) # [B, N], [B, N], [B, N]

    objectness_prob = torch.sigmoid(objectness) # [B, N]

    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape) # No bbox will contain pixels outside the image

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, min_size)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep] # [¿?]

        # remove low scoring boxes
        keep = torch.where(scores >= score_thresh)[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep] # [¿?']

        # non-maximum suppression, independently done per level
        keep = nms_funct(boxes, scores, lvl) # [¿?'']

        # keep only topk scoring predictions
        keep = keep[: post_nms_top_n]
        boxes, scores = boxes[keep], scores[keep] # [¿?''' <= post_nms_top_n]

        final_boxes.append(boxes)
        final_scores.append(scores)
    return final_boxes, final_scores # lists of B elements like [¿? <= post_nms_top_n]

def build_batches_proposals_filter(detach, pre_nms_top_n, min_size, score_thresh, nms_thresh, post_nms_top_n):
    nms_funct = build_nms_funct(nms_thresh)

    def batches_proposals_filter(proposals, objectness, image_list, num_proposals_per_level):
        return default_batches_proposals_filter(proposals, objectness, image_list, num_proposals_per_level, detach=detach, pre_nms_top_n=pre_nms_top_n, min_size=min_size, score_thresh=score_thresh, nms_funct=nms_funct, post_nms_top_n=post_nms_top_n)
    return batches_proposals_filter
