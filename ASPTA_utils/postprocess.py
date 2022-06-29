
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection._utils as det_utils
from torchvision.ops import boxes as box_ops


########### Faster R-CNN #################

def apply_deltas(box_regression, proposals, bbox_xform_clip=math.log(1000.0 / 16)):
    # proposals         : list of B tensors [N_i, 4]
    # box_regression    : [N, 4 * #cls]
    # output            : [N, #cls, 4]
    
    proposals = torch.cat(proposals, dim=0) # [N, 4]
    box_regression = box_regression.reshape(proposals.shape[0], -1) # [N, 4 * #cls]
    
    # [N] x 4
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    ctr_x = proposals[:, 0] + 0.5 * widths
    ctr_y = proposals[:, 1] + 0.5 * heights

    # different deltas per class [N, #cls] x 4
    dx = box_regression[:, 0::4]
    dy = box_regression[:, 1::4]
    dw = box_regression[:, 2::4]
    dh = box_regression[:, 3::4]

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    # [N, #cls] x 4
    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    # Distance from center to box's corner.
    c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

    pred_boxes1 = pred_ctr_x - c_to_c_w
    pred_boxes2 = pred_ctr_y - c_to_c_h
    pred_boxes3 = pred_ctr_x + c_to_c_w
    pred_boxes4 = pred_ctr_y + c_to_c_h
    pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2) # [N, #cls, 4]

    return pred_boxes

def postprocess_detections(class_logits, box_regression, proposals, image_list, score_thresh=0, nms_thresh=0, detections_per_img=None, bbox_xform_clip=math.log(1000.0 / 16)):
    # class_logits      : [N, #cls]
    # box_regression    : [N, #cls * 4]
    # proposals         : list of B tensors [N_i, 4]

    device = class_logits.device
    num_classes = class_logits.shape[-1]
    image_shapes = [img.shape[1:] for img in image_list]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

    pred_boxes = apply_deltas(box_regression, proposals, bbox_xform_clip=bbox_xform_clip)
    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

        return all_boxes, all_scores, all_labels


#################### Head Hunter Object Detector ######################

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


#################### Detection ######################

def default_postprocess(class_logits, box_regression, proposals, image_list, original_image_sizes, score_thresh=0, nms_thresh=0, detections_per_img=None):

    image_shapes = [img.shape[1:] for img in image_list]

    all_boxes, all_scores, all_labels = postprocess_detections(class_logits, box_regression, proposals, image_list, score_thresh=score_thresh, nms_thresh=nms_thresh, detections_per_img=detections_per_img)

    for i, (bbox, im_s, o_im_s) in enumerate(zip(all_boxes, image_shapes, original_image_sizes)):
        bbox = resize_boxes(bbox, im_s, o_im_s)
        all_boxes[i] = bbox
    
    return all_boxes, all_scores

def build_postprocess(score_thresh=0, nms_thresh=0, detections_per_img=None):
    def postprocess(class_logits, box_regression, proposals, image_list, original_image_sizes):
        return default_postprocess(class_logits, box_regression, proposals, image_list, original_image_sizes, score_thresh=score_thresh, nms_thresh=nms_thresh, detections_per_img=detections_per_img)
    return postprocess


###################### Head Hunter Tracker (preprocess) ############################

def constraint_boxes(detections, im_shape):
    (startX, startY, endX, endY, score) = detections

    startX, startY = max(0, startX), max(0, startY)
    max_x, max_y = im_shape[1], im_shape[0]
    endX, endY = min(endX, max_x-1), min(endY, max_y-1)

    detections = (startX, startY, endX, endY, score)
    return detections

def check_area(detections, im_shape):
    if len(detections) < 1 : raise ValueError("Invalid detection length")

    (startX, startY, endX, endY) = detections[:4]

    box_width = (endX - startX)
    box_height = (endY - startY)
    
    area =  box_width * box_height
    
    if area < 10 or box_height < 2 or box_width < 2:
        return False
    return True

def get_refined_detection(detections, im_shape, confidence):
    """
    Constraint the output of detector to lie within the image.
    Also check if the detection is valid by measuring area of BB.
    detections : [[x_min, y_min, x_max, y_max, score]]
    im_shape : (H, W, 3)
    """

    refined_detection = []
    for dets in detections:
        score = dets[-1]
        
        if score < confidence:
            continue

        dets = constraint_boxes(dets, im_shape)

        if check_area(dets, im_shape) is False:
            continue
        
        refined_detection.append(dets)
        
    refined_detection = np.array(refined_detection)
    return refined_detection

