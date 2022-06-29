
import csv
import numpy as np
import os
import os.path as osp
import torch

from ASPTA_utils.build_model import build_model, adapt_state_dict
from ASPTA_utils.build_tracker import build_tracker
from ASPTA_utils.im_utils import *
from ASPTA_utils.postprocess import build_postprocess, get_refined_detection
from ASPTA_utils.pseudo_dataset import build_image_transform
from ASPTA_utils.pseudo_dataset import SequencesPseudoDataset, SequenceImagesPseudoDataset


# I/O VARIABLES OR HYPERPARAMETER I CAN TOUCH
PRETRAINED_PATH = '/mnt/c/Users/Ignasi/Downloads/FT_R50_epoch_24.pth'
BASE_DIR = '/mnt/c/Users/Ignasi/Downloads/HT21/'
DATASETS = [('train', 'test'), ('train', ), ('test', )][0]

SAVE_FRAMES = True
SAVE_DIR_IMGS = './outputs/imgs/'
SAVE_DIR_PREDS = './outputs/preds/'

min_size = None
max_size = None
dset_mean_std = [[117, 110, 105], [67.10, 65.45, 66.23]]
image_mean = [i/255. for i in dset_mean_std[0]]
image_std = [i/255. for i in dset_mean_std[1]]



def build_apply_det_model(postprocess, detection_confidence=0.006): # detection_confidence=0.6
    def apply_det_model(one_image_list, model):

        assert len(one_image_list) == 1

        class_logits, box_regression, proposals = model(one_image_list)

        original_image_sizes = [one_image_list[0].shape]
        boxes, scores = postprocess(class_logits, box_regression, proposals, one_image_list, original_image_sizes) # original_image_sizes == image_shape 'cause no min_size and max_size on Tracker
        
        boxes, scores = boxes[0], scores[0]
        detections = torch.cat((boxes, torch.unsqueeze(scores, 1)), 1).cpu().detach().numpy()

        refined_det = get_refined_detection(detections, original_image_sizes[0][::-1], detection_confidence)
        if len(refined_det.shape) == 1:
            refined_det = refined_det.reshape((0, 5)) # or refined_det.reshape((-1, 5))
        boxes, det_scores = refined_det[:, :4], np.squeeze(refined_det[:, 4:])

        detections = np.c_[boxes, det_scores]

        return boxes, det_scores, detections
    return apply_det_model


# BUILD DETECTOR HEAD DETECTION; return class_logits, box_regression, proposals # before filtering and postprocessing
det_model = build_model()
state_dict = torch.load('/mnt/c/Users/Ignasi/Downloads/FT_R50_epoch_24.pth', map_location=torch.device('cpu'))
state_dict = adapt_state_dict(state_dict)
det_model.load_state_dict(state_dict, strict=False)
det_model.eval()

# BUILD AUXILIAR DETECTOR FOR PARTICLES REGRESSION; RPN will be changed on the tracker
reg_model = build_model()
reg_model.load_state_dict(state_dict, strict=False)
reg_model.eval()

def change_rpn(reg_model, particles):
    # particles should be a list of 1 tensor [N, 4]
    def get_particles(self, image_list, features=None):
        return particles, None, None, None
    reg_model.rpn = get_particles

# BUILD PRE AND POST PROCESSING FUNCTS
transform = build_image_transform(min_size, max_size, image_mean, image_std)
postprocess = build_postprocess(score_thresh=0, nms_thresh=0, detections_per_img=None)
apply_det_model = build_apply_det_model(postprocess)
apply_reg_model = build_apply_det_model(postprocess)

# Process data, no DataLoader is used due to lack of time
for dset in DATASETS:
    # BUILD SEQUENCE DATASET
    squences_dataset = SequencesPseudoDataset(BASE_DIR, dset='train', start_ind=0)

    # BUILD SEQUENCE TRACKER
    _, im_shape, _, _ = squences_dataset[0]
    tracker = build_tracker(im_shape, reg_model, change_rpn, apply_reg_model)

    for seq_images, im_shape, cam_motion, im_path in squences_dataset:
        # BUILD IMAGES DATASET
        images_dataset = SequenceImagesPseudoDataset(seq_images, transform)

        for one_image_batch in images_dataset: # one_image_batch is a tensor [1, C, H, W], it works as a list of 1 element [C, H, W]
            bboxes, det_scores, detections = apply_det_model([one_image_batch[0].float()], det_model)
            results, matched_map, frame_number = tracker(one_image_batch, bboxes, det_scores)

            if SAVE_FRAMES:
                plotted_im = plot_boxes(one_image_batch[0].permute(1, 2, 0).cpu().numpy(), matched_map)
                save_dir = osp.join(SAVE_DIR_IMGS, f"{im_path[len(BASE_DIR):]}")
                os.makedirs(save_dir, exist_ok=True)
                imsave(osp.join(save_dir, f"{frame_number:06d}" + '.jpg'), plotted_im)
        
        traj_dir = osp.join(SAVE_DIR_PREDS, im_path[len(BASE_DIR):])
        os.makedirs(traj_dir, exist_ok=True)
        with open(osp.join(traj_dir, 'pred.txt'), "w+") as of:
            writer = csv.writer(of, delimiter=',')

            for i, track in results.items():
                for frame, bb in track.items():

                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]

                    writer.writerow([frame, i, x1, y1, x2 - x1 + 1, y2 - y1 + 1, 1, 1, 1, 1])
