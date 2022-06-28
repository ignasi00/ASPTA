# TODO: Better divide code in stand-alone blocs and try to reach a near functional implementation
# The final algithms should be constructed combining Callables
# The constructors shouldn't have too much parameters

import os
import sys
from collections import Counter
from glob import glob
from tqdm import tqdm
import numpy as np
import os.path as osp
import cv2
import json
import pycocotools.mask as cocomask
from munkres import Munkres, print_matrix, make_cost_matrix

from track import Track
from im_utils import * # check_area, compute_histogram, warp_pos, compute_centroid
from scipy.misc import imread, imsave
from scipy.spatial.distance import cdist
from copy import deepcopy
from operator import itemgetter
import base64
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision.ops.boxes import clip_boxes_to_image, nms

munkres_inst = Munkres()


def compute_crop(img, tr_pos, im_shape):
    ## Crop the head for gathering HSV features.

    max_y, max_x = im_shape[:2]

    xmin, ymin, xmax, ymax = [int(round(i)) for i in tr_pos[:4]]
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(max_x, xmax), min(max_y, ymax)

    cropped_target = img[ymin:ymax, xmin:xmax, :]

    return cropped_target

def align(position, warp_matrix, device):
    """Aligns the positions of active and inactive tracks depending on camera motion.
        Code borrowed from Tim Meinhardt
        """

    position_gpu = torch.tensor(position, dtype=torch.float32, device=device)

    aligned_pos = []
    for pos in position_gpu:

        aligned_pos.append(warp_pos(pos, warp_matrix).numpy().tolist())
        
    if len(aligned_pos) > 0:
        return np.array(aligned_pos).reshape(-1, 4)
    else:
        return np.array(aligned_pos)

class Tracker():

    def __init__(self, im_shape, compute_warp_matrix, reg_model, change_rpn, apply_reg_model, cam_motion=True, inactive_patience=25, n_particles=100, regression_thresh=0.1, regression_nms_thresh=0.3, detection_nms_thresh=0.3, use_reid=True, lambd=0.9, device=None):
        self.im_shape = im_shape
        
        self.compute_warp_matrix = compute_warp_matrix
        self.obj_detector = reg_model
        self.change_rpn = change_rpn
        self.apply_reg_model = apply_reg_model

        self.cam_motion = cam_motion
        self.inactive_patience = inactive_patience
        self.n_particles = n_particles
        self.regression_thresh = regression_thresh
        self.regression_nms_thresh = regression_nms_thresh
        self.detection_nms_thresh = detection_nms_thresh
        self.use_reid = use_reid
        self.lambd = lambd
        self.device = device or torch.device("cpu")

        self.frame_number = 0
        self.last_image = None
        self.track_num = 0
        self.tracks = []
        self.inactive_tracks = []
        self.results = {}

    def get_track(self, cur_id):
        for tr in self.tracks:
            if tr.id == cur_id:
                return tr

    def get_lost_track(self, cur_id):
        for tr in self.inactive_tracks:
            if tr.id == cur_id:
                return tr
    
    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        self.inactive_tracks += tracks
    
    def track_found_again(self, track):
        # reset inactive count and add to self.track
        self.inactive_tracks = [it for it in self.inactive_tracks if it not in [track]]
        self.tracks.append(track)

    def regress_lost_track(self):
        for lt in self.inactive_tracks:
            lt.step()

        temp_lost = []
        for lt in self.inactive_tracks:

            if not check_area(lt.pos, self.im_shape):
                continue

            if lt.kill_flag:
                continue

            if lt.count_inactive > self.inactive_patience:
                continue

            temp_lost.append(lt)

        self.inactive_tracks = temp_lost
        return self.inactive_tracks

    def regress_single_track(self, img, regress_pos):

        regress_pos = torch.tensor(regress_pos)
        regress_pos = regress_pos.to(self.device)
        self.change_rpn(self.obj_detector, [regress_pos])
        t_box, t_score, detections = self.apply_reg_model(img)

        #t_score = t_score[:, 1:].squeeze(dim=1).detach()
        #t_box = t_box.cpu()
        #t_score = t_score.cpu().numpy()
        
        t_box = clip_boxes_to_image(t_box, self.im_shape[:-1]).cpu().numpy()
        return t_score, t_box

    def regress_particles(self, img, active_particles, n_particles):

        t_score, t_box = self.regress_single_track(img, active_particles)

        # Remove last one as we need everything from penultimate
        split_indices = np.cumsum(n_particles)[:-1]

        # how much to add to each split to get overall argmax
        amax_ind_offest = np.r_[0, split_indices].astype(np.int32)

        # splitting each particles into a list
        # List because each track can have unequal particles
        # and cannot be made into a np.array
        split_scores = np.split(t_score, split_indices)
        split_pos = np.split(t_box, split_indices)
        split_amax = np.array([np.argmax(i) for i in split_scores])

        best_ind = split_amax + amax_ind_offest

        mean_scores = t_score[best_ind]
        mean_pos = t_box[best_ind]

        return split_scores, split_pos, mean_scores, mean_pos
    
    def filter_regressed_tracks(self, all_score, all_pos, mean_scores, mean_pos):
        existing_ids = [t.id for t in self.tracks]

        f_best_scores = []
        f_all_scores = []
        f_all_pos = []
        f_best_pos = []
        filtered_ids = []

        iter_zip = zip(existing_ids, all_score, all_pos, mean_scores, mean_pos)

        for i, (t_i, as_i, apos_i, bs_i, bpos_i) in enumerate(iter_zip):
            # check area of box
            if check_area(bpos_i, self.im_shape) is False:
                continue

            if bs_i > self.regression_thresh:

                f_all_scores.append(as_i)
                f_best_scores.append(bs_i)
                f_best_pos.append(bpos_i)
                filtered_ids.append(t_i)
                f_all_pos.append(apos_i)

        f_best_scores = torch.tensor(f_best_scores, dtype=torch.float32, device=self.device)
        f_best_pos = torch.tensor(f_best_pos, dtype=torch.float32, device=self.device)
        
        filtered_ids = np.array(filtered_ids)
        f_all_scores = np.array(f_all_scores)
        f_all_pos = np.array(f_all_pos)

        return f_all_scores, f_all_pos, f_best_scores, f_best_pos, filtered_ids

    def _id_filtered_update_active(self, img, nms_ids, nms_val, nms_all_scores, nms_all_pos):
        for (m_id, m_pos, a_score, a_pos) in zip(nms_ids, nms_val, nms_all_scores, nms_all_pos):
            t = self.get_track(m_id)
            t.update_position(m_pos[:4], a_score, a_pos)

        # Copy the signature of matched map
        matched_map = {k : v for k, v in zip(nms_ids, nms_val[:, :4])}

        # Update for matched IDs
        for old_id, new_pos in zip(nms_ids, nms_val[:, :4]):
            cur_track = self.get_track(old_id)
            histogram = compute_histogram(compute_crop(img, new_pos, self.im_shape))
            cur_track.update_track(self.frame_number, hist_vec=histogram)

            return matched_map
    
    def _propose_new_tracks(self, bboxes, det_scores, nms_pos, nms_scores):
        det_pos_gpu = torch.tensor(bboxes, dtype=torch.float32, device=self.device)
        combined_pos = torch.cat([nms_pos, det_pos_gpu])
        combined_scores = torch.cat(
                    [2 * torch.tensor( np.ones(nms_scores.shape[0]), dtype=torch.float32, device=self.device),
                     torch.tensor(     det_scores,                   dtype=torch.float32, device=self.device)])

        keep_det = nms(combined_pos, combined_scores, self.detection_nms_thresh).cpu().numpy().tolist()

        new_ind = [i - len(nms_pos) for i in keep_det if i >= len(nms_pos)]

        new_boxes = det_pos_gpu[new_ind].cpu().numpy()
        new_scores = det_scores[new_ind]

        return new_boxes, new_scores
    
    def appearance_match(self, img, new_boxes, thresh=1.):
        rematch_map = {}
        box_ind = []

        regress_pos = np.asarray([lt.pos for lt in self.inactive_tracks])
        regress_id = [lt.id for lt in self.inactive_tracks]

        regress_cent = compute_centroid(regress_pos)
        box_cent = compute_centroid(boxes)

        range_ar = 2 * np.max(boxes[:, 2:4] - boxes[:, 0:2], axis=1).reshape(-1, 1)

        dist_matrix = cdist(box_cent, regress_cent, metric='cityblock')
        dist_matrix = dist_matrix / range_ar
        dist_cond = dist_matrix < 1.
        dist_matrix = dist_matrix * dist_cond + (1-dist_cond)*1e20

        # Appearance matrix
        box_hist = [compute_histogram(compute_crop(img, b_pos, self.im_shape)).flatten() for b_pos in boxes]
        box_hist = np.array(box_hist)

        regress_hist = np.asarray([lt.hist_vec.flatten() for lt in self.inactive_tracks])

        appearance_mat = matrix_histcmp(box_hist, regress_hist)

        # remove far away appearance info
        appearance_mat = appearance_mat * dist_cond + (1 - dist_cond) * 1e20

        if self.use_reid:
            cost_matrix = (1 - self.lambd) * appearance_mat + self.lambd * dist_matrix
        else:
            cost_matrix = dist_matrix

        max_indexes = munkres_inst.compute(cost_matrix.tolist())
        for row, col in max_indexes:
            if cost_matrix[row][col] <= thresh:
                rematch_map[regress_id[col]] = boxes[row]
                box_ind.append(row)

        return rematch_map, box_ind

    def track_match(self, img, warp_matrix, bboxes, det_scores):
        new_id_map = {}
        regress_matches = {}
        max_id = self.track_num

        prev_ids = [t.id for t in self.tracks]
        inactive_ids = [lt.id for lt in self.inactive_tracks]

        # Align the position of existing particles
        aligned_particles = [align(particle, warp_matrix, self.device) for particle in [t.get_particles() for t in self.tracks]]
        _ = [t.align_particles(a_p) for (t, a_p) in zip(self.tracks, aligned_particles)]

        # Align the positions of inactive track
        for lt in self.inactive_tracks:
            lt.pos = align([lt.pos], warp_matrix, self.device)[0]

        # Check scores of active track
        active_particles = [torch.tensor(t.predict_particles(), dtype=torch.float32, device=self.device) for t in self.tracks]

        t_n_particles = [t.roiPF.created_particles for t in self.tracks]
        t_active_particles = torch.cat(active_particles, axis=0)

        all_scores, all_pos, mean_scores, mean_pos = self.regress_particles(t_active_particles, t_n_particles)
        f_all_scores, f_all_pos, f_best_scores, f_best_pos, match_ids = self.filter_regressed_tracks(all_scores, all_pos, mean_scores, mean_pos)

        # PERFORM NMS for ACTIVE TRACKS UPDATING
        keep_tracks = nms(f_best_pos, f_best_scores, self.regression_nms_thresh).detach().cpu().numpy()

        nms_ids = match_ids[keep_tracks] # np.ndarray
        nms_pos = f_best_pos[keep_tracks] # torch.array
        nms_scores = f_best_scores[keep_tracks] # torch.array

        nms_val = np.c_[nms_pos.detach().cpu().numpy(), nms_scores.detach().cpu().numpy()] # np.ndarray

        nms_all_scores = f_all_scores[keep_tracks] # np.ndarray
        nms_all_pos = f_all_pos[keep_tracks] # np.ndarray

        # ACTIVE TRACKS UPDATING
        matched_map = self._id_filtered_update_active(img, nms_ids, nms_val, nms_all_scores, nms_all_pos)

        # NMS to PROPOSE NEW TRACKS
        new_boxes, new_scores = self._propose_new_tracks(bboxes, det_scores, nms_pos, nms_scores)

        # TRY TO REACTIVATE OLD TRACKS
        if len(inactive_ids) > 0 and len(new_boxes) > 0:
            regress_matches, matched_ind = self.appearance_match(img, new_boxes)
            new_boxes = [v for i, v in enumerate(new_boxes) if i not in matched_ind]

        # START NEW TRACKS
        for new_b in new_boxes:
            max_id += 1
            new_id_map[max_id] = new_b

        # REACTIVATE OLD TRACK
        for old_id, new_pos in regress_matches.items():
            ls_t = self.get_lost_track(old_id)

            self.track_found_again(ls_t)
            histogram = compute_histogram(compute_crop(img, new_pos, self.im_shape))

            ls_t.update_position(new_pos, all_scores=None, all_pos=None)
            ls_t.update_track(self.frame_number, hist_vec=histogram, rematch=True)

        # DEACTIVATE LOST TRACKS
        lost_ids = list(set(prev_ids) - set(nms_ids))
        lost_tracks = [self.get_track(i) for i in lost_ids]
        self.tracks_to_inactive(lost_tracks)

        matched_map = {**matched_map, **regress_matches}
        return matched_map, new_id_map

    def update(self, img, new_id_map):

        obj_ids = list(new_id_map.keys())
        n_dets = len(obj_ids)

        for nd_i, (cur_id, box_loc) in enumerate(new_id_map.items()):
            histogram = compute_histogram(compute_crop(img, box_loc, self.im_shape))
 
            n_track = Track(track_id=cur_id, time_stamp=self.frame_number,
                            pos=box_loc,
                            count_inactive=0,
                            inactive_patience=self.inactive_patience,
                            im_shape=self.im_shape, hist_vec=histogram,
                            max_particles=self.n_particles)

            self.tracks.append(n_track)
        self.track_num += n_dets

        return self.tracks, self.track_num

    def forward(self, img, bboxes, scores):
        
        # Motion compensation
        if self.frame_number > 1:
            warp_matrix = self.compute_warp_matrix(img, self.last_image)
        
        # Track matching
        matched_map = None
        matched_mask = np.zeros(img.shape[:-1]).astype(np.int16)
        if len(self.tracks) > 0:
            
            # Manage inactive tracks (remove or keep)
            self.inactive_tracks = self.regress_lost_track()

            prev_ids = [t.id for t in self.tracks]
            inactive_ids = [lt.id for lt in self.inactive_tracks]
            
            # Search for a match or create a new one
            matched_map, new_id_map = self.track_match(warp_matrix, bboxes, scores)

            self.inactive_tracks = [t for t in self.inactive_tracks if t.count_inactive <= self.inactive_patience]

        else:
            # Initialize tracks
            init_ids = list(range(1, len(bboxes) + 1))
            new_id_map = {n_id : n_pos for (n_id, n_pos) in zip(init_ids, bboxes)}
            matched_map = new_id_map
        
        # Update tracking data
        self.tracks, self.track_num = self.update(img, new_id_map)

        # Generate outputs
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            position = t.pos
            self.results[t.id][self.frame_number] = np.concatenate([position, np.array([1.])])
        self.last_image = img

        return self.results, matched_map, self.frame_number # matched_map in orther to generate and the current processed frame outside
    
    __call__ = forward

    def get_results(self):
        return self.results
    
