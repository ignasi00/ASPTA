
from copy import deepcopy
import cv2
import numpy as np

from .tracker import Tracker


def build_compute_warp_matrix(number_of_iterations, termination_eps, warp_mode):
    def compute_warp_matrix(img, last_image):
        im1 = deepcopy(last_image)
        im2 = deepcopy(img)

        im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

        warp_matrix = torch.from_numpy(warp_matrix)
        # motion_vector = np.tile(warp_matrix[:, 2],2)

        return warp_matrix
    return compute_warp_matrix

def build_tracker(im_shape, reg_model, change_rpn, apply_reg_model):
    compute_warp_matrix = build_compute_warp_matrix(number_of_iterations=100, termination_eps=0.00001, warp_mode=cv2.MOTION_EUCLIDEAN)
    tracker = Tracker(im_shape, compute_warp_matrix, reg_model, change_rpn, apply_reg_model)
    return tracker
