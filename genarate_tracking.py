
import torch

from ASPTA_utils.build_model import build_model, adapt_state_dict
from ASPTA_utils.postprocess import build_postprocess, get_refined_detection
from ASPTA_utils.pseudo_dataset import build_image_transform
from ASPTA_utils.pseudo_dataset import SequencesPseudoDataset, SequenceImagesPseudoDataset


# I/O VARIABLES OR HYPERPARAMETER I CAN TOUCH
PRETRAINED_PATH = '/mnt/c/Users/Ignasi/Downloads/FT_R50_epoch_24.pth'
BASE_DIR = '/mnt/c/Users/Ignasi/Downloads/HT21/'
DATASETS = [('train', 'test'), ('train', ), ('test', )][0]

min_size = None
max_size = None
dset_mean_std = [[117, 110, 105], [67.10, 65.45, 66.23]]
image_mean = [i/255. for i in dset_mean_std[0]]
image_std = [i/255. for i in dset_mean_std[1]]



def apply_det_model(one_image_list, model, postprocess):

    assert len(one_image_list) == 1

    class_logits, box_regression, proposals = model(one_image_list)

    original_image_sizes = [one_image_list[0].shape]
    boxes, scores = postprocess(class_logits, box_regression, proposals, one_image_list, original_image_sizes) # original_image_sizes == image_shape 'cause no min_size and max_size on Tracker
    
    boxes, scores = boxes[0], scores[0]
    detections = torch.cat((boxes, torch.unsqueeze(scores, 1)), 1).cpu().numpy()

    refined_det = get_refined_detection(detections, original_image_sizes[0], detection_confidence)
    boxes, det_scores = refined_det[:, :4], np.squeeze(refined_det[:, 4:])

    detections = np.c_[boxes, det_scores]

    return boxes, det_scores, detections


# BUILD DETECTOR; return class_logits, box_regression, proposals # before filtering and postprocessing
det_model = build_model()
state_dict = torch.load('/mnt/c/Users/Ignasi/Downloads/FT_R50_epoch_24.pth', map_location=torch.device('cpu'))
state_dict = adapt_state_dict(state_dict)
det_model.load_state_dict(state_dict, strict=False)
det_model.eval()

# BUILD PRE AND POST PROCESSING FUNCTS
transform = build_image_transform(min_size, max_size, image_mean, image_std)
postprocess = build_postprocess(score_thresh=0, nms_thresh=0, detections_per_img=None)


# Process data, no DataLoader is used due to lack of time
for dset in DATASETS:
    # BUILD SEQUENCE DATASET
    squences_dataset = SequencesPseudoDataset(base_dir, dset='train', start_ind=0)

    for seq_images, im_shape, cam_motion, im_path in squences_dataset:
        images_dataset = SequenceImagesPseudoDataset(seq_images, transform)

        for one_image_batch in images_dataset: # one_image_batch is a tensor [1, C, H, W], it works as a list of 1 element [C, H, W]
            boxes, det_scores, detections = apply_det_model(one_image_batch, det_model, postprocess)
            # TODO: call tracker
