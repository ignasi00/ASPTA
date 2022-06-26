
import torch

from ASPTA_utils.build_model import build_model, adapt_state_dict


# I/O VARIABLES OR HYPERPARAMETER I CAN TOUCH
PRETRAINED_PATH = '/mnt/c/Users/Ignasi/Downloads/FT_R50_epoch_24.pth'


# BUILD DETECTOR; return class_logits, box_regression, proposals # before filtering and postprocessing
model = build_model()
state_dict = torch.load('/mnt/c/Users/Ignasi/Downloads/FT_R50_epoch_24.pth', map_location=torch.device('cpu'))
state_dict = adapt_state_dict(state_dict)
model.load_state_dict(state_dict, strict=False)




