
import torch
import torch.nn as nn


class AnchorGenerator(nn.Module):
    # Based on Torchvision's AnchorGenerator implementation # TODO: finish

    # The AnchorGenerator generates anchors (a.k.a. base bboxes: 2D windows at fixed positions) over all the pixels (locations) of a feature map.
    # The RPNHead generates a score and correction for each 2D window (num_anchors_per_location windows) at each location (num_anchors).

    def _generate_windows(self, scales, aspect_ratios):
        # zero-centered and area 1 windows on CPU
        scales = torch.as_tensor(scales)
        aspect_ratios = torch.as_tensor(aspect_ratios)

        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_windows = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_windows.round()

    def __init__(self, scales, aspect_ratios):
        # Aspect ratios as height / width
        super(AnchorGenerator, self).__init__()

        self.base_windows = self._generate_windows(scales, aspect_ratios)
    
    def _slide_and_scale_windows(self):
        pass

    def forward(self, image_list, features):
        pass

