
import torch
import torch.nn as nn


class UniformAnchorGenerator(nn.Module):
    # Based on Torchvision's AnchorGenerator implementation

    # Generate uniform distributed anchors (a.k.a. base bboxes: 2D windows at fixed positions) over images.
    # To get pyTorch AnchorGenerator results, use the number of feature_map pixels and the original image aspect ratio.

    # Usually the implementations generate one anchor position (for all the base anchors) per feature map pixel.
    # This means that number of anchors depen on the input image size, this may collide with the Heads, that needs the number of anchors at construction.
    # As each channel of the RPNHead has an associaded anchor, my intuition says that it (somehow) learns which anchors are more useful for each relative location.
    # But, as the backbone plus RPNHead initially were fully convolutional, it shouldn't happen (each pixel is procesed equaly).
    # It is possible that the full system generate pseudo-paralel-pipes through the channels to get position information at the pixel level (filtering anchors by location)...
    # If my intuition is correct, the anchors must keep an independent "anchor aspect ratio".

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
    
    def _slide_windows(self, base_windows, num_anchors, anchor_aspect_ratio):
        # On CPU; TODO: study if it is better to precompute anchors or do it on GPU at runtime
        
        # Height:
        num_rows = torch.sqrt(num_anchors * anchor_aspect_ratio).round()
        # Width:
        num_cols = torch.sqrt(num_anchors / anchor_aspect_ratio).round()

        shifts_x = torch.arange(0, num_cols) / num_cols
        shifts_y = torch.arange(0, num_rows) / num_rows
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        base_anchors = (shifts.view(-1, 1, 4) + base_windows.view(1, -1, 4)).reshape(-1, 4)
        return base_anchors

    def __init__(self, scales, aspect_ratios, num_anchors, anchor_aspect_ratio=2/3):
        # Aspect ratios as height / width
        super(UniformAnchorGenerator, self).__init__()

        base_windows = self._generate_windows(scales, aspect_ratios)
        self.precomputed_anchors = self._slide_windows(base_windows, num_anchors, anchor_aspect_ratio)
    
    def _scale_anchors(self, precomputed_anchors, image_list):
        # To device and scale
        # images like [..., H, W]
        # anchors like [B, #A, 4]

        # TODO: Does precomputed_anchors.repeat keep the new tensor on GPU?
        precomputed_anchors = precomputed_anchors.to(image_list[-1].device)

        image_sizes = torch.tensor([img.shape[-2:] for img in image_list], device=image_list[-1].device).repeat(1, 2).unsqueeze(1)
        anchors = precomputed_anchors.repeat(len(image_list), *([1] * len(precomputed_anchors.shape)))

        anchors = anchors * image_sizes

        return anchors

    def forward(self, image_list, features=None):
        # features is unused but it allows compatibility with torchvision, maybe some anchor generator could use them
        anchors = _scale_anchors(self.precomputed_anchors, image_list)
        # Torchvision returns a list of B elements like [#A, 4], currently I return a tensor like [B, #A, 4]
        return anchors

class PyramidUniformAnchorGenerator(nn.Module):

    def __init__(self, scales, aspect_ratios, num_anchors_list, anchor_aspect_ratio=2/3):
        # TODO: Maybe scales and aspect_ratios could be list of lists, different feature maps using diferent values.
        super(PyramidAnchorGenerator, self).__init__()
        self.anchor_generators = nn.ModuleList([UniformAnchorGenerator(scales, aspect_ratios, num_anchors, anchor_aspect_ratio) for num_anchors in num_anchors_list])

    def forward(self, image_list, features=None):
        # features is unused but it allows compatibility with torchvision
        anchors = []
        for anchor_generator in self.anchor_generators:
            anchors.append(anchor_generator(image_list))
        return torch.cat(anchors, dim=1) # Torchvision uses a list of B elements instead of a tensor with the first dimension being B

