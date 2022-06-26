
import torch
import torch.nn as nn

from .utils import conv1x1_bn_relu, conv3x3_bn_relu


def _conv_bn1X1_leaky(in_planes, out_planes, stride=1):
    leaky = 0.1 if out_planes <= 64 else 0
    return conv1x1_bn_relu(in_planes, out_planes, stride=stride, leaky=leaky)

def _conv_bn3X3_leaky(in_planes, out_planes, stride=1, padding=1):
    leaky = 0.1 if out_planes <= 64 else 0
    return conv3x3_bn_relu(in_planes, out_planes, stride=stride, padding=padding, leaky=leaky)

def _nearest_interpolator(intput_, out_size):
    return F.interpolate(input_, size=out_size, mode="nearest")

class FPN(nn.Module):

    def __init__(self, backbone, in_channels_list, out_channels, channels_funnel=None, interpolator=None, interpolator_filter=None, extra_blocks=None):
        super(FPN, self).__init__()
        
        self.backbone = backbone

        channels_funnel = channels_funnel or _conv_bn1X1_leaky
        self.preprocessing = nn.ModuleList([channels_funnel(in_channels, out_channels) for in_channels in in_channels_list])

        self.interpolator = interpolator or _nearest_interpolator

        interpolator_filter = interpolator_filter or _conv_bn3X3_leaky
        self.postprocessing = nn.ModuleList([interpolator_filter(out_channels, out_channels) for _ in in_channels_list])

        self.extra_blocks = extra_blocks

    def _merge(self, output_big, output_small, postprocess):
        output_small_up = self.interpolator(output_small, [output_big.size(2), output_big.size(3)])
        out = output_big + output_small_up
        # They say that postprocessing reduce the aliasing
        return postprocess(out)

    def forward(self, input_):

        # The backbone already has a Feature Extractor which return a list (lateral connections) for different levels
        backbone_output_list = self.backbone(input_)

        # Preprocessing in order to have the same number of channels at each level
        output_list = [preprocess(output) for preprocess, output in zip(self.preprocessing, backbone_output_list)]
        smallest_output = self.postprocessing[-1](output_list[-1])

        # Smaller levels are aggregated with bigger levels (the smallest was only processed further at the previous step)
        output_list = [self._merge(self, output_big, output_small, postprocess) for output_big, output_small, postprocess in zip(output_list[:-1], output_list[1:], self.postprocessing[:-1])]
        output_list.append(smallest_output)

        if self.extra_blocks is not None:
            results = self.extra_blocks(output_list, backbone_output_list)
            if isinstance(results, Tensor):
                output_list.append(results)
            else:
                output_list.extend(results)

        return output_list
