
from frameworks.pytorch.models.expressions.deconvolutional_single_shot_detector import DSSD_ProcessingModule
from frameworks.pytorch.models.expressions.feature_pyramid_network import FPN
from frameworks.pytorch.models.expressions.resnet import resnet50
from frameworks.pytorch.models.expressions.single_stage_headless import SSH_Residual_Context
from frameworks.pytorch.models.head_hunter import HH_ContextSensitivePrediction
from frameworks.pytorch.models.utils.feature_extractor import FeatureExtractor


def build_model(num_classes_backbone):
    backbone = resnet50(num_classes_backbone)
    # TODO: pretrained backbone
    backbone = FeatureExtractor(backbone, ['layer1', 'layer2', 'layer3', 'layer4'])
    in_channels_list = [256, 512, 1024, 2048]
    backbone = FPN(backbone, in_channels_list=in_channels_list, out_channeles=256, channels_funnel=None, interpolator=None, interpolator_filter=None)

    preprocessing_context = DSSD_ProcessingModule(256, 1024)
    context_processing = SSH_Residual_Context(1024, 512)
    context_sensitive_module = HH_ContextSensitivePrediction(preprocessing_context, context_processing, 512, 256)

    model = nn.Sequential( # TODO: most likely not Sequential
        backbone,
        context_sensitive_module,
        # Faster RCNN
    )
