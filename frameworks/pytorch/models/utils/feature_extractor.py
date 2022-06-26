""" code from https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904 also in https://gist.github.com/fkodom/27ed045c9051a39102e8bcf4ce31df76#file-feature_extractor_hook-py """

import torch
from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output): # omit the model (_ == self <- the hooked one) and and the input (__ == x)
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x) # TODO: if it fails because the output, make veriosn with a try-except "WhateverException" as e: self.exception_handler(e) (default functional print)
        output = [self._features[layer_id] for layer_id in self.layers]
        return output #self._features[self.layers[0]], *[self._features[layer_id] for layer_id in self.layers[1:]]
