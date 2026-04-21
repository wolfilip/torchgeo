# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pretrained Distillation for EO (DEO) model implementation."""

from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torchvision import models as torchvision_models
from torchvision.models._api import Weights, WeightsEnum
from torchvision.ops.misc import Permute


class DEO(nn.Module):
    """Pretrained Distillation for EO (DEO) model implementation.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2602.19863

    .. versionadded:: 0.9
    """

    def __init__(self, model: str = 'swin_b') -> None:
        """Initialise DEO model.

        Args:
            model: backbone type (for now swin_b).
        """
        super().__init__()

        # initialize the backbone
        self.feat_extr = torchvision_models.__dict__[model]()
        del self.feat_extr.features[0]
        # Conv layers for Swin
        norm_layer_ms = partial(nn.LayerNorm, eps=1e-5)
        norm_layer_rgb = partial(nn.LayerNorm, eps=1e-5)
        self.feat_extr.conv_ms = nn.Sequential(
            nn.Conv2d(
                10,
                self.feat_extr.features[0][0].norm1.normalized_shape[0],
                kernel_size=(4, 4),
                stride=(4, 4),
            ),
            Permute([0, 2, 3, 1]),
            norm_layer_ms(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
        )
        self.feat_extr.conv_rgb = nn.Sequential(
            nn.Conv2d(
                3,
                self.feat_extr.features[0][0].norm1.normalized_shape[0],
                kernel_size=(4, 4),
                stride=(4, 4),
            ),
            Permute([0, 2, 3, 1]),
            norm_layer_rgb(self.feat_extr.features[0][0].norm1.normalized_shape[0]),
        )

    def forward_swin(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get multi-stage swin features.

        Args:
            x: input image tensor (b, c, h, w).

        Returns:
            list of swin feature tensors list[(b, c, h', w')].
        """
        features = []
        # apply the appropriate conv layer based on the number of input channels
        if x.shape[1] == 10:
            x = self.feat_extr.conv_ms(x)
        else:
            x = self.feat_extr.conv_rgb(x)

        # extract intermediate swin layers
        for i, layer in enumerate(self.feat_extr.features):
            x = layer(x)
            if i in [0, 2, 4, 6]:
                features.append(x)

        return features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward function for DEO."""
        features = self.forward_swin(x)

        return features


class DEO_Weights(WeightsEnum):
    """DEO base model weights.

    .. versionadded:: 0.9
    """

    DEO_SWIN = Weights(
        url='https://huggingface.co/SolaireTheSun/DEO/resolve/main/DEO_swin_b.pth',
        transforms=nn.Identity(),
        meta={
            'dataset': 'fMoW',
            'model': 'swin',
            'publication': 'https://arxiv.org/abs/2602.19863',
            'repo': 'https://github.com/wolfilip/DEO-FM',
            'ssl_method': 'deo',
        },
    )


def deo_base(weights: DEO_Weights | None = None, *args: Any, **kwargs: Any) -> DEO:
    """DEO Swin model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2602.19863

    Args:
        weights: Pretrained weights to load.
        *args: Additional arguments to pass to :class:`DEO`.
        **kwargs: Additional keyword arguments to pass to :class:`DEO`.

    Returns:
        DEO Swin model.
    """
    model = DEO(*args, **kwargs)

    if weights:
        state_dict = weights.get_state_dict(progress=True)
        model.load_state_dict(state_dict, strict=False)

    return model
