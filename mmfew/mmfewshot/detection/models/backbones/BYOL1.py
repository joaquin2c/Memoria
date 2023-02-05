# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from BYOL.models import *
from mmdet.models.builder import BACKBONES
from torch import Tensor


@BACKBONES.register_module()
class BYOLDRAW(BYOL):

    def __init__(self,
            net,
            image_size,
            hidden_layer = -2,) -> None:
        super().__init__(
            net,
            image_size,
            hidden_layer = -2,
        )


    def forward(
        self,
        x
    ):

        return self.online_encoder(x, return_projection = False)

