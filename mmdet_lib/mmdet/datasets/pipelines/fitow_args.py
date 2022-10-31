# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..builder import PIPELINES



@PIPELINES.register_module()
class FitowArgs:
    """
    自定义参数
    """

    def __init__(self,
                 eval_threshold=0.9
                 ):
        self.eval_threshold = eval_threshold

    def __call__(self, results):
        return results

    def __repr__(self):
        return self.__class__.__name__
