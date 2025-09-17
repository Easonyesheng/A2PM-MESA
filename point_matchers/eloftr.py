'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2025-09-16 13:34:11
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-16 13:48:32
FilePath: /A2PM-MESA/point_matchers/eloftr.py
Description: wrapper of ELoFTR
'''

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any, List, Optional
from loguru import logger
import random
from copy import deepcopy

from .EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter


from .abstract_point_matcher import AbstractPointMatcher

class ELoFTRMatcher(AbstractPointMatcher):
    """ ELoFTR Matcher Warpper
    """

    def __init__(
        self, 
        weights: str,
        ):
        super().__init__()
        self._name = "ELoFTRMatcher"
        
        # Initialize the matcher with default settings
        _default_cfg = deepcopy(full_default_cfg)

        matcher = LoFTR(config=_default_cfg)
        matcher.load_state_dict(torch.load(weights)['state_dict'])
        self.matcher = reparameter(matcher)  # Essential for good performance
        self.matcher = self.matcher.eval().cuda()

    def match(self, img0, img1, mask0=None, mask1=None):
        """
        """
        if len(img0.shape) == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
        w, h = img0.shape[1], img0.shape[0]
        assert img0.shape == img1.shape, "Image shapes are different"
        assert w % 32 == 0 and h % 32 == 0, "Image dimensions must be multiples of 32"

        # Convert to tensors
        img_t0 = torch.from_numpy(img0)[None][None].cuda() / 255.
        img_t1 = torch.from_numpy(img1)[None][None].cuda() / 255.
        batch = {'image0': img_t0, 'image1': img_t1}

        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy() # Nx2
            mkpts1 = batch['mkpts1_f'].cpu().numpy()

        self.matched_corrs = self.convert_matches2list(mkpts0, mkpts1)

        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)

        self.corrs = self.matched_corrs # used in SGAM

        logger.info(f"matched corrs num is {len(self.matched_corrs)}")

        return self.matched_corrs