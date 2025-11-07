'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2025-09-10 11:27:08
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-07 15:06:27
FilePath: /A2PM-MESA/point_matchers/mast3r.py
Description: 
'''

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any, List, Optional
from loguru import logger
import random
import PIL

from .abstract_point_matcher import AbstractPointMatcher
import torchvision.transforms as tvf

import sys
sys.path.append("/opt/data/private/A2PM-git/A2PM-MESA/point_matchers/mast3r")  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r

sys.path.append("/opt/data/private/A2PM-git/A2PM-MESA/point_matchers/mast3r/dust3r")  # noqa
from dust3r.inference import inference
from dust3r.utils.image import load_images


class Mast3rMatcher(AbstractPointMatcher):
    """MASt3R Matcher Warper
    """

    def __init__(
        self, 
        weight_path: str,
        device: str = 'cuda',
        fixed_shape=512, # we fix the input shape to 512x512 for mast3r #FIXME: should be configurable
        ) -> None: 
        """
        """
        self.ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.matcher = AsymmetricMASt3R.from_pretrained(weight_path).to(device)
        self.matcher.eval().cuda()
        self.fixed_shape = fixed_shape
        self.device = device
        self._name = "Mast3rMatcher"

    def match(self, img0, img1, mask0: Optional[Any]=None, mask1: Optional[Any]=None) -> List[List[float]]:
        """
        Args:
            img0, img1: np.ndarray, H,W,3, [0,255], uint8
        """

        # convert to PIL image
        if not isinstance(img0, PIL.Image.Image):
            img0 = PIL.Image.fromarray(img0)
        if not isinstance(img1, PIL.Image.Image):
            img1 = PIL.Image.fromarray(img1)
        
        assert img0.size == img1.size, "img0 and img1 should have the same size"
        H, W = img0.size[1], img0.size[0]
        assert H == self.fixed_shape and W == self.fixed_shape, f"img0 and img1 should have the size of {self.fixed_shape}x{self.fixed_shape}, but got {H}x{W}"

        assert img0.mode == 'RGB' and img1.mode == 'RGB', f"img0 and img1 should be RGB image, but got {img0.mode} and {img1.mode}"

        # convert to tensor
        img0_ = self.ImgNorm(img0)[None].cuda()  # 1,3,H,W
        img1_ = self.ImgNorm(img1)[None].cuda()  # 1,3,H,W

        images = [
            dict(img=img0_, true_shape=np.int32([img0.size[::-1]]), idx=0, instance=str(0)),
            dict(img=img1_, true_shape=np.int32([img1.size[::-1]]), idx=1, instance=str(1)),
            ]

        output = inference([tuple(images)], self.matcher, self.device, batch_size=1, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        """ what's inside the pred dict?"""
        # for k in pred1.keys():
        #     logger.debug(f"pred1 key: {k}, shape: {pred1[k].shape if isinstance(pred1[k], torch.Tensor) else 'N/A'}")
        # for k in pred2.keys():
        #     logger.debug(f"pred2 key: {k}, shape: {pred2[k].shape if isinstance(pred2[k], torch.Tensor) else 'N/A'}")
        """
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:88 - pred1 key: pts3d, shape: torch.Size([1, 512, 512, 3])
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:88 - pred1 key: conf, shape: torch.Size([1, 512, 512])
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:88 - pred1 key: desc, shape: torch.Size([1, 512, 512, 24])
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:88 - pred1 key: desc_conf, shape: torch.Size([1, 512, 512])
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:90 - pred2 key: conf, shape: torch.Size([1, 512, 512])
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:90 - pred2 key: desc, shape: torch.Size([1, 512, 512, 24])
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:90 - pred2 key: desc_conf, shape: torch.Size([1, 512, 512])
        2025-11-07 14:47:53.249 | INFO     | point_matchers.mast3r:match:90 - pred2 key: pts3d_in_other_view, shape: torch.Size([1, 512, 512, 3])
        """

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=self.device, dist='dot', block_size=2**13)
        

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        logger.debug(f"matches are in image0 shape: {H0}x{W0}") # 512x512
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        logger.info(f"Found {matches_im0.shape[0]} matches between the two images")

        self.matched_corrs = self.convert_matches2list(matches_im0, matches_im1)

        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)
        
        self.corrs = self.matched_corrs # used in SGAM

        return self.matched_corrs


    def get_coarse_mkpts_c(self, area0, area1):
        """ match region and only use coarse level to get coarse mkpts
        Args:
            area0, area1: np.ndarray, H,W,3, [0,255], uint8
        Returns:
        """
        if not isinstance(area0, PIL.Image.Image):
            area0 = PIL.Image.fromarray(area0)
        if not isinstance(area1, PIL.Image.Image):
            area1 = PIL.Image.fromarray(area1)

        assert area0.size == area1.size, "area0 and area1 should have the same size"
        H, W = area0.size[1], area0.size[0]
        assert H == self.fixed_shape and W == self.fixed_shape, f"area0 and area1 should have the size of {self.fixed_shape}x{self.fixed_shape}, but got {H}x{W}"
        assert area0.mode == 'RGB' and area1.mode == 'RGB', f"area0 and area1 should be RGB image, but got {area0.mode} and {area1.mode}"

