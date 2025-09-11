'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2025-09-11 11:01:17
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-11 13:06:39
FilePath: /A2PM-MESA/point_matchers/dust3r.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any, List, Optional
from loguru import logger
import random
import PIL
import torchvision.transforms as tvf
from .abstract_point_matcher import AbstractPointMatcher

import sys
sys.path.append("/opt/data/private/A2PM-git/A2PM-MESA/point_matchers/mast3r/dust3r")  # noqa
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

class Dust3rMatcher(AbstractPointMatcher):
    """DUSt3R Matcher Warper
    """

    def __init__(
        self, 
        weight_path: str,
        device: str = 'cuda',
        fixed_shape=512, # we fix the input shape to 512x512 for dust3r #FIXME: should be configurable
        ) -> None:
        """
        """
        self.ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.matcher = AsymmetricCroCo3DStereo.from_pretrained(weight_path).to(device)
        self.matcher.eval().cuda()
        self.fixed_shape = fixed_shape
        self.device = device
        self._name = "Dust3rMatcher"
    
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

        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.matcher, self.device, batch_size=1)


        # find 2D-2D matches between the two images
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=100, schedule='cosine', lr=0.01)

        # retrieve useful values from scene:
        imgs = scene.imgs
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        
        try:
            reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)

            matches_im1 = pts2d_list[1][reciprocal_in_P2]
            matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
        except Exception as e:
            # may fail when no matches found
            logger.warning(f"find reciprocal matches failed: {e}")
            matches_im0 = np.zeros((0, 2))
            matches_im1 = np.zeros((0, 2))
            num_matches = 0
        
        logger.info(f"Found {matches_im0.shape[0]} matches between the two images")

        self.matched_corrs = self.convert_matches2list(matches_im0, matches_im1)

        if len(self.matched_corrs) > self.match_num:
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
        
        self.corrs = self.matched_corrs # used in SGAM

        return self.matched_corrs