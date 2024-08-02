'''
Author: EasonZhang
Date: 2024-06-12 21:03:00
LastEditors: EasonZhang
LastEditTime: 2024-07-25 23:16:34
FilePath: /SA2M/hydra-mesa/point_matchers/aspanformer.py
Description: aspanformer point matcher

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Any, List, Optional
from loguru import logger
import random

from .ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer
from .ASpanFormer.src.config.default import get_cfg_defaults
from .ASpanFormer.src.utils.misc import lower_config

from .abstract_point_matcher import AbstractPointMatcher


class ASpanMatcher(AbstractPointMatcher):
    """ASpanFormer Matcher Warper
    """

    def __init__(
        self, 
        config_path: str,
        weights: str,
        dataset_name: str,
        ) -> None:
        super().__init__()
        self._name = "ASpanMatcher"
        _default_cfg = get_cfg_defaults()
        # indoor
        if dataset_name == "ScanNet" or dataset_name == "Matterport3D" or dataset_name == "KITTI" or dataset_name == "ETH3D":
            main_cfg_path = f"{config_path}/aspan/indoor/aspan_test.py"
            data_config = f"{config_path}/data/scannet_test_1500.py"
        # outdoor
        elif dataset_name == "MegaDepth" or dataset_name == "YFCC":
            main_cfg_path = f"{config_path}/aspan/outdoor/aspan_test.py"
            data_config = f"{config_path}/data/megadepth_test_1500.py"
        else:
            raise NotImplementedError(f"dataset {dataset_name} not implemented")

        _default_cfg.merge_from_file(main_cfg_path)
        _default_cfg.merge_from_file(data_config)

        _default_cfg = lower_config(_default_cfg)
        matcher = ASpanFormer(config=_default_cfg['aspan'])
        matcher.load_state_dict(torch.load(weights)["state_dict"], strict=False)

        self.matcher = matcher.eval().cuda()
    
    
    def match(self, img0: np.ndarray, img1: np.ndarray, mask0: Optional[Any]=None, mask1: Optional[Any]=None) -> List[List[float]]:
        """
        Returns:
            matched_corrs: list of [u0, v0, u1, v1]
        """
        if len(img0.shape) == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        

        logger.info(f"img shape is {img0.shape}")

        img_tensor0 = torch.from_numpy(img0 / 255.)[None][None].cuda().float()
        img_tensor1 = torch.from_numpy(img1 / 255.)[None][None].cuda().float()

        batch = {"image0": img_tensor0, "image1": img_tensor1}

        if mask0 is not None and mask1 is not None:
            mask0 = torch.from_numpy(mask0).cuda()  # type: ignore
            mask1 = torch.from_numpy(mask1).cuda() # type: ignore
            [ts_mask_0, ts_mask_1] = F.interpolate(
                torch.stack([mask0, mask1], dim=0)[None].float(),
                scale_factor=0.125,
                mode='nearest',
                recompute_scale_factor=False
            )[0].bool().to("cuda")
            batch.update({'mask0': ts_mask_0.unsqueeze(0), 'mask1': ts_mask_1.unsqueeze(0)})


        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy() # Nx2
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            m_bids = batch['m_bids'].cpu().numpy()

        
        self.matched_corrs = self.convert_matches2list(mkpts0, mkpts1)

        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)
        
        self.corrs = self.matched_corrs # used in SGAM

        return self.matched_corrs
    
    def get_coarse_mkpts_c(self, area0, area1):
        """ match region and only use coarse level to get coarse mkpts
        """
        assert area0.shape == area1.shape

        if len(area0.shape) == 3:
            area0 = cv2.cvtColor(area0, cv2.COLOR_BGR2GRAY)
            area1 = cv2.cvtColor(area1, cv2.COLOR_BGR2GRAY)
        
        area0_tensor = torch.from_numpy(area0 / 255.)[None][None].cuda().float()
        area1_tensor = torch.from_numpy(area1 / 255.)[None][None].cuda().float()

        batch = {"image0": area0_tensor, "image1": area1_tensor}

        with torch.no_grad():
            self.matcher.coarse_match_mkpts_c(batch)

        conf_matrix = batch["conf_matrix"]
        mkpts0_c = batch["mkpts0_c"]
        mkpts1_c = batch["mkpts1_c"]
        mconf = batch["mconf"]

        return mkpts0_c, mkpts1_c, mconf, conf_matrix