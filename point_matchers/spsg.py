'''
Author: EasonZhang
Date: 2024-06-13 22:05:13
LastEditors: EasonZhang
LastEditTime: 2024-06-29 11:03:39
FilePath: /SA2M/hydra-mesa/point_matchers/spsg.py
Description: SuperPoint + SuperGlue
    

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import os
from copy import deepcopy

import torch
import cv2
import numpy as np
from loguru import logger
import random
import torch.nn.functional as F

from .abstract_point_matcher import AbstractPointMatcher
from .SuperGluePretrainedNetwork.models.matching import Matching

class SPSGMatcher(AbstractPointMatcher):
    """
    Specific:
        configs["weights"]: indoor or outdoor
    """

    def __init__(self,
        weights,
        dataset_name="ScanNet"
        ):
        super().__init__()

        """specific"""
        self._name = "SPSG"
        self.SG_weights = weights

        # use superglue recommended settings
        # indoor
        dataset = dataset_name
        if dataset == "ScanNet" or dataset == "Matterport3D" or dataset == "KITTI" or dataset == "ETH3D": 
            config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 1024,
                },
                'superglue': {
                    'weights': self.SG_weights,
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2,
                }
            }

        elif dataset == "MegaDepth" or dataset == "YFCC":
            config = {
                'superpoint': {
                    'nms_radius': 4,
                    'keypoint_threshold': 0.005,
                    'max_keypoints': 1024,
                },
                'superglue': {
                    'weights': self.SG_weights,
                    'sinkhorn_iterations': 20,
                    'match_threshold': 0.2,
                }
            }
        else:
            config = {}
            raise NotImplementedError(f"dataset {dataset} not implemented")

        # print(f"detector_config: {detector_config}")
        self.matcher = Matching(config).eval().to('cuda')

    def match(self, img0, img1, mask0=None, mask1=None):
        """
        """
        # detect keypoints
        # turn to gray scale
        if len(img0.shape) == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        
        img0 = torch.from_numpy(img0/255.).float()[None][None].to('cuda')
        img1 = torch.from_numpy(img1/255.).float()[None][None].to('cuda')
        pred = self.matcher({'image0': img0, 'image1': img1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        self.matched_corrs = self.convert_matches2list(mkpts0, mkpts1)

        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)

        self.corrs = deepcopy(self.matched_corrs)

        return self.matched_corrs
    
