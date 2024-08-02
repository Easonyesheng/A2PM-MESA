

import os
from copy import deepcopy

import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.cm as cm
from loguru import logger
import random
import torch.nn.functional as F

from .DKM.dkm import DKMv3_outdoor, DKMv3_indoor
from .abstract_point_matcher import AbstractPointMatcher


class DKMMatcher(AbstractPointMatcher):
    """
    """
    def __init__(self,
        dataset_name,
        weights,
        ):

        super().__init__()
    
        self._name = "DKM"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.datasetName = dataset_name
        if self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "ETH3D":
            self.matcher = DKMv3_indoor(device=device, path_to_weights=weights)
        elif self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
            self.matcher = DKMv3_outdoor(device=device, path_to_weights=weights)
        else:
            raise NotImplementedError

    def match(self, img0, img1, mask0=None, mask1=None):
        """
        Args:
            img0, img1: cv img
            NOTE the image size is not in consideration, as the DKM outputs normalized coordinates
        """

        # turn to PIL image
        img0 = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        # scale to orignal size
        w1, h1 = img0.size
        w2, h2 = img1.size

        assert w1 == w2 and h1 == h2, "DKM only support same size image"

        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
            self.matcher.h_resized = h1
            self.matcher.w_resized = w1 # the crop area size
            self.matcher.upsample_preds = True
            self.matcher.upsample_res = (1152, 1536)
            self.matcher.use_soft_mutual_nearest_neighbours = False
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI":
            self.matcher.h_resized = h1
            self.matcher.w_resized = w1
            self.matcher.upsample_preds = False
        else:
            raise NotImplementedError

        # match
        # import time
        # start = time.time()
        dense_matches, dense_certainty = self.matcher.match(img0, img1)
        # logger.info(f"DKM match time: {time.time() - start}")
        # sample
        sparse_matches,_ = self.matcher.sample(
            dense_matches, dense_certainty, 5000
        )

        kpts1 = sparse_matches[:, :2]
        kpts1 = (
            torch.stack(
                (
                    w1 * (kpts1[:, 0] + 1) / 2,
                    h1 * (kpts1[:, 1] + 1) / 2,
                ),
                axis=-1,
            )
        )
        kpts2 = sparse_matches[:, 2:]
        kpts2 = (
            torch.stack(
                (
                    w2 * (kpts2[:, 0] + 1) / 2,
                    h2 * (kpts2[:, 1] + 1) / 2,
                ),
                axis=-1,
            )
        )

        # put kpts into numpy
        kpts1 = kpts1.cpu().numpy()
        kpts2 = kpts2.cpu().numpy()

        self.matched_corrs = self.convert_matches2list(kpts1, kpts2)

        if len(self.matched_corrs) > self.match_num:
            logger.info(f"sample {self.match_num} corrs from {len(self.matched_corrs)} corrs")
            self.matched_corrs = random.sample(self.matched_corrs, self.match_num)
        
        self.corrs = self.matched_corrs # used in SGAM

        return self.matched_corrs