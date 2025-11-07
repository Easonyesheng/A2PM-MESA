'''
Author: EasonZhang
Date: 2024-06-19 22:34:22
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-07 16:22:44
FilePath: /SA2M/hydra-mesa/geo_area_matchers/abstract_gam.py
Description: abstract geo area matcher for post-processing

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import numpy as np
import abc
import os
from loguru import logger

from utils.common import test_dir_if_not_create

class AbstractGeoAreaMatcher(abc.ABC):
    def __init__(self) -> None:
        self.initialized = False
        pass

    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def init_dataloader(self, dataloader):
        raise NotImplementedError

    @abc.abstractmethod
    def load_point_matcher(self, point_matcher):
        raise NotImplementedError

    @abc.abstractmethod
    def load_ori_corrs(self, ori_corrs):
        raise NotImplementedError

    @abc.abstractmethod
    def init_gam(self):
        raise NotImplementedError

    @abc.abstractmethod
    def geo_area_matching_refine(self, matched_areas0, matched_areas1):
        """ Main Func
        Returns:
            alpha_corrs_dict: dict, inside-area corrs under each alpha
            alpha_inlier_idxs_dict: dict, inlier idxs of input areas under each alpha
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doubtful_area_match_predict(self, doubt_match_pairs):
        """
        """
        raise NotImplementedError


    def set_outpath(self, outpath: str):
        """Run after init_dataloader 
        """
        self.out_path = os.path.join(outpath, f"{self.scene_name}_{self.name0}_{self.name1}", 'gam')
        if self.draw_verbose == 1:
            test_dir_if_not_create(self.out_path)
        logger.info(f"GAM Output path set to: {self.out_path}")