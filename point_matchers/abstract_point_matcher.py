'''
Author: EasonZhang
Date: 2024-06-12 20:35:29
LastEditors: EasonZhang
LastEditTime: 2024-06-27 21:20:01
FilePath: /SA2M/hydra-mesa/point_matchers/abstract_point_matcher.py
Description: abstrat point matcher class

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import sys
import numpy as np
from typing import List, Optional, Any
import abc


class AbstractPointMatcher(abc.ABC):
    """
    Abstract class for point matcher
    """
    def name(self) -> str:
        """
        Return the name of the matcher
        """
        try:
            return self._name
        except AttributeError:
            raise NotImplementedError("Not use the abstract class directly")

    def set_corr_num_init(self, num: int):
        """
        Set the number of correspondences to be matched
        """
        self.match_num = num

    @abc.abstractmethod
    def match(self, img0: np.ndarray, img1: np.ndarray, mask0: Optional[Any]=None, mask1: Optional[Any]=None) -> List[List[float]]:
        """
        Match two images and return the correspondences
        Returns:
            self.matched_corrs
        """
        pass
   
    def return_matches(self):
        """"""
        return self.matched_corrs

    @staticmethod
    def convert_matches2list(mkpts0, mkpts1) -> List[List[float]]:
        """
        Args:
            mkpts0/1: np.ndarray Nx2
        Returns:
            matches: list [[corr]s]
        """
        matches = []

        assert mkpts0.shape == mkpts1.shape, f"different shape: {mkpts0.shape} != {mkpts1.shape}"

        for i in range(mkpts0.shape[0]):
            u0, v0 = mkpts0[i,0], mkpts0[i,1]
            u1, v1 = mkpts1[i,0], mkpts1[i,1]

            matches.append([u0, v0, u1, v1])
        
        return matches

    