'''
Author: EasonZhang
Date: 2024-06-19 22:43:14
LastEditors: EasonZhang
LastEditTime: 2024-06-28 11:30:23
FilePath: /SA2M/hydra-mesa/area_matchers/abstract_am.py
Description: abstract area matcher for pre-processing

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import os
import numpy as np
import abc

from utils.common import test_dir_if_not_create

class AbstractAreaMatcher(abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def init_dataloader(self, dataloader):
        raise NotImplementedError

    @abc.abstractmethod
    def area_matching(self):
        """ Main Func
        Returns:
            area_match_src: list of [u_min, u_max, v_min, v_max] in src img
            area_match_dst: list of [u_min, u_max, v_min, v_max] in dst img
        """
        raise NotImplementedError

    def set_outpath(self, outpath: str):
        """Run after init_dataloader 
        """
        self.out_path = os.path.join(outpath, f"{self.scene_name}_{self.name0}_{self.name1}", 'am')
        if self.draw_verbose == 1:
            test_dir_if_not_create(self.out_path)