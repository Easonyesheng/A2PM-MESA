'''
Author: EasonZhang
Date: 2024-10-19 21:54:21
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-13 11:19:25
FilePath: /SA2M/hydra-mesa/dataloader/demo_pair_loader.py
Description: data loader for demo pair

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''
import os
import cv2
from loguru import logger
from typing import List, Optional, Any, Tuple
import numpy as np

from .abstract_dataloader import AbstractDataloader


class DemoPairLoader(AbstractDataloader):
    """ dataloader for demo pair
    """

    def __init__(self, 
        root_path, 
        image_name0, 
        image_name1,
        color_folder,
        color_post,
        sem_folder,
        sem_post,
        intrin_folder="",
        intrin_post="",
        scene_name="demo", # not used
        ) -> None:
        super().__init__(root_path, scene_name, image_name0, image_name1)
        self.color_folder = color_folder
        self.color_post = color_post
        self.sem_folder = sem_folder
        self.sem_post = sem_post
        self.intrin_folder = intrin_folder
        self.intrin_post = intrin_post
        self._name = 'DemoPairLoader'

        self._path_assemble()

    def _path_assemble(self) -> None:
        """ assemble path
        """
        self.img0_path = os.path.join(self.root_path, self.color_folder, self.image_name0 + f".{self.color_post}")
        self.img1_path = os.path.join(self.root_path, self.color_folder, self.image_name1 + f".{self.color_post}")

        self.sem0_path = os.path.join(self.root_path, self.sem_folder, self.image_name0 + f".{self.sem_post}")
        self.sem1_path = os.path.join(self.root_path, self.sem_folder, self.image_name1 + f".{self.sem_post}")

        self.K0_path = os.path.join(self.root_path, self.intrin_folder, self.image_name0 + f".{self.intrin_post}")
        self.K1_path = os.path.join(self.root_path, self.intrin_folder, self.image_name1 + f".{self.intrin_post}")

    def load_Ks(self, scale0, scale1):
        """ load Ks
        """
        if self.intrin_folder == "":
            logger.warning(f"no intrinsic parameter provided, should avoid using egam and use gam.")
            return None, None
        
        K0 = open(self.K0_path, 'r').readlines()
        K0 = K0[0].strip().split(' ')
        fx0, fy0, cx0, cy0 = [float(x) for x in K0]
        if scale0 is List:
            scale0_x, scale0_y = scale0
        else:
            scale0_x = scale0
            scale0_y = scale0
        fx0 *= scale0_x
        fy0 *= scale0_y
        cx0 *= scale0_x
        cy0 *= scale0_y
        K0 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]])
        

        K1 = open(self.K1_path, 'r').readlines()
        K1 = K1[0].strip().split(' ')
        fx1, fy1, cx1, cy1 = [float(x) for x in K1]
        if scale1 is List:
            scale1_x, scale1_y = scale1
        else:
            scale1_x = scale1
            scale1_y = scale1
        fx1 *= scale1_x
        fy1 *= scale1_y
        cx1 *= scale1_x
        cy1 *= scale1_y
        K1 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])

        return K0, K1
    


    def load_depths(self):
        """ load depth data
        """
        raise NotImplementedError

    def load_poses(self):
        """ load pose
        Note: no GT pose provided in demo pair, can be implemented if you have
        """
        logger.warning(f"load pose not implemented")
        return None, None

    def get_eval_info(self):
        """ get eval info
        """
        raise NotImplementedError
    
    def get_sem_paths(self):
        """ get semantic paths
        """
        return self.sem0_path, self.sem1_path

    def load_semantics(self) -> Tuple[np.ndarray, np.ndarray]:
        """ load semantic info
        """
        if self.sem_folder == "":
            logger.warning(f"no semantic segmentation provided, return None")
            return None, None

        sem0 = np.load(self.sem0_path, allow_pickle=True)
        sem1 = np.load(self.sem1_path, allow_pickle=True)
        return sem0, sem1

    def tune_corrs_size_to_eval(self):
        """ tune corrs size to eval
        """
        raise NotImplementedError