'''
Author: EasonZhang
Date: 2024-06-29 11:45:30
LastEditors: EasonZhang
LastEditTime: 2024-07-01 19:46:51
FilePath: /SA2M/hydra-mesa/dataloader/megadepth.py
Description: dataloader for MegaDepth
    Most data of MegaDepth is saved in npz file
    only npz_file_name is needed

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''


import os
import numpy as np
import cv2
from typing import List, Optional, Any, Tuple

from .abstract_dataloader import AbstractDataloader
from utils.load import ( # beyond the current directory, use absolute import, use sys.path.append('..') to add the parent directory in the main script
    load_cv_img_resize, 
    load_cv_depth,
    load_K_txt,
    load_pose_txt,
)

from utils.img_process import load_img_padding_rt_size
from utils.geo import tune_corrs_size_diff

class MegaDepthDataloader(AbstractDataloader):
    """ TODO:
    """

    def __init__(self,
        root_path,
        scene_name, # not work
        image_name0, # specific structure 00xx_x_x_id
        image_name1,
        sem_mode,
        sem_folder,
        sem_post,
        ):
        """
        """
        super().__init__(root_path, scene_name, image_name0, image_name1)

        self._name = 'MegaDepthDataloader'

        self.sem_mode = sem_mode
        assert self.sem_mode in ['SAM']
        self.sem_folder = sem_folder
        self.sem_post = sem_post

        self._path_assemble()
    
    def _path_assemble(self):
        """
        """
        npz_folder = os.path.join(self.root_path, "scene_info_val_1500")
        pair0 = self.image_name0 # npz_file_name_id, e.g. 0000_0.1_0.2_id 
        pair1 = self.image_name1 # npz_file_name has "_id" in the end
        # get the npz file name
        pair0_npz_name = "_".join(pair0.split("_")[:-1])
        pair1_npz_name = "_".join(pair1.split("_")[:-1])
        assert pair0_npz_name == pair1_npz_name, "pair0 and pair1 should be in the same npz file"
        id0 = (pair0.split("_")[-1])
        id1 = (pair1.split("_")[-1])

        npz_path = os.path.join(npz_folder, pair0_npz_name+".npz")

        npz_data = np.load(npz_path, allow_pickle=True)

        # get the image path
        img_path0 = npz_data["image_paths"][int(id0)]
        img_path1 = npz_data["image_paths"][int(id1)]
        img_folder0 = img_path0.split("/")[1]
        img_folder1 = img_path1.split("/")[1]
        img_name0 = img_path0.split("/")[-1].split(".")[0]
        img_name1 = img_path1.split("/")[-1].split(".")[0]

        if self.sem_mode == "SAM":
            sem_path = self.sem_folder
            sem_post = self.sem_post
            sem_path0 = os.path.join(sem_path, "MegaDepth1500", img_folder0, f'{img_name0}.{sem_post}')
            sem_path1 = os.path.join(sem_path, "MegaDepth1500", img_folder1, f'{img_name1}.{sem_post}')
            self.sem0_path = sem_path0
            self.sem1_path = sem_path1
        else:
            raise NotImplementedError(f"semantic mode {semantic_mode} not implemented")

        self.img0_path = os.path.join(self.root_path, img_path0)
        self.img1_path = os.path.join(self.root_path, img_path1)

        img0 = cv2.imread(self.img0_path, cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.img1_path, cv2.IMREAD_COLOR)
        self.eval_W0, self.eval_H0 = img0.shape[1], img0.shape[0]
        self.eval_W1, self.eval_H1 = img1.shape[1], img1.shape[0]


        self.K0 = npz_data["intrinsics"][int(id0)].astype(np.float32)
        self.K1 = npz_data["intrinsics"][int(id1)].astype(np.float32)

        self.pose0 = npz_data["poses"][int(id0)]
        self.pose1 = npz_data["poses"][int(id1)]
        self.pose0 = np.matrix(self.pose0).astype(np.float32)
        self.pose1 = np.matrix(self.pose1).astype(np.float32)

    # override
    def load_images(self, W=None, H=None, PMer=False):
        """
        """
        if not PMer:
            return super().load_images(W, H)
        else:
            # specific for PMer: need padding
            match_color0, mask0, size0_ = load_img_padding_rt_size(self.img0_path, [W, H])
            crop_W0, crop_H0 = size0_ # NOTE: only used for PMer
            match_color1, mask1, size1_ = load_img_padding_rt_size(self.img1_path, [W, H])
            crop_W1, crop_H1 = size1_

            return match_color0, mask0, crop_W0, crop_H0, match_color1, mask1, crop_W1, crop_H1

    def load_Ks(self, scale0=None, scale1=None):
        """
        """
        return self.K0, self.K1

    def load_poses(self):
        """
        """
        return self.pose0, self.pose1

    def load_depths(self):
        """
        """
        raise NotImplementedError("MegaDepth does not provide depth info")
    
    def load_semantics(self):
        """
        """
        if self.sem_mode == "SAM":
            sem0 = np.load(self.sem0_path, allow_pickle=True)
            sem1 = np.load(self.sem1_path, allow_pickle=True)
        else:
            raise NotImplementedError(f"sem_mode {self.sem_mode} not implemented")
        
        return sem0, sem1
    
    def get_sem_paths(self):
        """
        """
        return self.sem0_path, self.sem1_path
    
    def get_eval_info(self, eval_W, eval_H):
        """
        """
        image0 = cv2.imread(self.img0_path, cv2.IMREAD_COLOR)
        image1 = cv2.imread(self.img1_path, cv2.IMREAD_COLOR)

        eval_W0, eval_H0 = image0.shape[1], image0.shape[0]
        eval_W1, eval_H1 = image1.shape[1], image1.shape[0]

        K0, K1 = self.load_Ks()
        P0, P1 = self.load_poses()
        sem0, sem1 = self.load_semantics()

        eval_info = {
            "dataset_name": "MegaDepth",
            "image0": image0,
            "image1": image1,
            'eval_W0': eval_W0,
            'eval_H0': eval_H0,
            'eval_W1': eval_W1,
            'eval_H1': eval_H1,
            "K0": K0,
            "K1": K1,
            "P0": P0,
            "P1": P1,
            "sem0": sem0,
            "sem1": sem1,
        }

        return eval_info

    def tune_corrs_size_to_eval(self, corrs, match_W0, match_H0, match_W1, match_H1):
        """ match at the same size, eval at the original size
        """
        eval_corrs = tune_corrs_size_diff(corrs, match_W0, match_W1, match_H0, match_H1, self.eval_W0, self.eval_W1, self.eval_H0, self.eval_H1)
        return eval_corrs


        