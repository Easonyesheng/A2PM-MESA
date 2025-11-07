'''
Author: EasonZhang
Date: 2024-06-20 11:42:48
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-07 16:10:28
FilePath: /SA2M/hydra-mesa/area_matchers/mesa.py
Description: traning-free version mesa

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import os
import os.path as osp
import numpy as np
import copy
import torch
import cv2
from loguru import logger
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

from .AreaGrapher import AreaGraph
from .AGConfig import areagraph_configs
from .AGUtils import GraphCutSolver
from .CoarseAreaMatcher import CoarseAreaMatcher
from .AGBasic import AGNode
from utils.vis import draw_matched_area, draw_matched_area_list, draw_matched_area_with_mkpts
from utils.common import test_dir_if_not_create, validate_type
from utils.geo import calc_areas_iou

from dataloader.abstract_dataloader import AbstractDataloader
from .abstract_am import AbstractAreaMatcher


class MesaAreaMatcher(AbstractAreaMatcher):
    """ MESAAreaMatcher, NOTE this is a training free version
    """
    def __init__(self, 
        W,
        H,
        coarse_matcher_name,
        level_num,
        level_step,
        adj_weight,
        stop_match_level,
        coarse_match_thd,
        patch_size,
        similar_area_dist_thd,
        area_w,
        area_h,
        sigma_thd,
        global_energy_weights,
        iou_fusion_thd,
        candi_energy_thd,
        global_refine,
        global_energy_candy_range,
        fast_version,
        energy_norm_way,
        datasetName, 
        draw_verbose=0):
        """
        """
        self.coarse_matcher_name = coarse_matcher_name
        self.level_num = level_num
        self.level_step = level_step
        self.adj_weight = adj_weight
        self.stop_match_level = stop_match_level
        self.coarse_match_thd = coarse_match_thd
        self.patch_size = patch_size
        self.similar_area_dist_thd = similar_area_dist_thd
        self.area_w = area_w
        self.area_h = area_h
        self.sigma_thd = sigma_thd
        self.global_energy_weights = global_energy_weights
        self.iou_fusion_thd = iou_fusion_thd
        self.candi_energy_thd = candi_energy_thd
        self.global_refine = global_refine
        self.global_energy_candy_range = global_energy_candy_range
        self.fast_version = fast_version
        self.energy_norm_way = energy_norm_way
        
        self.W, self.H = W, H
        self.datasetName = datasetName
        self.draw_verbose = draw_verbose


    def init_dataloader(self, dataloader):
        """
        """
        validate_type(dataloader, AbstractDataloader)
        self.sem_path0, self.sem_path1 = dataloader.get_sem_paths()

        self.name0 = dataloader.image_name0
        self.name1 = dataloader.image_name1

        self.image0_path, self.image1_path = dataloader.img0_path, dataloader.img1_path

        self.scene_name = dataloader.scene_name

        self.img0, self.img1, self.scale0, self.scale1 = dataloader.load_images(self.W, self.H)

    def init_from_imgs(self, img0, img1, ):
        """
        """

    def name(self):
        return "MesaAreaMatcher-TrainingFree"

    def init_area_matcher(self):
        """
        """
        AM_config = {
            "matcher_name": self.coarse_matcher_name,
            "datasetName": self.datasetName,
            "out_path": self.out_path,
            "level_num": self.level_num,
            "level_step": self.level_step,
            "adj_weight": self.adj_weight,
            "stop_match_level": self.stop_match_level,
            "W": self.W,
            "H": self.H,
            "coarse_match_thd": self.coarse_match_thd,
            "patch_size": self.patch_size,
            "similar_area_dist_thd": self.similar_area_dist_thd,
            "area_w": self.area_w if self.coarse_matcher_name != 'mast3r' else 512,
            "area_h": self.area_h if self.coarse_matcher_name != 'mast3r' else 512,
            "show_flag": self.draw_verbose,
            "sigma_thd": self.sigma_thd,
            "global_energy_weights": self.global_energy_weights,
            "iou_fusion_thd": self.iou_fusion_thd,
            "candi_energy_thd": self.candi_energy_thd,
            "global_refine": self.global_refine,
            "global_energy_candy_range": self.global_energy_candy_range,
            "fast_version": self.fast_version,
            "energy_norm_way": "minmax",
        }

        from .AGMatcherFree import AGMatcherF
        self.area_matcher = AGMatcherF(configs=AM_config)

    def area_matching(self, dataloader, out_path):
        """
        """
        logger.info(f"start area matching")

        self.init_dataloader(dataloader)
        self.set_outpath(out_path)

        self.init_area_matcher()
        self.area_matcher.path_loader(self.image0_path, self.sem_path0, self.image1_path, self.sem_path1, self.name0, self.name1)
        self.area_matcher.init_area_matcher()
        self.area_matcher.img_areagraph_construct(efficient=True)

        # debug - end the total process for only AG construction
        # import sys
        # sys.exit(0)

        area_match_srcs, area_match_dsts = self.area_matcher.dual_graphical_match(self.draw_verbose)
        

        self.area_match_srcs = area_match_srcs
        self.area_match_dsts = area_match_dsts

        if self.draw_verbose:
            flag = draw_matched_area_list(self.img0, self.img1, area_match_srcs, area_match_dsts, self.out_path, self.name0, self.name1, self.draw_verbose)
            if not flag:
                logger.critical(f"Something wrong with area matching, please check the code for {self.out_path.split('/')[-1]}")

            # draw each area's match
            for i, src_area in enumerate(area_match_srcs):
                dst_area = area_match_dsts[i]
                draw_matched_area(self.img0, self.img1, src_area, dst_area, (0,255,0), self.out_path, f"{i}_"+self.name0, self.name1, self.draw_verbose)

        logger.success(f"finish area matching")


        return area_match_srcs, area_match_dsts
