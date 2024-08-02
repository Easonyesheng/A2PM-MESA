

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
import time

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


class DMesaAreaMatcher(AbstractAreaMatcher):
    """ DMESAAreaMatcher
    """
    def __init__(self, 
        W,
        H,
        coarse_matcher_name,
        level_num,
        level_step,
        stop_match_level,
        area_crop_mode,
        patch_size_ratio,
        valid_gaussian_width,
        source_area_selection_mode,
        iou_fusion_thd,
        patch_match_num_thd,
        match_mode,
        coarse_match_all_in_one,
        dual_match,
        datasetName, 
        step_gmm=None,
        draw_verbose=0):
        """
        """
        self.coarse_matcher_name = coarse_matcher_name
        self.level_num = level_num
        self.level_step = level_step
        self.area_crop_mode = area_crop_mode
        self.patch_size_ratio = patch_size_ratio
        self.valid_gaussian_width = valid_gaussian_width
        self.source_area_selection_mode = source_area_selection_mode
        self.iou_fusion_thd = iou_fusion_thd
        self.patch_match_num_thd = patch_match_num_thd
        self.match_mode = match_mode
        if self.match_mode == "EM":
            self.step_gmm = step_gmm
        self.coarse_match_all_in_one = coarse_match_all_in_one
        self.dual_match = dual_match
        self.stop_match_level = stop_match_level
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
            "stop_match_level": self.stop_match_level,
            "W": self.W, # original image size, NOTE also the match size
            "H": self.H,
            "area_crop_mode": self.area_crop_mode, # first expand to square, then padding
            "patch_size_ratio": self.patch_size_ratio,
            "valid_gaussian_width": self.valid_gaussian_width,
            "show_flag": self.draw_verbose,
            "source_area_selection_mode": self.source_area_selection_mode,
            "iou_fusion_thd": self.iou_fusion_thd, # iou threshold for repeat area identification
            "patch_match_num_thd": self.patch_match_num_thd, # threshold for patch match number
            "match_mode": self.match_mode,
            "coarse_match_all_in_one": self.coarse_match_all_in_one,
            "dual_match": self.dual_match,
        }

        if self.match_mode == "EM":
            AM_config.update({"step_gmm": self.step_gmm})

        from .AreaMatchDense import AGMatcherDense
        self.area_matcher = AGMatcherDense(configs=AM_config)

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

        area_match_srcs, area_match_dsts = self.area_matcher.dense_area_matching_dual()
        

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

        logger.info(f"finish area matching")


        return area_match_srcs, area_match_dsts

    def area_matching_rt_time(self, dataloader, out_path):
        """
        """
        logger.info(f"start area matching")

        self.init_dataloader(dataloader)
        self.set_outpath(out_path)

        self.init_area_matcher()
        self.area_matcher.path_loader(self.image0_path, self.sem_path0, self.image1_path, self.sem_path1, self.name0, self.name1)
        
        self.area_matcher.init_area_matcher()
        times0 = cv2.getTickCount()
        self.area_matcher.img_areagraph_construct(efficient=True)

        area_match_srcs, area_match_dsts = self.area_matcher.dense_area_matching_dual()
        times1 = cv2.getTickCount()
        time_match = (times1 - times0) / cv2.getTickFrequency()


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

        logger.info(f"finish area matching")

        return area_match_srcs, area_match_dsts, time_match