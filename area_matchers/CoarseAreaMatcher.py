'''
Author: EasonZhang
Date: 2023-06-28 22:11:54
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-24 14:59:43
FilePath: /SA2M/hydra-mesa/area_matchers/CoarseAreaMatcher.py
Description: Input two sub-images, output inside coarse point matches using off-the-shelf point matcher.

Copyright (c) 2023 by EasonZhang, All Rights Reserved. 
'''

import sys
sys.path.append("..")

import os
import os.path as osp
import numpy as np
import math
import cv2
from loguru import logger
# plt
import matplotlib.pyplot as plt


from utils.geo import tune_corrs_size, tune_mkps_size
from utils.common import test_dir_if_not_create

supported_matchers = ["ASpan", "LoFTR", "mast3r", "ASpanCD", "LoFTRCD"]

class CoarseAreaMatcher(object):
    """
    """
    def __init__(self, configs={}) -> None:
        """
        """
        self.matcher_name = configs["matcher_name"]
        self.mast3r_weight_path = configs.get("mast3r_weight_path", "")
        self.matcher = None
        self.datasetName = configs["datasetName"]
        self.out_path = configs["out_path"]
        self.area_w = configs["area_w"]
        self.area_h = configs["area_h"]
        self.patch_size = configs["patch_size"]
        self.conf_thd = configs["conf_thd"]
        self.out_path = configs["out_path"]
        self.pair_name = configs["pair_name"]
        pass

    def init_matcher(self):
        """
        """
        cur_path = osp.dirname(osp.abspath(__file__))

        if self.matcher_name == "ASpan":
            from point_matchers.aspanformer import ASpanMatcher
            logger.debug("Initialize ASpan Matcher")
            if self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "ETH3D":
                weight_path = f"{cur_path}/../point_matchers/ASpanFormer/weights/indoor.ckpt"
            elif self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
                weight_path = f"{cur_path}/../point_matchers/ASpanFormer/weights/outdoor.ckpt"
            else:
                raise NotImplementedError(f"Dataset {self.datasetName} not implemented yet!")

            aspan_configs = {
                "config_path": f"{cur_path}/../point_matchers/ASpanFormer/configs",
                "weights": weight_path,
                "dataset_name": self.datasetName,
            }
            self.matcher = ASpanMatcher(**aspan_configs)

        elif self.matcher_name == "ASpanCD":
            from point_matchers.aspanformer import ASpanMatcher
            logger.debug("Initialize ASpan Matcher")
            if self.datasetName == "ScanNet" or self.datasetName == "KITTI":
                weight_path = f"{cur_path}/../point_matchers/ASpanFormer/weights/outdoor.ckpt" # cross domain
            elif self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
                weight_path = f"{cur_path}/../point_matchers/ASpanFormer/weights/indoor.ckpt" # cross domain
            else:
                raise NotImplementedError(f"Dataset {self.datasetName} not implemented yet!")
            
            aspan_configs = {
                "config_path": f"{cur_path}/../point_matchers/ASpanFormer/configs",
                "weights": weight_path,
                "dataset_name": self.datasetName,
            }
            self.matcher = ASpanMatcher(**aspan_configs)
        
        elif self.matcher_name == "LoFTR":
            from point_matchers.loftr import LoFTRMatcher
            logger.debug("Initialize LoFTR Matcher")
            if self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "ETH3D":
                weight_path = f"{cur_path}/../point_matchers/LoFTR/weights/indoor_ds_new.ckpt"
            elif self.datasetName == "MegaDepth":
                weight_path = f"{cur_path}/../point_matchers/LoFTR/weights/outdoor_ds.ckpt"
            else:
                raise NotImplementedError(f"Dataset {self.datasetName} not implemented yet!")
            
            loftr_configs = {
                "weights": weight_path,
                "cuda_idx": 0,
            }
            self.matcher = LoFTRMatcher(loftr_configs, self.datasetName, mode="tool")
        elif self.matcher_name == "LoFTRCD":
            from point_matchers.loftr import LoFTRMatcher
            logger.debug("Initialize LoFTR Matcher")
            if self.datasetName == "ScanNet" or self.datasetName == "KITTI":
                weight_path = f"{cur_path}/../point_matchers/LoFTR/weights/outdoor_ds.ckpt" # cross domain
            elif self.datasetName == "MegaDepth":
                weight_path = f"{cur_path}/../point_matchers/LoFTR/weights/indoor_ds_new.ckpt" # cross domain
            else:
                raise NotImplementedError(f"Dataset {self.datasetName} not implemented yet!")
            loftr_configs = {
                "weights": weight_path,
                "cuda_idx": 0,
            }
            self.matcher = LoFTRMatcher(loftr_configs, self.datasetName, mode="tool")
        elif self.matcher_name == "mast3r":
            from point_matchers.mast3r import Mast3rMatcher
            logger.debug("Initialize Mast3r Matcher")
            weight_path = self.mast3r_weight_path
            self.matcher = Mast3rMatcher(weight_path, device="cuda:0")
        else:
            raise NotImplementedError(f"Matcher {self.matcher_name} not implemented yet!")
    
    def match(self, area0, area1, resize_flag=True):
        """ NOTE: this is the main match function
        Return:
            mkpts0_c: (N, 2), np.array
            mkpts1_c: (N, 2), np.array
        """
        assert self.matcher is not None, "Matcher not initialized yet!"
        area0_h, area0_w, _ = area0.shape
        area1_h, area1_w, _ = area1.shape

        if resize_flag:
            # resize area0 and area1 to the same size
            area0 = cv2.resize(area0, (self.area_w, self.area_h))
            area1 = cv2.resize(area1, (self.area_w, self.area_h))

        logger.info(f"match areas with size: {area0.shape}, {area1.shape}")

        try:
            ret = self.matcher.get_coarse_mkpts_c(area0, area1)
        except Exception as e:
            logger.exception(e)
            return None

        mkpts0_c, mkpts1_c, mconf, conf_mat = ret
        # put in cpu
        mkpts0_c = mkpts0_c.cpu().numpy()
        mkpts1_c = mkpts1_c.cpu().numpy()
        mconf = mconf.cpu().numpy()

        # tune mkpts size back to original area size
        if resize_flag:
            mkpts0_c = tune_mkps_size(mkpts0_c, self.area_w, self.area_h, area0_w, area0_h)
            mkpts1_c = tune_mkps_size(mkpts1_c, self.area_w, self.area_h, area1_w, area1_h)
            mkpts0_c = np.array(mkpts0_c)
            mkpts1_c = np.array(mkpts1_c)

        return mkpts0_c, mkpts1_c, mconf, conf_mat

    def match_ret_activity(self, area0, area1, sigma_thd=0.1, draw_match=False, name=""):
        """ match two areas, return the activity
        Return:
            sigma0: float, the activity of area0, = len(mkp0_c)*64 / (area_w * area_h)
            sigma1: float, the activity of area1, = len(mkp1_c)*64 / (area_w * area_h)
        """
        conf_thd = self.conf_thd
        assert self.matcher is not None, "Matcher not initialized yet!"

        # resize area0 and area1 to the same size
        area0 = cv2.resize(area0, (self.area_w, self.area_h))
        area1 = cv2.resize(area1, (self.area_w, self.area_h))

        # logger.info(f"match areas with size: {area0.shape}, {area1.shape}")
        if self.matcher_name in supported_matchers:
            try:
                # time_start = cv2.getTickCount()
                ret = self.matcher.get_coarse_mkpts_c(area0, area1)
                # time_end = cv2.getTickCount()
                # logger.info(f"single matching time: {(time_end - time_start) / cv2.getTickFrequency()}s")

            except Exception as e:
                logger.exception(e)
                return None

            mkpts0_c, mkpts1_c, mconf, _ = ret
            # put in cpu
            mkpts0_c = mkpts0_c.cpu().numpy()
            mkpts1_c = mkpts1_c.cpu().numpy()
            mconf = mconf.cpu().numpy()
            
            # filter by conf_thd
            mkpts0_c = mkpts0_c[mconf > conf_thd]
            mkpts1_c = mkpts1_c[mconf > conf_thd]

            # calc activity
            sigma0 = self.calc_activity_by_occ(mkpts0_c, self.area_w, self.area_h)
            sigma1 = self.calc_activity_by_occ(mkpts1_c, self.area_w, self.area_h)
            # logger.info(f"real activity: {sigma0}, {sigma1}")
            sigma0 = 0 if sigma0 < sigma_thd else sigma0
            sigma1 = 0 if sigma1 < sigma_thd else sigma1

            if draw_match:
                # tune mkpts size
                # mkpts0_c = tune_mkps_size(mkpts0_c, self.area_w, self.area_h, area0_w, area0_h)
                # mkpts1_c = tune_mkps_size(mkpts1_c, self.area_w, self.area_h, area1_w, area1_h)
                # mkpts0_c = np.array(mkpts0_c)
                # mkpts1_c = np.array(mkpts1_c)
                self.visualization(area0, area1, mkpts0_c, mkpts1_c, mconf, name=name+f"_{sigma0 :.2f}_{sigma1 :.2f}")
                pass
        else:
            raise NotImplementedError(f"Matcher {self.matcher_name} not implemented yet!")

        # logger.info(f"sigma0: {sigma0}")
        # logger.info(f"sigma1: {sigma1}")

        return sigma0, sigma1

    def calc_activity_by_occ(self, mkpts, area_w, area_h):
        """ calc activity by occlusion
            each mkpt is a 2d point, representing the center of a patch with radius patch_r

        """
        patch_r = self.patch_size
        occ_map = np.zeros((area_h, area_w), dtype=np.uint8)
        for i in range(mkpts.shape[0]):
            pt = mkpts[i]
            pt[0] = min(max(pt[0], patch_r), area_w-patch_r)
            pt[1] = min(max(pt[1], patch_r), area_h-patch_r)
            pt = (int(pt[0]), int(pt[1]))
            occ_map[pt[1]-patch_r:pt[1]+patch_r, pt[0]-patch_r:pt[0]+patch_r] = 1
        
        occ_num = np.sum(occ_map)
        occ_ratio = occ_num / (area_w * area_h)
        return occ_ratio

    def visualization(self, area0, area1, mkpts0_c, mkpts1_c, mconf, name=""):
        """ visualization the matching result in two areas
        Args:
            area0: np.array, shape: [area_h, area_w, 3]
            area1: np.array, shape: [area_h, area_w, 3]
            mkpts0_c: np.array, shape: [n, 2]
            mkpts1_c: np.array, shape: [n, 2]
            mconf: np.array, shape: [n, 1]
            conf_mat: np.array, shape: [area_h*area_w, area_h*area_w]
        """
        # if area is gray, convert to rgb
        if len(area0.shape) == 2:
            area0 = cv2.cvtColor(area0, cv2.COLOR_GRAY2RGB)
        if len(area1.shape) == 2:
            area1 = cv2.cvtColor(area1, cv2.COLOR_GRAY2RGB)

        # tune mkpts size
        mkpts0_c = tune_mkps_size(mkpts0_c, area0.shape[1], area0.shape[0], self.area_w, self.area_h)
        mkpts1_c = tune_mkps_size(mkpts1_c, area1.shape[1], area1.shape[0], self.area_w, self.area_h)
        
        # resize area0 and area1 to the same size
        area0 = cv2.resize(area0, (self.area_w, self.area_h))
        area1 = cv2.resize(area1, (self.area_w, self.area_h))


        # draw mkpts with 8x8 red rectangle in the combined image
        area0_mkpts = area0.copy()
        area1_mkpts = area1.copy()

        patch_radius = self.patch_size // 2

        img_out = np.zeros((self.area_h, self.area_w*2, 3), dtype=np.uint8)
        img_out_rect = np.zeros((self.area_h, self.area_w*2, 3), dtype=np.uint8)
        img_out[:, :self.area_w, :] = area0_mkpts
        img_out[:, self.area_w:, :] = area1_mkpts


        for i in range(mkpts0_c.shape[0]):
            pt0 = mkpts0_c[i]
            pt1 = mkpts1_c[i]
            conf = mconf[i]
            if conf < self.conf_thd:
                continue
            
            # random color
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            # logger.info(f"color: {color}")

            # fix coordinate
            pt0[0] = min(max(pt0[0], patch_radius), self.area_w-patch_radius)     
            pt0[1] = min(max(pt0[1], patch_radius), self.area_h-patch_radius)
            pt1[0] = min(max(pt1[0], patch_radius), self.area_w-patch_radius)
            pt1[1] = min(max(pt1[1], patch_radius), self.area_h-patch_radius)

            pt0 = (int(pt0[0]), int(pt0[1]))
            pt1 = (int(pt1[0]+self.area_w), int(pt1[1]))
            
            cv2.rectangle(img_out_rect, (pt0[0]-patch_radius, pt0[1]-patch_radius), (pt0[0]+patch_radius, pt0[1]+patch_radius), color, -1)
            cv2.rectangle(img_out_rect, (pt1[0]-patch_radius, pt1[1]-patch_radius), (pt1[0]+patch_radius, pt1[1]+patch_radius), color, -1)
            # cv2.line(img_out, pt0, pt1, (0, 255, 0), 1)

        # img add
        img_out = cv2.addWeighted(img_out, 1.0, img_out_rect, 0.5, 1)


        # save the image
        if name == "":
            name_img = f"{self.matcher_name}_patch_matches.png"
        else:
            name_img = f"{name}_patch_matches.png"

        out_folder = os.path.join(self.out_path, f"coarse_patch_matches_{self.pair_name}")
        test_dir_if_not_create(out_folder+"/"+"/".join(name_img.split("/")[:-1]))

        name_img = os.path.join(out_folder, name_img)

        logger.info(f"save patch matches image to {name_img}")
        cv2.imwrite(name_img, img_out)