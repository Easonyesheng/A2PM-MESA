'''
Author: EasonZhang
Date: 2024-03-23 09:43:01
LastEditors: EasonZhang
LastEditTime: 2024-07-20 15:19:35
FilePath: /SA2M/hydra-mesa/area_matchers/AreaMatchDense.py
Description: Dense Area Matching via patch-wise match rendering

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import sys
sys.path.append("..")

import os
import numpy as np
from loguru import logger
import copy
import torch
import cv2
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import multivariate_normal


from .AreaGrapher import AreaGraph
from .AGConfig import areagraph_configs
from .CoarseAreaMatcher import CoarseAreaMatcher
from .AGBasic import AGNode
from utils.vis import (
    draw_matched_area, 
    draw_matched_area_list, 
    draw_matched_area_with_mkpts,
)
from utils.img_process import (
    img_crop_with_padding_expand_square_rt_area,
)
from utils.common import test_dir_if_not_create
from utils.geo import calc_areas_iou, recover_pts_offset_scales

class AGMatcherDense(object):
    """ Dense Area Matching via patch-wise match rendering
    Funcs:
        Source Area Selection 
        Patch Match Splatting
    """

    dft_cfgs = {
        "matcher_name": "ASpan",
        "datasetName": "ScanNet",
        "out_path": "",
        "level_num": 5,
        "level_step": [560, 480, 340, 200, 60, 0],
        "stop_match_level": 3,
        "W": 640, # original image size, also the match size
        "H": 480,
        "area_crop_mode": "expand_padding", # first expand to square, then padding
        "patch_size_ratio": 1/8,
        "valid_gaussian_width": 1,
        "show_flag": 0,
        "source_area_selection_mode": "direct",
        "iou_fusion_thd": 0.8, # iou threshold for repeat area identification
        "patch_match_num_thd": 10,
        "convex_hull_collect_thd": 10,
        "match_mode": "patch_match_splatting",
        "coarse_match_all_in_one": 0,
        "dual_match": 0,
    }

    def __init__(self, configs={}) -> None:
        """"""
        self.dft_cfgs.update(**configs)

        self.matcher_name = self.dft_cfgs["matcher_name"]
        self.datasetName = self.dft_cfgs["datasetName"]
        self.out_path = self.dft_cfgs["out_path"]
        self.level_num = self.dft_cfgs["level_num"]
        self.level_step = self.dft_cfgs["level_step"]
        self.stop_match_level = self.dft_cfgs["stop_match_level"]
        self.W = self.dft_cfgs["W"]
        self.H = self.dft_cfgs["H"]
        self.area_crop_mode = self.dft_cfgs["area_crop_mode"]
        self.patch_size_ratio = self.dft_cfgs["patch_size_ratio"]
        self.valid_gaussian_width = self.dft_cfgs["valid_gaussian_width"]
        self.show_flag = self.dft_cfgs["show_flag"]
        if self.show_flag == 1:
            test_dir_if_not_create(self.out_path)

        self.source_area_selection_mode = self.dft_cfgs["source_area_selection_mode"]
        self.iou_fusion_thd = self.dft_cfgs["iou_fusion_thd"]
        self.patch_match_num_thd = self.dft_cfgs["patch_match_num_thd"]
        self.convex_hull_collect_thd = self.dft_cfgs["convex_hull_collect_thd"]
        self.match_mode = self.dft_cfgs["match_mode"]

        if self.match_mode == 'EM':
            try:
                self.step_gmm = self.dft_cfgs["step_gmm"]
            except KeyError:
                logger.error("step_gmm should be set in the config")
                raise KeyError

        self.coarse_match_all_in_one = self.dft_cfgs["coarse_match_all_in_one"]
        self.dual_match = self.dft_cfgs["dual_match"]

        self.areagraph0 = None
        self.areagraph1 = None
        self.activity_map = None
        self.img_path0 = None
        self.img_path1 = None
        self.name0 = None
        self.name1 = None
        self.img0 = None
        self.img1 = None
        self.reverse_flag = False

        # collect final matched areas during the match flow (idxs)
        self.final_area_match_list_src = []
        self.final_area_match_list_dst = []

        # results
        self.res_area_match_list_src = []
        self.res_area_match_list_dst = []


        self.gaussian_sigma_u = self.patch_size_ratio * self.W / 2
        self.gaussian_sigma_v = self.patch_size_ratio * self.H / 2
        self.gaussian_kernel_width = int(2 * self.gaussian_sigma_u)*2//2+1
        self.gaussian_kernel_height = int(2 * self.gaussian_sigma_v)*2//2+1
        
        self.gaussian_constant = (2*np.pi)

        logger.debug(f"sigma_u: {self.gaussian_sigma_u}, sigma_v: {self.gaussian_sigma_v}")


    def init_area_matcher(self):
        """
        """
        matcher_configs = {
            "matcher_name": self.matcher_name,
            "datasetName": self.datasetName,
            "out_path": self.out_path,
            "pair_name": self.name0 + "_" + self.name1,
            "area_w": self.W,
            "area_h": self.H,
            "patch_size": 64, # NOT WORK here
            "conf_thd": 0.2, # NOT WORK here
        }
        self.area_matcher = CoarseAreaMatcher(matcher_configs)
        self.area_matcher.init_matcher()

    def path_loader(self, img_path0, sam_res_path0, img_path1, sam_res_path1, name0, name1):
        """
        """
        logger.info("load paths of two images")
        self.name0 = name0
        self.name1 = name1
        self.img_path0 = img_path0
        self.img_path1 = img_path1
        self.sam_res_path0 = sam_res_path0
        self.sam_res_path1 = sam_res_path1
        self.out_path = self.out_path + "/" + self.name0 + "_" + self.name1
        # test_dir_if_not_create(self.out_path)
        self._set_ag_config()

    def ori_img_load(self):
        """
        """
        if self.img0 is not None and self.img1 is not None:
            return
        assert self.img_path0 is not None and self.img_path1 is not None, "img_path0 and img_path1 should not be None"
        self.img0 = cv2.imread(self.img_path0, cv2.IMREAD_COLOR)
        self.img1 = cv2.imread(self.img_path1, cv2.IMREAD_COLOR)

        # resize
        self.img0 = cv2.resize(self.img0, (self.W, self.H))
        self.img1 = cv2.resize(self.img1, (self.W, self.H))

        return self.img0, self.img1
    
    def _set_ag_config(self):
        """
        """
        self.ag_config0 = deepcopy(areagraph_configs)
        self.ag_config0["ori_img_path"] = self.img_path0
        self.ag_config0["sam_res_path"] = self.sam_res_path0
        self.ag_config0["level_num"] = self.level_num
        self.ag_config0["level_step"] = self.level_step
        save_path0 = os.path.join(self.out_path, "area_graph0")
        self.ag_config0["save_path"] = save_path0
        self.ag_config0["show_flag"] = self.show_flag
        if self.show_flag == 1:
            test_dir_if_not_create(save_path0)


        self.ag_config1 = deepcopy(areagraph_configs)
        self.ag_config1["ori_img_path"] = self.img_path1
        self.ag_config1["sam_res_path"] = self.sam_res_path1
        self.ag_config1["level_num"] = self.level_num
        self.ag_config1["level_step"] = self.level_step
        save_path1 = os.path.join(self.out_path, "area_graph1")
        self.ag_config1["save_path"] = save_path1
        self.ag_config1["show_flag"] = self.show_flag
        if self.show_flag == 1:
            test_dir_if_not_create(save_path1)
    
    def img_areagraph_construct(self, efficient=False):
        """
        """
        # logger.info("construct area graph from two images")
        self.areagraph0 = AreaGraph(self.ag_config0)
        self.areagraph1 = AreaGraph(self.ag_config1)
        # clear
        self.final_area_match_list_src = []
        self.final_area_match_list_dst = []

        return self.areagraph0, self.areagraph1

    """Dense Area Matching"""
    def dense_area_matching_dual(self):
        """
        Flow:
            load ori images and crop areas inside the model
        """
        logger.info("start dense area matching")

        self.ori_img_load()

        if self.coarse_match_all_in_one == 0:
            matched_area0s, matched_area1s = self.dense_area_matching_single()
            if self.dual_match == 1:
                r_matched_area1s, r_matched_area0s = self.dense_area_matching_single(reverse=True, matched_source_areas=matched_area1s)
            
        elif self.coarse_match_all_in_one == 1:
            matched_area0s, matched_area1s = self.dense_area_matching_single_all_in_one()
            if self.dual_match == 1:
                r_matched_area1s, r_matched_area0s = self.dense_area_matching_single_all_in_one(reverse=True, matched_source_areas=matched_area1s)
        else: 
            raise NotImplementedError

        if self.dual_match == 1:
            matched_area0s = matched_area0s + r_matched_area0s
            matched_area1s = matched_area1s + r_matched_area1s



        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"final_matches")
            test_dir_if_not_create(out_img_folder)
            draw_matched_area_list(self.img0, self.img1, matched_area0s, matched_area1s, out_img_folder, "final", "match")

        return matched_area0s, matched_area1s

    def dense_area_matching_single_all_in_one(self, reverse=False, matched_source_areas=None):
        """coarse match only once
        """
        assert self.areagraph0 is not None and self.areagraph1 is not None, "areagraph0 and areagraph1 should not be None"


        # coarse match once
        mkpts0, mkpts1, mconf, _ = self.area_matcher.match(self.img0, self.img1, resize_flag=False)

        if self.match_mode == 'EM':
            mkpts1_, mkpts0_, mconf_, _ = self.area_matcher.match(self.img1, self.img0, resize_flag=False)

        if len(mkpts0) <= 10 or len(mkpts1) <= 10:
            return [], []

        if reverse:
            ag_src = self.areagraph1
            ag_dst = self.areagraph0
            mkpts_d2s_src = mkpts_src = mkpts1
            mkpts_d2s_dst = mkpts_dst = mkpts0
            mconf_d2s = mconf
            src_img = deepcopy(self.img1)
            dst_img = deepcopy(self.img0)

            if self.match_mode == 'EM':
                mkpts_s2d_src = mkpts1_
                mkpts_s2d_dst = mkpts0_
                mconf_s2d = mconf_


        else:
            ag_src = self.areagraph0
            ag_dst = self.areagraph1
            mkpts_s2d_src = mkpts_src = mkpts0
            mkpts_s2d_dst = mkpts_dst = mkpts1
            mconf_s2d = mconf
            src_img = deepcopy(self.img0)
            dst_img = deepcopy(self.img1)

            if self.match_mode == 'EM':
                mkpts_d2s_src = mkpts0_
                mkpts_d2s_dst = mkpts1_
                mconf_d2s = mconf_

        # select source areas
        if self.source_area_selection_mode == "direct":
            area_match_list_src = self.select_source_areas_direct(ag_src, matched_areas=matched_source_areas)
        elif self.source_area_selection_mode == "non-repeat":
            area_match_list_src = self.select_source_areas_non_repeat(ag_src, matched_areas=matched_source_areas)
            print(f"iou thd: {self.iou_fusion_thd} area_match_list_src: {len(area_match_list_src)}")
        else:
            logger.error(f"source area selection mode {self.source_area_selection_mode} not implemented")
            raise NotImplementedError 

        # find matched target areas for each source area
        area_match_list_dst_f = []
        area_match_list_src_f = []
        if self.match_mode == 'pms_GF':
            for i, area_src in enumerate(area_match_list_src):
                area_dst, area_src_real = self.match_given_node_with_mkpts(area_src, mkpts_src, mkpts_dst, mconf, src_img, dst_img, name=f"{i}_reverse_{reverse:1d}")
            
                if area_dst is None:
                    continue

                area_match_list_dst_f.append(area_dst)
                area_match_list_src_f.append(area_src_real)
        elif self.match_mode == 'EM':
            for i, area_src in enumerate(area_match_list_src): # TODO:
                area_dst, area_src_real = self.match_given_node_with_mkpts_EM(
                        area_src, 
                        mkpts_s2d_src, mkpts_s2d_dst, mconf_s2d,
                        mkpts_d2s_src, mkpts_d2s_dst, mconf_d2s,
                        src_img, dst_img, name=f"{i}_reverse_{reverse:1d}")
            
                if area_dst is None:
                    continue

                area_match_list_dst_f.append(area_dst)
                area_match_list_src_f.append(area_src_real)
            
        return area_match_list_src_f, area_match_list_dst_f

    def match_given_node_with_mkpts_EM(self, 
        area_src, 
        mkpts_s2d_src, mkpts_s2d_dst, mconf_s2d,
        mkpts_d2s_src, mkpts_d2s_dst, mconf_d2s, 
        src_img, dst_img, name=""):
        """find the matched target area in the target image for a given source area with matched coarse mkpts
        Args:
            area_src: [u_min, u_max, v_min, v_max]
            mkpts_src: [mkpt], mkpt is [u, v]
        Returns:
            area_dst: [u_min, u_max, v_min, v_max], if None, means no matched area
        """
        # 1. find the mkpts in the source area from mkpts_src and their correspondence in mkpts_dst
        inside_area_idx = self._find_inside_area_mkpts(area_src, mkpts_s2d_src)
        mkpts_s2d_src_np = np.array(mkpts_s2d_src)
        mkpts_s2d_src_inside = mkpts_s2d_src_np[inside_area_idx]
        mkpts_s2d_dst_np = np.array(mkpts_s2d_dst)
        mkpts_s2d_dst_inside = mkpts_s2d_dst_np[inside_area_idx]
        mconf_s2d_np = np.array(mconf_s2d)
        mconf_s2d_inside = mconf_s2d_np[inside_area_idx]

        # the other direction
        inside_area_idx = self._find_inside_area_mkpts(area_src, mkpts_d2s_src)
        mkpts_d2s_src_np = np.array(mkpts_d2s_src)
        mkpts_d2s_src_inside = mkpts_d2s_src_np[inside_area_idx]
        mkpts_d2s_dst_np = np.array(mkpts_d2s_dst)
        mkpts_d2s_dst_inside = mkpts_d2s_dst_np[inside_area_idx]
        mconf_d2s_np = np.array(mconf_d2s)
        mconf_d2s_inside = mconf_d2s_np[inside_area_idx]
        
        if len(mkpts_s2d_dst_inside) <= self.patch_match_num_thd or len(mkpts_d2s_dst_inside) <= self.patch_match_num_thd:
            return None, None

        # 2. match the area in the target image
        if self.match_mode == "EM":
            # TODO:
            area_src_real, area_dst = self.patch_match_splatting_EM_with_mkpts(
                mkpts_s2d_src_inside, mkpts_s2d_dst_inside, mconf_s2d_inside,
                mkpts_d2s_src_inside, mkpts_d2s_dst_inside, mconf_d2s_inside,
                area_src, src_img, dst_img, name)

        else:
            raise NotImplementedError

        logger.debug(f"area_dst is {area_dst}")
        
        # debug-draw the area match
        if self.show_flag and area_dst is not None:
            out_img_folder = os.path.join(self.out_path, f"final_match_single")
            test_dir_if_not_create(out_img_folder)
            color = (0, 255, 0)
            draw_matched_area(src_img, dst_img, area_src_real, area_dst, color, out_img_folder, name, "")

        return area_dst, area_src_real

    def match_given_node_with_mkpts(self, area_src, mkpts_src, mkpts_dst, mconf, src_img, dst_img, name=""):
        """find the matched target area in the target image for a given source area with matched coarse mkpts
        Args:
            area_src: [u_min, u_max, v_min, v_max]
            mkpts_src: [mkpt], mkpt is [u, v]
        Returns:
            area_dst: [u_min, u_max, v_min, v_max], if None, means no matched area
        """
        # 1. find the mkpts in the source area from mkpts_src and their correspondence in mkpts_dst
        inside_area_idx = self._find_inside_area_mkpts(area_src, mkpts_src)
        mkptsd_dst_np = np.array(mkpts_dst)
        mconf_np = np.array(mconf)
        mkptsd_dst_inside = mkptsd_dst_np[inside_area_idx]
        mconf_inside = mconf_np[inside_area_idx]
        
        if len(mkptsd_dst_inside) <= self.patch_match_num_thd:
            return None, None

        # 2. match the area in the target image
        if self.match_mode == "pms_GF":
            render_img = self._patch_match_splatting_GF_with_mkpts(mkptsd_dst_inside, mconf_inside, area_src, src_img, dst_img, name)
            area_dst = self.find_max_area_in_render(render_img)
        else:
            raise NotImplementedError

        logger.debug(f"area_dst is {area_dst}")
        
        # debug-draw the area match
        if self.show_flag and area_dst is not None:
            out_img_folder = os.path.join(self.out_path, f"final_match_single")
            test_dir_if_not_create(out_img_folder)
            color = (0, 255, 0)
            draw_matched_area(src_img, dst_img, area_src, area_dst, color, out_img_folder, name, "")

        return area_dst, area_src

    def _patch_match_splatting_GF_with_mkpts(self, mkpts_dst, mconf, area_src, src_img, dst_img, name=""):
        """
        Args:
            mkpts_dst: np.ndarray, [mkpt], mkpt is [u, v]
            mconf: np.ndarray, [conf]
        Returns:
            render_img: np.ndarray, [H, W]
        """
        render_img = np.zeros((self.H, self.W), dtype=np.float32)
        render_img_res = np.zeros((self.H, self.W), dtype=np.float32)
        
        mkpts1 = mkpts_dst.tolist()
        mconf1 = mconf.tolist()
        
        if len(mkpts1) <= self.patch_match_num_thd:
            return None
        
        logger.debug(f"matched points num: {len(mkpts1)}")

        # debug: draw the matched points
        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"match")
            test_dir_if_not_create(out_img_folder)
            out_img_match = deepcopy(dst_img)
            out_img_name = f"{name}_match"
            for mkpt in mkpts1:
                u, v = mkpt
                u = int(u)
                v = int(v)
                cv2.circle(out_img_match, (u, v), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(out_img_folder, f"{out_img_name}.png"), out_img_match)

        
        mkpts1_np = np.array(mkpts1)
        mkpts1_u = mkpts1_np[:, 0]
        mkpts1_v = mkpts1_np[:, 1]
        render_img_list = np.array(
            [self._render_direct_assign(render_img, u, v, conf) for u, v, conf in zip(mkpts1_u, mkpts1_v, mconf1)]
        )

        render_img_res = np.sum(render_img_list, axis=0)

        render_img_res = cv2.GaussianBlur(render_img_res, (self.gaussian_kernel_width, self.gaussian_kernel_height), 0)

        # normalize
        render_img_res = render_img_res / np.max(render_img_res)

        # vis
        if self.show_flag:
            self._draw_render(render_img_res, area_src, src_img, dst_img, mkpts1, name)

        return render_img_res

    def _draw_render(self, render_img, src_area, src_img, dst_img, means, name):
        """
        """
        out_img_folder = os.path.join(self.out_path, f"render")
        test_dir_if_not_create(out_img_folder)
        # draw means in the target image
        dst_img_show = deepcopy(dst_img)
        for mean in means:
            u, v = mean
            u = int(u)
            v = int(v)
            cv2.circle(dst_img_show, (u, v), 2, (0, 255, 0), -1)

        # apply colormap
        render_img_show = (render_img * 255).astype(np.uint8)
        render_img_show = cv2.applyColorMap(render_img_show, cv2.COLORMAP_JET)
        # draw together with the target image
        render_img_show = cv2.addWeighted(dst_img_show, 0.7, render_img_show, 0.3, 0)

        # draw src_area in the source image
        area_src_show = [int(x) for x in src_area]
        u_min, u_max, v_min, v_max = area_src_show
        src_img_temp = deepcopy(src_img)
        src_img_show = cv2.rectangle(src_img_temp, (u_min, v_min), (u_max, v_max), (0, 255, 0), 2)



        # stack with the source area image
        stack_img = np.hstack([src_img_show, render_img_show])

        cv2.imwrite(os.path.join(out_img_folder, f"{name}.png"), stack_img)

    def patch_match_splatting_EM_with_mkpts(self, 
        mkpts_s2d_src, # np.ndarray, [mkpt], mkpt is [u, v]
        mkpts_s2d_dst,
        mconf_s2d, 
        mkpts_d2s_src,
        mkpts_d2s_dst,
        mconf_d2s,
        area_src, 
        src_img, dst_img, name=""):
        """ TODO:
            use GMM to refine params and render the matched areas
        Args:
            mkpts_dst: [mkpt], mkpt is [u, v], Nx2
        """
        render_W, render_H = self.W, self.H
        render_img = np.zeros((render_H, render_W), dtype=np.float32)

        # find initial area matches from mkpts
        if self.show_flag: # only used for visualization
            dst_render_img_init = self._patch_match_splatting_GF_with_mkpts(mkpts_s2d_dst, mconf_s2d, area_src, src_img, dst_img, name+'_dst_temp_in_EM')
            dst_area_init = self.find_max_area_in_render(dst_render_img_init)

            src_render_img_init = self._patch_match_splatting_GF_with_mkpts(mkpts_d2s_src, mconf_d2s, dst_area_init, src_img, dst_img, name+'_src_temp_in_EM')
            src_area_init = self.find_max_area_in_render(src_render_img_init)
        else:
            dst_area_init = None
            src_area_init = None            

        # get means and conf by EM 
        mean_src_fused, cov_src_fused, weight_src_fused = self.gmm_fusion(mkpts_s2d_src, mconf_s2d, mkpts_d2s_src, mconf_d2s, 
                                                            filter_thd=0.3, sample_time=16, step_gmm=self.step_gmm, name=name+"_src",
                                                            area_src=dst_area_init, src_img=dst_img, target_img=src_img)

        mean_dst_fused, cov_dst_fused, weight_dst_fused = self.gmm_fusion(mkpts_s2d_dst, mconf_s2d, mkpts_d2s_dst, mconf_d2s, 
                                                            filter_thd=0.3, sample_time=16, step_gmm=self.step_gmm, name=name+"_dst",
                                                            area_src=src_area_init, src_img=src_img, target_img=dst_img)

        # render the matched areas
        src_render_img = self._render_GMM(render_img, weight_src_fused, mean_src_fused, cov_src_fused)
        dst_render_img = self._render_GMM(render_img, weight_dst_fused, mean_dst_fused, cov_dst_fused)

        area_src_final = self.find_max_area_in_render(src_render_img)
        area_dst_final = self.find_max_area_in_render(dst_render_img)

        return area_src_final, area_dst_final
    
    def _update_means(self, probs, points, num_clusters):
        """
        """
        means = np.zeros((num_clusters, 2))
        for l in range(num_clusters):
            means[l] = np.average(points, axis=0, weights=probs[:, l])
        return means
    
    def _update_vars(self, probs, points, means, num_clusters):
        """
        """
        covs = np.zeros((num_clusters, 2, 2))
        for l in range(num_clusters):
            diff = points - means[l]
            covs[l] = np.average(np.einsum('ij,ik->ijk', diff, diff), axis=0, weights=probs[:, l])
            # check if the covs is singular
            if np.linalg.matrix_rank(covs[l]) < 2:
                covs[l] += np.eye(2) * 1e-5
        return covs

    def _render_GMM(self, img, weights, means, covs, sample_step=2):
        """
        """
        img = img.copy()
        img = img.astype(np.float32)
        ori_W, ori_H = img.shape
        assert ori_W / sample_step == ori_W // sample_step, "ori_W should be divisible by sample_step"
        assert ori_H / sample_step == ori_H // sample_step, "ori_H should be divisible by sample_step"
        grid_u, grid_v = np.meshgrid(np.arange(0, self.W, sample_step), np.arange(0, self.H, sample_step))
        points = np.stack((grid_u.flatten(), grid_v.flatten()), axis=1)

        try:
            probs = np.array(
                [weight * multivariate_normal.pdf(points, mean=mean, cov=cov) for weight, mean, cov in zip(weights, means, covs)]
            )
        except Exception as e:
            return None

        img_temp = np.sum(probs, axis=0).reshape(self.H // sample_step, self.W // sample_step)

        # resize to the original size
        img_temp = cv2.resize(img_temp, (self.W, self.H))

        img += img_temp

        # normalize
        img /= np.max(img)

        # # Gaussian filter
        # img = cv2.GaussianBlur(img, (self.gaussian_kernel_width, self.gaussian_kernel_height), 0)

        return img

    def _find_inside_area_mkpts(self, area, mkpts):
        """
        """
        u_min, u_max, v_min, v_max = area
        mkpts_np = np.array(mkpts)  
        mkpts_inside_mask = (mkpts_np[:, 0] >= u_min) & (mkpts_np[:, 0] <= u_max) & (mkpts_np[:, 1] >= v_min) & (mkpts_np[:, 1] <= v_max)
        
        return mkpts_inside_mask

    def dense_area_matching_single(self, reverse=False, matched_source_areas=None):
        """ For each source area, find the matched target area
        Flow:
            1. select source areas
            2. match target areas
        Returns:
            area_match_list_src: [area]
            area_match_list_dst: [area]
        """
        assert self.areagraph0 is not None and self.areagraph1 is not None, "areagraph0 and areagraph1 should not be None"

        if reverse:
            ag_src = self.areagraph1
            source_img = self.img1
            target_img = self.img0
        else:
            ag_src = self.areagraph0
            source_img = self.img0
            target_img = self.img1

        # select source areas
        if self.source_area_selection_mode == "direct":
            area_match_list_src = self.select_source_areas_direct(ag_src, matched_areas=matched_source_areas)
        else:
            raise NotImplementedError
        
        # find matched target areas for each source area
        area_match_list_dst_f = []
        area_match_list_src_f = []
        for i, area_src in enumerate(area_match_list_src):
            area_dst, area_src_real = self.match_given_node(area_src, source_img, target_img, name=f"{i}_reverse_{reverse:1d}")

            if area_dst is None:
                continue

            area_match_list_dst_f.append(area_dst)
            area_match_list_src_f.append(area_src_real)
            
            # debug
            # break


        return area_match_list_src_f, area_match_list_dst_f

    def match_given_node(self, area_src, source_img, target_img, name=""):
        """ find the matched target area in the target image for a given source area
        Args:
            area_src: [u_min, u_max, v_min, v_max]
            target_img: np.ndarray
        Returns:
            area_dst: [u_min, u_max, v_min, v_max], if None, means no matched area
        """
        # 1. crop the area from the source image
        if self.area_crop_mode == "expand_padding":
            # NOTE: the src_area is spreaded with ratio = 1.2
            src_area_img, _, _, src_area_real = img_crop_with_padding_expand_square_rt_area(source_img, area_src, self.W, self.H, spread_ratio=1.0)
        else:
            raise NotImplementedError

        # 2. match the area in the target image
        if self.match_mode == "pms_GF":
            render_img = self.patch_match_splatting_Gaussian_filter(src_area_img, target_img, name=name)
            area_dst = self.find_max_area_in_render(render_img)
        elif self.match_mode == 'EM':
            src_area_real, area_dst = self.patch_match_splatting_gmm_dual(area_src=src_area_real, src_img=source_img, dst_img=target_img, name=name, step_gmm=self.step_gmm)
            pass
        else:
            raise NotImplementedError

        logger.debug(f"area_dst is {area_dst}")
        
        # debug-draw the area match
        if self.show_flag and area_dst is not None:
            out_img_folder = os.path.join(self.out_path, f"final_match_single")
            test_dir_if_not_create(out_img_folder)
            color = (0, 255, 0)
            draw_matched_area(source_img, target_img, src_area_real, area_dst, color, out_img_folder, name, "")

        return area_dst, src_area_real

    def patch_match_splatting_gmm_dual(self, area_src, src_img, dst_img, step_gmm=5, sample_time=16, name=""):
        """
        """
        logger.info(f"start patch match splatting gmm dual")
        render_img = np.zeros((self.H, self.W), dtype=np.float32)

        src_area_img, scale_src_st, offset_src_st, src_area_real = img_crop_with_padding_expand_square_rt_area(src_img, area_src, self.W, self.H, spread_ratio=1.0)

        # match the area in the target image, putatively
        render_dst_img_st, mean_dst_st, conf_dst_st, mean_src_st_temp = self.patch_match_splatting_Gaussian_filter_rt_gmm_both(src_area_img, dst_img, name=name+"_putative_st")

        if render_dst_img_st is None:
            return None, None

        # tune mean_src_st_temp to src img size
        mean_src_st = recover_pts_offset_scales(mean_src_st_temp, offset_src_st, scale_src_st)
        conf_src_st = conf_dst_st.copy()

        # get dst_area_init
        dst_area_init = self.find_max_area_in_render(render_dst_img_st, valid_gaussian_width=4)

        # get dst_area_img_init
        dst_area_img_init, scale_dst_st, offset_dst_st, dst_area_real_init = img_crop_with_padding_expand_square_rt_area(dst_img, dst_area_init, self.W, self.H, spread_ratio=1.0)

        # match the target area in the source image
        render_src_img_ts, mean_src_ts, conf_src_ts, mean_dst_ts_temp = self.patch_match_splatting_Gaussian_filter_rt_gmm_both(dst_area_img_init, src_img, name=name+"_putative_ts")
        
        if render_src_img_ts is None:
            return None, None

        # tune mean_dst_ts_temp to target img size
        mean_dst_ts = recover_pts_offset_scales(mean_dst_ts_temp, offset_dst_st, scale_dst_st)
        conf_dst_ts = conf_src_ts.copy()

        logger.info(f"start gmm fusion")

        # fusion
        mean_src_fused, cov_src_fused, weights_src_fused = self.gmm_fusion(mean_src_st, conf_src_st, mean_src_ts, conf_src_ts, filter_thd=0.3, sample_time=sample_time, step_gmm=step_gmm, name=name+"_src", area_src=dst_area_real_init, src_img=dst_img, target_img=src_img)
        mean_dst_fused, cov_dst_fused, weights_dst_fused = self.gmm_fusion(mean_dst_st, conf_dst_st, mean_dst_ts, conf_dst_ts, filter_thd=0.3, sample_time=sample_time, step_gmm=step_gmm, name=name+'_dst', area_src=src_area_real, src_img=src_img, target_img=dst_img)

        logger.info(f"start render")
        try:
            # render img
            src_render_img = self._render_GMM(render_img, weights_src_fused, mean_src_fused, cov_src_fused)
            dst_render_img = self._render_GMM(render_img, weights_dst_fused, mean_dst_fused, cov_dst_fused)

            # get the final area
            logger.info(f'start get final area')
            area_src_final = self.find_max_area_in_render(src_render_img, valid_gaussian_width=4)
            area_dst_final = self.find_max_area_in_render(dst_render_img, valid_gaussian_width=4)
        except Exception as e:
            logger.exception(f"Error: {e}")
            return None, None

        return area_src_final, area_dst_final

    def gmm_fusion(self, mean_src, conf_src, mean_dst, conf_dst, 
        filter_thd=0.3, sample_time=256, step_gmm=5, name="",
        area_src=None, src_img=None, target_img=None):
        """ fusion two sets of GMM params using EM
        Args:
            mean_src: np.ndarray, [mean], mean is [u, v] # use as the init GMM params
            conf_src: np.ndarray, [conf]
            mean_dst: np.ndarray, [mean], mean is [u, v] # used to sample points
            conf_dst: np.ndarray, [conf]
            area_src: area in the area img (the other image, not the means-belongs image)
        Returns:
            mean_fused: np.ndarray, [mean], mean is [u, v]
            conf_fused: np.ndarray, [conf]
            weights_fused: np.ndarray, [weights]
        """
        # filter the mean_src and conf_src
        assert len(mean_src) == len(conf_src), "len(mean_src) should be equal to len(conf_src)"
        mean_src = np.array(mean_src)
        conf_src = np.array(conf_src)
        filtered_idx = np.where(conf_src > filter_thd)
        mean_src = mean_src[filtered_idx]
        conf_src = conf_src[filtered_idx]

        # filter the mean_dst and conf_dst
        assert len(mean_dst) == len(conf_dst), "len(mean_dst) should be equal to len(conf_dst)"
        mean_dst = np.array(mean_dst)
        conf_dst = np.array(conf_dst)
        filtered_idx = np.where(conf_dst > filter_thd)
        mean_dst = mean_dst[filtered_idx]
        conf_dst = conf_dst[filtered_idx]

        # sample points using dst params

        temp_dst_covs = np.array(
            [np.array([[self.gaussian_sigma_u / conf_dst[i], 0], [0, self.gaussian_sigma_v / conf_dst[i]]]) for i in range(len(mean_dst))]
        )
        
        sampled_pts = np.array(
            [np.random.multivariate_normal(mean_dst[i], temp_dst_covs[i], size=(sample_time, 2)) for i in range(len(mean_dst))]
        )

        sampled_pts = sampled_pts.reshape(-1, 2)

        # EM part
        # init params
        num_clusters = len(mean_src)
        means = mean_src.copy()
        covs = np.zeros((num_clusters, 2, 2)) # N x 2 x 2

        # init covs
        render_W, render_H = self.W, self.H
        assert render_W == self.img0.shape[1] and render_H == self.img0.shape[0], "render_W and render_H should be equal to self.img0.shape[1] and self.img0.shape[0]"
        assert render_W == self.img1.shape[1] and render_H == self.img1.shape[0], "render_W and render_H should be equal to self.img1.shape[1] and self.img1.shape[0]"

        covs = np.array(
            [np.array([[self.gaussian_sigma_u / conf_src[i], 0], [0, self.gaussian_sigma_v / conf_src[i]]]) for i in range(num_clusters)]
        )

        weights = np.ones(num_clusters) / num_clusters # init with equal weights # N


        # draw the initial Gaussian distribution
        if self.show_flag:
            render_img = np.zeros((self.H, self.W), dtype=np.float32)
            render_img_temp = self._render_GMM(render_img, weights, means, covs)
            # draw render image
            render_img_show_temp = (render_img_temp * 255).astype(np.uint8)
            img_name = f"{name}_beforeGMM"
            out_img_folder = os.path.join(self.out_path, f"render_GMM")
            test_dir_if_not_create(out_img_folder)
            cv2.imwrite(os.path.join(out_img_folder, f"{img_name}.png"), render_img_show_temp)

            self._draw_render(render_img_temp, area_src, src_img, target_img, means, name+"_beforeGMM")

        logger.info(f"start EM iteration")

        # save the old params
        weights_old = weights.copy()
        means_old = means.copy()
        covs_old = covs.copy()

        # EM iteration
        for i in range(step_gmm):
            logger.info(f"EM iteration {i}")

            # E-step
            logger.info(f"#start E-step get {num_clusters} clusters")
            probs = np.zeros((len(sampled_pts), num_clusters))

            try:
                probs = np.array(
                    [weights[l] * multivariate_normal.pdf(sampled_pts, mean=means[l], cov=covs[l]) for l in range(num_clusters)]
                )
            except Exception as e:
                logger.exception(f"Error: {e}")
                return means_old, covs_old, weights_old

            # save the old params
            weights_old = weights.copy()
            means_old = means.copy()
            covs_old = covs.copy()

            probs = probs.T

            probs = probs / (np.sum(probs, axis=1).reshape(-1, 1)+1e-5)


            # M-step
            logger.info(f"#start M-step")
            weights = np.sum(probs, axis=(0)) / np.sum(probs)

            try:
                means = self._update_means(probs, sampled_pts, num_clusters)
                covs = self._update_vars(probs, sampled_pts, means, num_clusters)
            except Exception as e:
                logger.exception(f"Error: {e}")
                return means_old, covs_old, weights_old

            # draw the middle Gaussian distribution
            if self.show_flag:
                render_img_temp = self._render_GMM(render_img, weights, means, covs)
                if render_img_temp is None:
                    return means_old, covs_old, weights_old
                render_img_show_temp = (render_img_temp * 255).astype(np.uint8)
                img_name = f"{name}_EM_{i}"
                out_img_folder = os.path.join(self.out_path, f"render_GMM")
                test_dir_if_not_create(out_img_folder)
                cv2.imwrite(os.path.join(out_img_folder, f"{img_name}.png"), render_img_show_temp)

                self._draw_render(render_img_temp, area_src, src_img, target_img, means, img_name)

            # determine the convergence
            if i > 0:
                if np.linalg.norm(weights - weights_old) < 1e-3 and np.linalg.norm(means - means_old) < 1e-3 and np.linalg.norm(covs - covs_old) < 1e-3:
                    break
            
        return means, covs, weights

    def patch_match_splatting_Gaussian_filter_rt_gmm_both(self, src_area_img, target_img, name=""):
        """direct assign confidence on the target image and then perform the Gaussian filter
        Returns:
            matched_area: [u_min, u_max, v_min, v_max], if None, means no matched area
        """
        render_img = np.zeros((self.H, self.W), dtype=np.float32)
        render_img_res = np.zeros((self.H, self.W), dtype=np.float32)
        
        mkpts0, mkpts1, mconf, _ = self.area_matcher.match(src_area_img, target_img, resize_flag=False)
        
        if len(mkpts1) <= self.patch_match_num_thd:
            return None, None, None, None
        
        logger.debug(f"matched points num: {len(mkpts1)}")

        # debug: draw the matched points
        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"match")
            test_dir_if_not_create(out_img_folder)
            out_img_match = deepcopy(target_img)
            out_img_name = f"{name}_match"
            for mkpt in mkpts1:
                u, v = mkpt
                u = int(u)
                v = int(v)
                cv2.circle(out_img_match, (u, v), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(out_img_folder, f"{out_img_name}.png"), out_img_match)

        for i, mkpt in enumerate(mkpts1):
            u, v = mkpt
            conf = mconf[i]
            temp_render_img = deepcopy(render_img)
            # render the gaussian distribution
            # 
            iter_render_img = self._render_direct_assign(temp_render_img, u, v, conf)
            render_img_res += iter_render_img

            # # debug-draw the render image
            # if self.show_flag:
            #     out_img_folder = os.path.join(self.out_path, f"iter_render")
            #     test_dir_if_not_create(out_img_folder)
            #     out_img_temp = (iter_render_img * 255).astype(np.uint8)
            #     cv2.imwrite(os.path.join(out_img_folder, f"{name}_iter_{i}.png"), out_img_temp)

        # perform the Gaussian filter, filter size is 2*sigma_u, 2*sigma_v
        kernel_width = int(2 * self.gaussian_sigma_u)*2//2+1
        kernel_height = int(2 * self.gaussian_sigma_v)*2//2+1
        render_img_res = cv2.GaussianBlur(render_img_res, (kernel_width, kernel_height), 0)

        # normalize
        render_img_res = render_img_res / (np.max(render_img_res) + 1e-5)

        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"render")
            test_dir_if_not_create(out_img_folder)
            # apply colormap
            render_img_show = (render_img_res * 255).astype(np.uint8)

            # cv2.imwrite(os.path.join(out_img_folder, f"{name}_pure_rander.png"), render_img_show)

            render_img_show = cv2.applyColorMap(render_img_show, cv2.COLORMAP_JET)
            # draw together with the target image
            render_img_show = cv2.addWeighted(target_img, 0.7, render_img_show, 0.3, 0)

            # stack with the source area image
            stack_img = np.hstack([src_area_img, render_img_show])

            cv2.imwrite(os.path.join(out_img_folder, f"{name}.png"), stack_img)

        return render_img_res, mkpts1, mconf, mkpts0

    def patch_match_splatting_Gaussian_filter_rt_gmm(self, src_area_img, target_img, name=""):
        """direct assign confidence on the target image and then perform the Gaussian filter
        Returns:
            matched_area: [u_min, u_max, v_min, v_max], if None, means no matched area
        """
        render_img = np.zeros((self.H, self.W), dtype=np.float32)
        render_img_res = np.zeros((self.H, self.W), dtype=np.float32)
        
        _, mkpts1, mconf, _ = self.area_matcher.match(src_area_img, target_img, resize_flag=False)
        
        if len(mkpts1) <= self.patch_match_num_thd:
            return None, None, None
        
        logger.debug(f"matched points num: {len(mkpts1)}")

        # debug: draw the matched points
        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"match")
            test_dir_if_not_create(out_img_folder)
            out_img_match = deepcopy(target_img)
            out_img_name = f"{name}_match"
            for mkpt in mkpts1:
                u, v = mkpt
                u = int(u)
                v = int(v)
                cv2.circle(out_img_match, (u, v), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(out_img_folder, f"{out_img_name}.png"), out_img_match)

        for i, mkpt in enumerate(mkpts1):
            u, v = mkpt
            conf = mconf[i]
            temp_render_img = deepcopy(render_img)
            # render the gaussian distribution
            # 
            iter_render_img = self._render_direct_assign(temp_render_img, u, v, conf)
            render_img_res += iter_render_img

            # # debug-draw the render image
            # if self.show_flag:
            #     out_img_folder = os.path.join(self.out_path, f"iter_render")
            #     test_dir_if_not_create(out_img_folder)
            #     out_img_temp = (iter_render_img * 255).astype(np.uint8)
            #     cv2.imwrite(os.path.join(out_img_folder, f"{name}_iter_{i}.png"), out_img_temp)

        # perform the Gaussian filter, filter size is 2*sigma_u, 2*sigma_v
        kernel_width = int(2 * self.gaussian_sigma_u)*2//2+1
        kernel_height = int(2 * self.gaussian_sigma_v)*2//2+1
        render_img_res = cv2.GaussianBlur(render_img_res, (kernel_width, kernel_height), 0)

        # normalize
        render_img_res = render_img_res / (np.max(render_img_res) + 1e-5)

        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"render")
            test_dir_if_not_create(out_img_folder)
            # apply colormap
            render_img_show = (render_img_res * 255).astype(np.uint8)

            # cv2.imwrite(os.path.join(out_img_folder, f"{name}_pure_rander.png"), render_img_show)

            render_img_show = cv2.applyColorMap(render_img_show, cv2.COLORMAP_JET)
            # draw together with the target image
            render_img_show = cv2.addWeighted(target_img, 0.7, render_img_show, 0.3, 0)

            # stack with the source area image
            stack_img = np.hstack([src_area_img, render_img_show])

            cv2.imwrite(os.path.join(out_img_folder, f"{name}.png"), stack_img)

        return render_img_res, mkpts1, mconf

    def patch_match_splatting_Gaussian_filter(self, src_area_img, target_img, name=""):
        """direct assign confidence on the target image and then perform the Gaussian filter
        Returns:
            matched_area: [u_min, u_max, v_min, v_max], if None, means no matched area
        """
        render_img = np.zeros((self.H, self.W), dtype=np.float32)
        render_img_res = np.zeros((self.H, self.W), dtype=np.float32)
        
        _, mkpts1, mconf, _ = self.area_matcher.match(src_area_img, target_img, resize_flag=False)
        
        if len(mkpts1) <= self.patch_match_num_thd:
            return None
        
        logger.debug(f"matched points num: {len(mkpts1)}")

        # debug: draw the matched points
        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"match")
            test_dir_if_not_create(out_img_folder)
            out_img_match = deepcopy(target_img)
            out_img_name = f"{name}_match"
            for mkpt in mkpts1:
                u, v = mkpt
                u = int(u)
                v = int(v)
                cv2.circle(out_img_match, (u, v), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(out_img_folder, f"{out_img_name}.png"), out_img_match)

        for i, mkpt in enumerate(mkpts1):
            u, v = mkpt
            conf = mconf[i]
            temp_render_img = deepcopy(render_img)
            # render the gaussian distribution
            # 
            iter_render_img = self._render_direct_assign(temp_render_img, u, v, conf)
            render_img_res += iter_render_img

            # # debug-draw the render image
            # if self.show_flag:
            #     out_img_folder = os.path.join(self.out_path, f"iter_render")
            #     test_dir_if_not_create(out_img_folder)
            #     out_img_temp = (iter_render_img * 255).astype(np.uint8)
            #     cv2.imwrite(os.path.join(out_img_folder, f"{name}_iter_{i}.png"), out_img_temp)

        # perform the Gaussian filter, filter size is 2*sigma_u, 2*sigma_v
        kernel_width = int(2 * self.gaussian_sigma_u)*2//2+1
        kernel_height = int(2 * self.gaussian_sigma_v)*2//2+1
        render_img_res = cv2.GaussianBlur(render_img_res, (kernel_width, kernel_height), 0)

        # normalize
        render_img_res = render_img_res / (np.max(render_img_res) + 1e-5)

        if self.show_flag:
            out_img_folder = os.path.join(self.out_path, f"render")
            test_dir_if_not_create(out_img_folder)
            # apply colormap
            render_img_show = (render_img_res * 255).astype(np.uint8)

            # cv2.imwrite(os.path.join(out_img_folder, f"{name}_pure_rander.png"), render_img_show)

            render_img_show = cv2.applyColorMap(render_img_show, cv2.COLORMAP_JET)
            # draw together with the target image
            render_img_show = cv2.addWeighted(target_img, 0.7, render_img_show, 0.3, 0)

            # stack with the source area image
            stack_img = np.hstack([src_area_img, render_img_show])

            cv2.imwrite(os.path.join(out_img_folder, f"{name}.png"), stack_img)

        return render_img_res

    def _render_direct_assign(self, img, u, v, conf):
        """
        """
        img = copy.deepcopy(img)
        img = img.astype(np.float32)

        sigma_u = self.gaussian_sigma_u
        sigma_v = self.gaussian_sigma_v

        img_h, img_w = img.shape
        assign_range = 1
        u_min = max(0, int(u - assign_range * sigma_u))
        u_max = min(img_w, int(u + assign_range * sigma_u))
        v_min = max(0, int(v - assign_range * sigma_v))
        v_max = min(img_h, int(v + assign_range * sigma_v))

        # assign the confidence to the target image
        img[v_min:v_max, u_min:u_max] = conf

        return img

    def find_max_area_in_render(self, render_img, valid_gaussian_width=None):
        """
        Args:
            render_img: np.ndarray, normalized
        Returns:
            max_area: [u_min, u_max, v_min, v_max]
        """
        self.valid_gaussian_width = str(self.valid_gaussian_width)
        if render_img is None:
            return None
        if valid_gaussian_width is not None:
            valid_value =  (1 / self.gaussian_constant) * np.exp(-valid_gaussian_width)
        elif self.valid_gaussian_width == '2sqrt2': # 2 sigma
            valid_value = (1 / self.gaussian_constant) * np.exp(-4)
        elif self.valid_gaussian_width == '2': # 2 sigma
            valid_value = (1 / self.gaussian_constant) * np.exp(-2)
        elif self.valid_gaussian_width == 'sqrt2': # sqrt2 sigma
            valid_value = (1 / self.gaussian_constant) * np.exp(-1)
        elif self.valid_gaussian_width == '1': # 1/2 sigma
            valid_value = (1 / self.gaussian_constant) * np.exp(-0.5)
        elif self.valid_gaussian_width == '0.5': # 2 sigma
            valid_value = (1 / self.gaussian_constant) * np.exp(-0.125)
        else:
            logger.error(f"valid_gaussian_width {self.valid_gaussian_width} got type: {type(self.valid_gaussian_width)}")
            raise NotImplementedError

        # find the max area
        valid_mask = render_img >= valid_value
        valid_mask = valid_mask.astype(np.uint8)
        # find the u_min, u_max, v_min, v_max
        try:
            valid_uv = np.where(valid_mask == 1)
            u_coords = valid_uv[1]
            v_coords = valid_uv[0]

        except Exception as e:
            logger.debug(f"valid_us is {valid_uv}")
            logger.exception(e)
            return None
            
        u_min = np.min(u_coords)
        u_max = np.max(u_coords)
        v_min = np.min(v_coords)
        v_max = np.max(v_coords)

        return [u_min, u_max, v_min, v_max]

    def select_source_areas_non_repeat(self, areagraph, matched_areas=None):
        """
        """
        src_level = self.stop_match_level
        src_area_idxs = areagraph.get_nodes_with_level(src_level)
        src_areas = []

        for i in src_area_idxs:
            src_areas = self._add_area_non_repeat(areagraph.AGNodes[i].area, src_areas)

        if matched_areas is not None:
            src_areas_f = self._add_areas_non_repeat(src_areas, matched_areas)
        else:
            return src_areas
            
        return src_areas_f

    def select_source_areas_direct(self, areagraph, matched_areas=None):
        """
        Returns:
            src_areas: [area], area is [u_min, u_max, v_min, v_max]
        """
        src_level = self.stop_match_level
        src_area_idxs = areagraph.get_nodes_with_level(src_level)
        src_areas = [areagraph.AGNodes[i].area for i in src_area_idxs]
        if matched_areas is not None:
            src_areas_f = self._add_areas_non_repeat(src_areas, matched_areas)
        else:
            return src_areas
            
        return src_areas_f

    def _add_area_non_repeat(self, area, areas):
        """
        """
        if len(areas) == 0:
            return [area]

        for area0 in areas:
            iou = calc_areas_iou(area, area0)
            if iou > self.iou_fusion_thd:
                return areas

        areas.append(area)

        return areas

    def _add_areas_non_repeat(self, areas, matched_areas):
        """
        """
        if len(matched_areas) == 0:
            return areas

        areas_f = []
        for area in areas:
            iou, _ = self._calc_max_iou(area, matched_areas)
            if iou < self.iou_fusion_thd:
                areas_f.append(area)

        return areas_f

    def _calc_max_iou(self, area0, areas):
        """
        """
        max_iou = 0
        max_id = -1
        for i, area in enumerate(areas):
            iou = calc_areas_iou(area0, area)
            max_iou = max(max_iou, iou)
            if max_iou != iou:
                max_id = i

        return max_iou, max_id