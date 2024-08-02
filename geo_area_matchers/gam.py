
import os
import torch
import numpy as np
import copy
import cv2
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from omegaconf import OmegaConf

from utils.load import (
    load_cv_img_resize, 
)

from utils.geo import (
    recover_corrs_offset_scales, 
    Homo_2d_pts, 
    cal_corr_F_and_mean_sd, 
    calc_sampson_dist,
)

from utils.img_process import (
    img_crop_direct, 
    img_crop_with_resize, 
    img_crop_fix_aspect_ratio
)

from utils.vis import plot_matches_lists_ud
from utils.common import test_dir_if_not_create

from .abstract_gam import AbstractGeoAreaMatcher

class PRGeoAreaMatcher(AbstractGeoAreaMatcher):
    """ # PRGeoAMer (Geometric Area Matcher with Predictor & Rejector)
    Args:
        ori_imgs: should be the same size of `crop_from_size'
        area_from_size: the size of img which is the source of area
        crop_from_size: the size of img which is cropped to area-imgs fed into PMer
        eval_from_size: the size of img which is used to evaluate the point matches (depth & K size)
            output corrs need to be put in this size
        areas: output of SAMer
    Returns:
        corrs, inliers, F
    ## Funcs:
        1. Predictor
        2. Rejector
    """
    def __init__(self,
        area_from_size_W,
        area_from_size_H,
        crop_size_W,
        crop_size_H,
        crop_from_size_W,
        crop_from_size_H,
        eval_from_size_W,
        eval_from_size_H,
        std_match_num,
        valid_inside_area_match_num,
        filter_area_num,
        reject_out_area_flag,
        adaptive_size_thd,
        alpha_list,
        datasetName="ScanNet", 
        verbose=0) -> None:
        """ initial with imgs
        """
        self._name = "RawGeoAMer"
        self.datasetName = datasetName
        self.reject_outarea = reject_out_area_flag

        ## area matches come from the area_from_size
        self.area_from_size_W, self.area_from_size_H = area_from_size_W, area_from_size_H
        ## to get point matches within areas, crop area with crop_size from original image with crop_from_size
        self.crop_size_W, self.crop_size_H = crop_size_W, crop_size_H
        self.crop_from_size_W, self.crop_from_size_H = crop_from_size_W, crop_from_size_H
        ## eval MMA, PoseAUC, etc. on eval_from_size
        self.eval_from_size_W, self.eval_from_size_H = eval_from_size_W, eval_from_size_H

        self.std_match_num = std_match_num
        self.valid_inside_area_match_num = valid_inside_area_match_num
        self.inlier_thd_mode = 0 # only 0-average is supported
        
        self.ori_doubt_areas0 = []
        self.ori_doubt_areas1 = []

        self.load_PMatcher_flag = False
        self.predict_AM_flag = False

        self.draw_verbose = verbose
        self.initialized = False
                
        self.alpha_list = OmegaConf.to_container(alpha_list)
        assert type(self.alpha_list) == list, f"alpha_list should be a list, but {self.alpha_list} type is {type(self.alpha_list)}"

        self.filter_area_num = filter_area_num
        self.adaptive_size_thd = adaptive_size_thd
        self.draw_adap = self.draw_verbose
        self.entire_img_corrs_list = []

        self.ori_img_corrs = []

    ######################### HYDRA START #####################
    def name(self):
        return self._name

    def init_dataloader(self, dataloader):
        """
        """
        self.name0 = dataloader.image_name0
        self.name1 = dataloader.image_name1
        self.scene_name = dataloader.scene_name

        # load imgs
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
            # load images with resize and padding, update the crop/eval_from_size
            # these datasets, eval and crop on the original images
            self.ori_img0, self.ori_img1, self.scale0, self.scale1 = dataloader.load_images()
            self.crop_from_size_W0, self.crop_from_size_H0 = self.ori_img0.shape[1], self.ori_img0.shape[0]
            self.eval_from_size_W0, self.eval_from_size_H0 = self.ori_img0.shape[1], self.ori_img0.shape[0]

            self.crop_from_size_W1, self.crop_from_size_H1 = self.ori_img1.shape[1], self.ori_img1.shape[0]
            self.eval_from_size_W1, self.eval_from_size_H1 = self.ori_img1.shape[1], self.ori_img1.shape[0]

            self.color0, self.color1 = self.ori_img0, self.ori_img1
            
        # ScanNet, KITTI, MatterPort3D
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            # ori_imgs are used for crop
            self.ori_img0, self.ori_img1, self.img_scale0, self.img_scale1 = dataloader.load_images(self.crop_from_size_W, self.crop_from_size_H)

            # color0, color1 are used for eval, load for their scale and resize the following Ks
            self.color0, self.color1, self.scale0, self.scale1 = dataloader.load_images(self.eval_from_size_W, self.eval_from_size_H)

            # fill the eval size, which is used in sampler
            self.eval_from_size_W0, self.eval_from_size_W1 = self.eval_from_size_W, self.eval_from_size_W
            self.eval_from_size_H0, self.eval_from_size_H1 = self.eval_from_size_H, self.eval_from_size_H
        else:
            raise NotImplementedError(f"datasetName '{self.datasetName}' not supported")
        

        # load geos
        self.depth0 = None
        self.depth1 = None
        self.depth_factor = None
        self.K0, self.K1 = dataloader.load_Ks(self.scale0, self.scale1)
        self.pose0, self.pose1 = dataloader.load_poses()
        
        if self.datasetName == "ScanNet" or self.datasetName == "MatterPort3D":
            self.depth0, self.depth1 = dataloader.load_depths()
            self.depth_factor = dataloader.get_depth_factor()
        elif self.datasetName == "YFCC":
            # turn ndarray to matrix, YFCC need to inverse
            self.pose0 = self.pose0.I
            self.pose1 = self.pose1.I
        elif self.datasetName == "KITTI" or self.datasetName == "ETH3D" or self.datasetName == "MegaDepth":
            pass
        else:
            raise NotImplementedError
    
    def load_point_matcher(self, point_matcher):
        """
        """
        self.matcher = point_matcher
    
    def load_ori_corrs(self, ori_corrs):
        """
        """
        self.ori_img_coors = ori_corrs
    
    def init_gam(self,
        dataloader,
        point_matcher,
        ori_corrs,
        out_path,
        ):
        """
        """
        self.load_point_matcher(point_matcher)
        self.init_dataloader(dataloader)
        self.load_ori_corrs(ori_corrs)
        self.set_outpath(out_path)
        self.initialized = True

        if self.draw_verbose == 1 or self.draw_adap == 1:
            self.draw_ori_img()
            test_dir_if_not_create(self.out_path)

    def doubtful_area_match_predict(self, doubt_match_pairs):
        """ 
        """
        if not self.initialized:
            logger.warning("not initialized")
            raise NotImplementedError
        predicted_areas0, predicted_areas1 = self.predict_area_match_main_flow(doubt_match_pairs)

        return predicted_areas0, predicted_areas1

    def geo_area_matching_refine(self, 
        matched_areas0, 
        matched_areas1):
        """ 
        """
        if not self.initialized:
            logger.warning("not initialized")
            raise NotImplementedError
        
        alpha_corrs_dict, alpha_inlier_idxs_dict = self.rejection_by_samp_dist_flow(matched_areas0, matched_areas1)

        return alpha_corrs_dict, alpha_inlier_idxs_dict, None

    ######################### HYDRA END #####################

    def draw_ori_img(self):
        """
        """
        cv2.imwrite(os.path.join(self.out_path, "Geo_ori0_.jpg"), self.ori_img0)
        cv2.imwrite(os.path.join(self.out_path, "Geo_ori1_.jpg"), self.ori_img1)


    """ Predictor Part===========================================================================================================================
    """
    def load_doubt_candis(self, areas0, areas1):
        """
        Args:
            area = [u_min, u_max, v_min, v_max]
        """
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
            self.ori_doubt_areas0 = self.tune_area_list_size(areas0, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W0, self.crop_from_size_H0)
            self.ori_doubt_areas1 = self.tune_area_list_size(areas1, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W1, self.crop_from_size_H1)
        else:
            self.ori_doubt_areas0 = self.tune_area_list_size(areas0, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
            self.ori_doubt_areas1 = self.tune_area_list_size(areas1, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H) 
        self.len_area0 = len(self.ori_doubt_areas0)
        self.len_area1 = len(self.ori_doubt_areas1)
        logger.info(f"load doubted {len(areas0)} area in areas0")
        logger.info(f"load doubted {len(areas1)} area in areas1")

    def gene_doubt_match_idx(self):
        """
        WorkFlow:
            n v.s. n 
            n v.s. m --> n v.s. n & C_m^n
        Returns:
            match_idx: [[(matched_area_idx_pair),...], [one matched situation], ...]
        """
        self.match_idx = []
        idx_list0 = [i for i in range(self.len_area0)]
        idx_list1 = [i for i in range(self.len_area1)]

        if self.len_area0 == self.len_area1:
            self.match_idx = self._gene_match_idx_nvn(idx_list0, idx_list1)
        elif self.len_area0 > self.len_area1:
            temp_set = list(itertools.combinations(idx_list0, self.len_area1))
            same_len_idx_list0 = [list(x) for x in temp_set]

            for same_len_candi0 in same_len_idx_list0:
                self.match_idx += self._gene_match_idx_nvn(same_len_candi0, idx_list1)
        elif self.len_area1 > self.len_area0:
            temp_set = list(itertools.combinations(idx_list1, self.len_area0))
            same_len_idx_list1 = [list(x) for x in temp_set]

            for same_len_candi1 in same_len_idx_list1:
                self.match_idx += self._gene_match_idx_nvn(idx_list0, same_len_candi1)
        
        logger.info(f"get match idx as {self.match_idx}")
        return self.match_idx
    
    def _gene_match_idx_nvn(self, idx_list0, idx_list1):
        """ mind the order
        """
        assert len(idx_list0) == len(idx_list1)

        matched_idx = [list(zip(x, idx_list1))  for x in itertools.permutations(idx_list0, len(idx_list1))]
        # logger.info(f"get matched index list as {matched_idx}")

        return matched_idx

    def match_all_doubt_areas(self):
        """ match to get all corrs according to match idx
        Args:
            area = [u_min, u_max, v_min, v_max]
            self.ori_area0
            self.ori_area1
            self.match_idx = [[(area_idx0, area_idx1), (...), (...)], [other match area situation]]
        Returns:
            self.corrs_doubt_all = [[[area_pair_corrs_ori: list], ...], [other area match situation]]
        """
        self.corrs_doubt_all = []
        logger.info(f"Got {len(self.match_idx)} match situations and each situation got {len(self.match_idx[0])} match pairs")
        for i, match_situation in enumerate(self.match_idx):
            self.corrs_doubt_all.append([])
            for j, match_pair in enumerate(match_situation):
                idx0 = match_pair[0]
                idx1 = match_pair[1]
                temp_corrs = self.match_area_pair_mind_size(self.ori_doubt_areas0[idx0], self.ori_doubt_areas1[idx1], name=f"sub_corr_{i}_{j}")
                # temp_corrs = self.match_area_pair(self.ori_doubt_areas0[idx0], self.ori_doubt_areas1[idx1], name=f"doubt_corr_{i}_{j}")
                self.corrs_doubt_all[i].append(temp_corrs)
                if len(temp_corrs) == 0: continue
                if self.draw_verbose == 1:
                    self.draw_area_match_res(temp_corrs, f"matches_{i}_{j}")
                logger.info(f"match for pair {idx0} in img0 and {idx1} in img1 done and get {len(temp_corrs)} correspondenses")

        return self.corrs_doubt_all

    def doubt_area_match_predict(self):
        """
        Args:
            self.corrs_doubt_all: [area match situations]
                area match situation: [corrs of a match area pairs]
                corrs: [u0, v0, u1, v1]s
        Returns:
            self.predicted_best_match_pair = [[matched_area0_idx, matched_area1_idx]s]
        """
        sampson_dists = []

        for i, situ_corrs in enumerate(self.corrs_doubt_all):
            temp_dist = self.calc_geo_consistency_single_situ(situ_corrs)
            logger.info(f"calc sampson distance = {temp_dist} for {i}-th area match pairs")
            sampson_dists.append(temp_dist)

        best_idx = sampson_dists.index(min(sampson_dists))

        best_match_pair = self.match_idx[best_idx]
        logger.info(f"best match pair is {best_match_pair}")
        self.predicted_best_match_pair = best_match_pair

        if self.draw_verbose == 1:
            self.draw_doubt_area_match(self.predicted_best_match_pair, "best_match_area")

    def draw_all_match_situ(self):
        """
        """
        for i, match_situ in enumerate(self.match_idx):
            self.draw_doubt_area_match(match_situ, str(i))

    def draw_doubt_area_match(self, match_pair_idx, name):
        """
        Args:
            match_pair_idx: [[area0_idx, area1_idx]s]
        """
        l_label = len(match_pair_idx)
        label_color_dict = {}

        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, l_label)))

        for i in range(l_label):
            c = cmap(i)
            label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
        
        W0, H0 = self.ori_img0.shape[1], self.ori_img0.shape[0]
        W1, H1 = self.ori_img1.shape[1], self.ori_img1.shape[0]

        H_s, W_s = max(H0, H1), W0+W1
        out = 255 * np.ones((H_s, W_s, 3), np.uint8)
        
        object_img0 = self.ori_img0.copy()
        object_img1 = self.ori_img1.copy()

        out[:H0, :W0, :] = object_img0
        out[:H1, W0:, :] = object_img1 
        
        for i, [area0_idx, area1_idx] in enumerate(match_pair_idx):
            patch0 = [int(x) for x in self.ori_doubt_areas0[area0_idx]]
            patch1 = [int(x) for x in self.ori_doubt_areas1[area1_idx]]
 
            patch1_s = [patch1[0]+W0, patch1[1]+W0, patch1[2], patch1[3]]

            # logger.info(f"patch0 are {patch0[0]}, {patch0[1]}, {patch0[2]}, {patch0[3]}")
            # logger.info(f"patch1 are {patch1_s[0]}, {patch1_s[1]}, {patch1_s[2]}, {patch1_s[3]}")

            cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(label_color_dict[i]), 2)
            cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), label_color_dict[i], 2)

            line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
            line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

            cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=label_color_dict[i], thickness=1, lineType=cv2.LINE_AA)

        out = cv2.resize(out, (self.eval_from_size_W*2, self.eval_from_size_H))        
        cv2.imwrite(os.path.join(self.out_path, "doubt_match_pred_" + name + ".jpg"), out)

    def calc_geo_consistency_single_situ(self, single_situation_corrs):
        """
        Args:
            single_situation_corrs: [[matches_of_a_pair], [], ...]
                matches_of_a_pair: [[]]
            F_from: int -- use which pair to calc F
        """
        if len(single_situation_corrs) == 0:
            logger.warning(f"this situation got no matched pair")
            return 1e8
        lens = []
        for corrs in single_situation_corrs:
            lens.append(len(corrs))
        
        max_idx = lens.index(max(lens))
        if lens[max_idx] < 10: 
            logger.warning("this situation got not enough corrs")
            return 1e8


        # calc F first
        corrs_F = single_situation_corrs[max_idx]
        F = self.calc_F(corrs_F)

        corrs_other_np = []
        for i, corrs in enumerate(single_situation_corrs):
            # if i == F_from: continue # use all corrs calc sampson dist
            corrs_other_np += corrs

        if len(corrs_other_np) <= self.valid_inside_area_match_num*1.5:
            logger.warning("this situation got not enough corrs")
            return 1e8

        corrs_other_np = np.array(corrs_other_np)
        logger.info(f"other corrs shape is {corrs_other_np.shape}")
        
        samp_dist = self.calc_sampson(F, corrs_other_np)

        return samp_dist
    
    def predict_area_match_main_flow(self, doubt_match_pairs):
        """ main flow for doubt area match predict based on geometry consistency
        Args:
            doubt_match_pairs: [doubt_match_pair, ...]
            doubt_match_pair: [[area0s], [area1s]]
        Returns:
            self.predicted_matched_area0/1s = [areas]
                area = [u_min, u_max, v_min, v_max]
        """
        logger.info(f"\n***\nstart perform doubt area match prediction.\n***")
        self.predicted_matched_area0s = []
        self.predicted_matched_area1s = []

        if len(doubt_match_pairs) == 0:
            logger.info(f"No doubt area matches")
            return [], []

        assert self.load_PMatcher_flag, "PMatcher not loaded"

        for i, doubt_area_pair in enumerate(doubt_match_pairs):
            temp_doubt_area0 = doubt_area_pair[0]
            temp_doubt_area1 = doubt_area_pair[1]
            self.load_doubt_candis(temp_doubt_area0, temp_doubt_area1) # areas in crop size
            self.gene_doubt_match_idx()
            self.match_all_doubt_areas()
            self.doubt_area_match_predict()
            assert len(self.predicted_best_match_pair[0]) == 2, f"strange match pair length: {len(self.predicted_best_match_pair[0])}"
            for best_match_idx_pair in self.predicted_best_match_pair:
                temp_match_area_idx0, temp_match_area_idx1 = best_match_idx_pair[0], best_match_idx_pair[1]
                self.predicted_matched_area0s.append(temp_doubt_area0[temp_match_area_idx0])
                self.predicted_matched_area1s.append(temp_doubt_area1[temp_match_area_idx1])
        
        logger.info(f"predict area match for {i+1} doubt match pairs.\n Achieve {len(self.predicted_matched_area0s)} area match pairs after prediction")

        for area0, area1 in zip(self.predicted_matched_area0s, self.predicted_matched_area1s):
            assert len(area0) == 4, f"invalid area: {area0}"
            assert len(area1) == 4, f"invalid area: {area1}"

        self.predict_AM_flag = True

        return self.predicted_matched_area0s, self.predicted_matched_area1s

    def get_predicted_area_match_corrs(self):
        """
        Returns:
            corrs: [match_pair_corrs]
                match_paur_corrs: [corrs]
        """
        self.predicted_AM_corrs = []
        if not self.predict_AM_flag:
            logger.warning("prediction is not performed")
            return []
        else:
            for i, area0 in enumerate(self.predicted_matched_area0s):
                temp_corrs = self.match_area_pair_mind_size(area0, self.predicted_matched_area1s[i], f"pred_AM_{i}")
                # temp_corrs = self.match_area_pair(area0, self.predicted_matched_area1s[i], f"pred_AM_{i}")
                self.predicted_AM_corrs.append(temp_corrs)
        
        return self.predicted_AM_corrs

    """ Rejector Part===========================================================================================================================
    Args:
        all matched areas
    Returns:
        corrs from inlier area matches
    """
    def load_all_area_matches(self, matched_area0s, matched_area1s):
        """
        Args:
            matched_area0/1s: [areas]
        Returns:
            self.rejecting_matched_area0s: in crop from size
            self.rejecting_matched_area1s
        """
        assert len(matched_area0s) == len(matched_area1s), "invalid pair"

        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
            matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W0, self.crop_from_size_H0)
            matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W1, self.crop_from_size_H1)
        else:
            matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
            matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)

        for area0, area1 in zip(matched_area0s, matched_area1s):
            assert len(area0) == 4, f"invalid area: {area0}"
            assert len(area1) == 4, f"invalid area: {area1}"

        logger.info(f"load {len(matched_area0s)} matched area pairs")

        self.rejecting_matched_area0s = matched_area0s
        self.rejecting_matched_area1s = matched_area1s

    def _match_on_entire_img(self):
        """
        Returns:
            entire_img_corrs_list: [[corrs], ...] in eval_from_size
            entire_img_corrs_np: np.ndarray of shape (N, 4) - [u0, v0, u1, v1]
        """
        match_num = self.std_match_num

        # # if self.datasetName == "MegaDepth":
        # if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
        #     ori_img0_resized = cv2.resize(self.ori_img0, (self.eval_from_size_W0, self.eval_from_size_H0))
        #     ori_img1_resized = cv2.resize(self.ori_img1, (self.eval_from_size_W1, self.eval_from_size_H1))
        # else:
        #     ori_img0_resized = cv2.resize(self.ori_img0, (self.eval_from_size_W, self.eval_from_size_H))
        #     ori_img1_resized = cv2.resize(self.ori_img1, (self.eval_from_size_W, self.eval_from_size_H))
        
        ori_img0_resized = cv2.resize(self.ori_img0, (self.crop_size_W, self.crop_size_H))
        ori_img1_resized = cv2.resize(self.ori_img1, (self.crop_size_W, self.crop_size_H))

        ## perform point matching
        corr_num = match_num
        self.matcher.set_corr_num_init(corr_num)
        try:
            self.matcher.match(ori_img0_resized, ori_img1_resized)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []

        # get entire img corrs
        temp_corrs = self.matcher.return_matches()

        # tune to eval_from_size
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
            temp_corrs = self.tune_corrs_size_diff(temp_corrs, self.crop_size_W, self.crop_size_W, self.crop_size_H, self.crop_size_H, self.eval_from_size_W0, self.eval_from_size_W1, self.eval_from_size_H0, self.eval_from_size_H1)
        else:   
            temp_corrs = self.tune_corrs_size(temp_corrs, self.crop_size_W, self.crop_size_H, self.eval_from_size_W, self.eval_from_size_H)

        temp_corrs_np = np.array(temp_corrs)
        if temp_corrs_np.shape[0] == 0: 
            logger.warning("No valid matches")
            return [], None
        
        return temp_corrs, temp_corrs_np

    def _match_on_entire_img_on_eval_size(self):
        """
        Returns:
            entire_img_corrs_list: [[corrs], ...]
            entire_img_corrs_np: np.ndarray of shape (N, 4) - [u0, v0, u1, v1]
        """
        match_num = self.std_match_num
        # if self.datasetName == "MegaDepth":
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
            ori_img0_resized = cv2.resize(self.ori_img0, (self.eval_from_size_W0, self.eval_from_size_H0))
            ori_img1_resized = cv2.resize(self.ori_img1, (self.eval_from_size_W1, self.eval_from_size_H1))
        else:
            ori_img0_resized = cv2.resize(self.ori_img0, (self.eval_from_size_W, self.eval_from_size_H))
            ori_img1_resized = cv2.resize(self.ori_img1, (self.eval_from_size_W, self.eval_from_size_H))

        ## perform point matching
        corr_num = match_num
        self.matcher.set_corr_num_init(corr_num)
        try:
            self.matcher.match(ori_img0_resized, ori_img1_resized)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []

        # get entire img corrs
        temp_corrs = self.matcher.return_matches()
        temp_corrs_np = np.array(temp_corrs)
        if temp_corrs_np.shape[0] == 0: 
            logger.warning("No valid matches")
            return [], None
        
        return temp_corrs, temp_corrs_np
    
    def match_all_rejecting_areas(self):
        """
        Args:
            self.rejecting_matched_area0/1s
        Returns:
            self.rejecting_all_corrs: [area_match_corrs] in eval_from_size
        """
        self.rejecting_all_corrs = []

        # Adaptive A2PM part ###########################################
        # if only one area; perform point matching on entire img and screen 
        if len(self.rejecting_matched_area0s) <= self.filter_area_num:
            logger.warning(f"only {len((self.rejecting_matched_area0s))} area pair, perform point matching on entire img and screen")
            if (len(self.ori_img_corrs)==0):
                self.entire_img_corrs_list, entire_img_corr_np = self._match_on_entire_img()
            else:
                entire_img_corr_np = np.array(self.ori_img_corrs)
                self.entire_img_corrs_list = self.ori_img_corrs

            if len(self.entire_img_corrs_list) <= 100:
                logger.warning(f"entire img corrs num is {len(self.entire_img_corrs_list)}, too small")
                return []

            # TODO for more areas!

            ## get areas
            temp_area0 = self.rejecting_matched_area0s[0]
            temp_area1 = self.rejecting_matched_area1s[0]

            ## get refined corrs
            temp_corrs = self.refine_image_corrs_by_single_area(entire_img_corr_np, temp_area0, temp_area1)
            self.rejecting_all_corrs.append(temp_corrs)
            
            return self.rejecting_all_corrs
        ## end of adaptive A2PM part ###########################################       

        for i, temp_area0 in enumerate(self.rejecting_matched_area0s):
            temp_corrs = self.match_area_pair_mind_size(temp_area0, self.rejecting_matched_area1s[i], f"rejecting_{i}") # corrs in eval_from_size

            # areas are in crop_from_size, tune to eval_from_size
            if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
                area0_in_eval_size = self.tune_area_size(temp_area0, self.crop_from_size_W0, self.crop_from_size_H0, self.eval_from_size_W0, self.eval_from_size_H0)
                area1_in_eval_size = self.tune_area_size(self.rejecting_matched_area1s[i], self.crop_from_size_W1, self.crop_from_size_H1, self.eval_from_size_W1, self.eval_from_size_H1)
            else:
                area0_in_eval_size = self.tune_area_size(temp_area0, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)
                area1_in_eval_size = self.tune_area_size(self.rejecting_matched_area1s[i], self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)

            # filter corrs outside area
            if self.reject_outarea:
                temp_corrs = self.reject_corrs_outside_area(temp_corrs, area0_in_eval_size, area1_in_eval_size)

            self.rejecting_all_corrs.append(temp_corrs)
            # self.rejecting_all_corrs.append(self.match_area_pair(temp_area0, self.rejecting_matched_area1s[i], f"rejecting_{i}"))
        
        logger.info(f"matched {i} area pairs")

        return self.rejecting_all_corrs
    
    def reject_corrs_outside_area(self, corrs, area0, area1):
        """
        Args:
            corrs in eval_from_size
            areas in eval_from_size
        """
        refined_corrs = []
        for corr in corrs:
            if self.is_corr_inside_area(corr, area0, area1):
                refined_corrs.append(corr)
            else:
                # logger.warning(f"corr {corr} is outside area {area0} and {area1}")
                pass
        
        return refined_corrs

    def refine_image_corrs_by_single_area(self, corrs, area0, area1):
        """
        Args:
            corrs: np.ndarray [[u0,v0,u1,v1]...] in eval_from_size
            area: [u_min, u_max, v_min, v_max] in crop_from_size
        Returns:
            refined_corrs: [corrs] list
        """
        logger.info(f"refine image corrs by single area, befre refine: {corrs.shape}")

        corrs_inside_area = []
        for corr in corrs:
            if self.is_corr_inside_area(corr, area0, area1):
            # if True:
                corrs_inside_area.append(corr)
        
        corr_num = len(corrs)
        if corr_num <= 100: return corrs
        corr_inside_area_num = len(corrs_inside_area)
        if corr_inside_area_num <= 0.5 * corr_num:
            logger.warning(f"corrs_inside_area_num: {corr_inside_area_num} is too small")
            # turn np.ndarray to list
            corrs = corrs.tolist()
            logger.info(f"refine image corrs by single area, after refine: {len(corrs)}")
            return corrs
        else:
            refined_corrs = corrs_inside_area
            # calc F and mean sd for corrs inside area
            F_inside, sd_inside = cal_corr_F_and_mean_sd(corrs_inside_area)
            corrs_inside_area = np.array(corrs_inside_area)
            corrs_inside_area = corrs_inside_area.tolist()

            # calc sd for corrs outside area
            for corr in corrs:
                corr_list = corr.tolist()
                try:
                    if corr_list not in corrs_inside_area:
                        corr_np = np.array([corr])
                        sd = calc_sampson_dist(F_inside, corr_np)
                        if sd <= sd_inside:
                            refined_corrs.append([corr[0], corr[1], corr[2], corr[3]])
                except ValueError:
                    logger.exception(f"at What?")
                    continue
            logger.info(f"refine image corrs by single area, after refine: {len(refined_corrs)}")
            return refined_corrs

    def is_corr_inside_area(self, corr, area0, area1):
        """
        """
        u0, v0, u1, v1 = corr
        u_min, u_max, v_min, v_max = area0
        uv0_area0 = False
        if u_min <= u0 <= u_max and v_min <= v0 <= v_max:
            uv0_area0 = True
        else:
            uv0_area0 = False
        
        u_min, u_max, v_min, v_max = area1
        uv1_area1 = False
        if u_min <= u1 <= u_max and v_min <= v1 <= v_max:
            uv1_area1 = True
        else:
            uv1_area1 = False
        
        if uv0_area0 and uv1_area1:
            return True
        else:
            return False

    def rejection_by_samp_dist_flow(self, matched_area0s, matched_area1s):
        """ perform area match pairs rejection through total corrs sampson dist
        TODO
            debug when corrs num is 0 # DONE
        
        Returns:
            alpha-corr-dict = 
            {
                alpha: corrs [corr, ...]
                    - corr: [u0, v0, u1, v1]
            }
        """
        logger.info(f"\n***\nstart perform geo-consistency-based area match pairs rejection.\n***")
        # logger.debug(f"areas got : {matched_area0s} \n {matched_area1s}")

        self.load_all_area_matches(matched_area0s, matched_area1s) # areas in area_from_size are loaded here and tuned to crop_from_size

        alpha_corrs_dict = defaultdict(list)
        alpha_idxs_dict = defaultdict(list)
        F_list = []

        
        if self.draw_verbose == 1:
            self.draw_match_areas(self.rejecting_matched_area0s, self.rejecting_matched_area1s, "before_rejection")

        self.match_all_rejecting_areas()
        area_match_num = len(self.rejecting_all_corrs)
        logger.warning(f"area_match_num: {area_match_num}")

        if area_match_num == 0:
            logger.warning(f"no area match pairs found")
            self.best_F = np.zeros((3,3))
            self.alpha_inlier_corrs_dict = alpha_corrs_dict
            # get entire img corrs in eval size
            if len(self.ori_img_corrs) == 0:
                self.ori_img_corrs, _ = self._match_on_entire_img()
            
            # fill with original corrs
            for alpha in self.alpha_list:
                alpha_corrs_dict[alpha] = [self.ori_img_corrs]
                alpha_idxs_dict[alpha] = [0]

            self.alpha_idxs_dict = alpha_idxs_dict
            return self.alpha_inlier_corrs_dict, self.best_F, self.alpha_idxs_dict


        if area_match_num == 1: # may not work in adaptive A2PM
            for alpha in self.alpha_list:
                alpha_corrs_dict[alpha] = self.rejecting_all_corrs
                alpha_idxs_dict[alpha] = [0]
            self.best_F = np.zeros((3,3))
            self.alpha_inlier_corrs_dict = alpha_corrs_dict
            self.alpha_idxs_dict = alpha_idxs_dict
            # return self.alpha_inlier_corrs_dict, self.best_F, self.alpha_idxs_dict


        alpha_thd_dict = self.calc_inlier_thd(self.rejecting_all_corrs, self.inlier_thd_mode, self.alpha_list)
        logger.info(f"inlier thd = {alpha_thd_dict}")
        
        total_corrs_np = self.stack_corrs(copy.deepcopy(self.rejecting_all_corrs))
        logger.warning(f"total corrs num = {len(total_corrs_np)}")
        if len(total_corrs_np) == 0: 
            logger.warning("No enough corners")
            return None, None, None


        sampson_dist_min = 1e8
        best_F_idx = 100
        
        if type(self.rejecting_all_corrs) != list:
            self.rejecting_all_corrs = self.rejecting_all_corrs.tolist()

        # calc F and sampson dist for each area match pair
        logger.warning(f"corr sets num is {len(self.rejecting_all_corrs)}")
        if len(self.rejecting_all_corrs) > 1:
            for i, src_corrs_ in enumerate(self.rejecting_all_corrs):
                # logger.warning(f"calc geo consistency for pair {i} for {len(self.rejecting_all_corrs)}")
                # logger.info(f"calc geo consistency for pair {i} with self.rejecting_all_corrs = \n{self.rejecting_all_corrs[-1]}")

                src_corrs = copy.deepcopy(src_corrs_)
                if len(src_corrs) < self.valid_inside_area_match_num:
                    logger.warning(f"matched area {i} has not enough corrs")
                    F_list.append(0)
                    continue

                src_F = self.calc_F(src_corrs)
                F_list.append(src_F)
                temp_sd = self.calc_sampson(src_F, total_corrs_np)
                logger.info(f"get sampson distance for match pair {i} = {temp_sd}")

                for alpha_temp in self.alpha_list:
                    temp_thd = alpha_thd_dict[alpha_temp]
                    if temp_sd <= temp_thd:
                        logger.info(f"for alpha {alpha_temp} get inlier match pair with sampson distance = {temp_sd}")
                        alpha_corrs_dict[alpha_temp].append(src_corrs)
                        alpha_idxs_dict[alpha_temp].append(i)
                        if temp_sd < sampson_dist_min:
                            best_F_idx = i
                            sampson_dist_min = temp_sd
        
        # logger.info(f"get {len(inlier_corrs)} inliers whose idx are {inlier_idxs}")

        # Adaptive A2PM begin #######################################
        for alpha_temp in self.alpha_list:
            after_reject_area0s = []
            after_reject_area1s = []

            for idx in alpha_idxs_dict[alpha_temp]:
                after_reject_area0s.append(self.rejecting_matched_area0s[idx])
                after_reject_area1s.append(self.rejecting_matched_area1s[idx])

            if self.draw_verbose == 1:
                self.draw_match_areas(after_reject_area0s, after_reject_area1s, f"phi{alpha_temp}_after_rejection")

            # calc the area size in two images
            if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
                size_ratio0 = self.calc_area_size_in_ori_ratio(after_reject_area0s, self.crop_from_size_W0, self.crop_from_size_H0)
                size_ratio1 = self.calc_area_size_in_ori_ratio(after_reject_area1s, self.crop_from_size_W1, self.crop_from_size_H1)
            else:
                size_ratio0 = self.calc_area_size_in_ori_ratio(after_reject_area0s, self.crop_from_size_W, self.crop_from_size_H)
                size_ratio1 = self.calc_area_size_in_ori_ratio(after_reject_area1s, self.crop_from_size_W, self.crop_from_size_H)

            if size_ratio0 <= self.adaptive_size_thd or size_ratio1 <= self.adaptive_size_thd:

                # perform adaptive A2PM
                # logger.critical(f"for alpha {alpha_temp} size ratio of areas in two images are {size_ratio0} and {size_ratio1}, less than {self.adaptive_size_thd}, so perform adaptive A2PM")
                
                # get entire img corrs
                if len(self.ori_img_corrs) == 0:
                    self.ori_img_corrs, _ = self._match_on_entire_img()

                # get corrs in the area
                areas_corrs = alpha_corrs_dict[alpha_temp] # a list of corrs_list
                areas_corrs_in_a_list = self._list_of_corrs2corr_list(areas_corrs) # a list of corrs

                # visulization
                if self.draw_adap == 1:
                    self.draw_match_areas_with_corrs(after_reject_area0s, after_reject_area1s, areas_corrs_in_a_list, f"phi{alpha_temp}_before_adaptive_rejection_areas")

                
                if len(areas_corrs_in_a_list) <= self.valid_inside_area_match_num:
                    logger.warning(f"area corrs in alpha {alpha_temp} is too less, {len(areas_corrs_in_a_list)} < {self.valid_inside_area_match_num}")
                    alpha_corrs_dict[alpha_temp] = [self.ori_img_corrs]
                    continue

                # calc the inlier thd
                F_inlier, samp_dist_inlier = cal_corr_F_and_mean_sd(areas_corrs_in_a_list)

                # write down the scene name for counting 
                ## change the out_path to its parent folder
                # count_out_path = os.path.dirname(self.out_path)
                # count_out_path = os.path.join(count_out_path, "MMARatios")
                # count_file_path = os.path.join(count_out_path, f"ada_count_{alpha_temp}.txt")
                # with open(count_file_path, "a") as f:
                #     f.write(f"{self.out_path.split('/')[-1]}\n")

                # collect inlier corrs from entire image corrs
                collect_out_area_corrs = []
                for corr in self.ori_img_corrs:
                    # test corr's sampson distance
                    samp_dist_temp = calc_sampson_dist(F_inlier, np.array([corr]))
                    if samp_dist_temp <= samp_dist_inlier*2:
                        collect_out_area_corrs.append(corr)

                if len(collect_out_area_corrs) > 0:
                    logger.info(f"get {len(collect_out_area_corrs)} inlier corrs from entire image corrs")
                    alpha_corrs_dict[alpha_temp].append(collect_out_area_corrs)

                # visulization
                if self.draw_adap == 1:
                    # get corrs in the area
                    areas_corrs_after = alpha_corrs_dict[alpha_temp] # a list of corrs_list
                    areas_corrs_in_a_list_after = self._list_of_corrs2corr_list(areas_corrs_after) # a list of corrs
                    self.draw_match_areas_with_corrs(after_reject_area0s, after_reject_area1s, areas_corrs_in_a_list_after, f"phi{alpha_temp}_after_adaptive_rejection_areas")
                    
        # Adaptive A2PM end #########################################
            else:
                # draw for unadapted part
                if self.draw_verbose == 1:
                    # get corrs in the area
                    areas_corrs = alpha_corrs_dict[alpha_temp] # a list of corrs_list
                    areas_corrs_in_a_list = self._list_of_corrs2corr_list(areas_corrs) # a list of corrs
                    self.draw_match_areas_with_corrs(after_reject_area0s, after_reject_area1s, areas_corrs_in_a_list, f"phi{alpha_temp}_area_corrs_after_rejection_areas")
                pass
                


        self.aplha_inlier_corrs_dict = alpha_corrs_dict
        if len(F_list) < best_F_idx + 1:
            logger.warning("no F is found")
            self.best_F = None
        else:
            self.best_F = F_list[best_F_idx]
        self.alpha_idxs_dict = alpha_idxs_dict

        return alpha_corrs_dict, alpha_idxs_dict

    def _list_of_corrs2corr_list(self, list_of_corrs):
        """
            [[corrs],...] -> [[corr], ...]
        """
        rt_corrs = []
        for corrs in list_of_corrs:
            rt_corrs += corrs
        
        return rt_corrs

    def calc_area_size_in_ori_ratio(self, area_list, ori_W, ori_H):
        """
        Args:
            area_list: list of area in crop_from_size
        """
        ori_size = ori_H * ori_W
        merge_size = self._calc_merge_size_of_areas(area_list, ori_W, ori_H)
        size_ratio = merge_size / ori_size
        return size_ratio

    def _calc_merge_size_of_areas(self, areas, ori_W, ori_H):
        """
        Args:
            areas: list of [u_min, u_max, v_min, v_max]
        Returns:
            merge_size: size of merged area
        """
        blank = np.zeros((ori_H, ori_W))
        for area in areas:
            logger.debug(f"area = {area} in {ori_W}x{ori_H}")
            u_min, u_max, v_min, v_max = area
            # turn to int
            u_min, u_max, v_min, v_max = int(u_min), int(u_max), int(v_min), int(v_max)
            blank[v_min:v_max, u_min:u_max] = 1
        
        merge_size = np.sum(blank)
        return merge_size
        
    def calc_inlier_thd(self, rejecting_all_corrs, mode=0, alpha_list=[2.5]):
        """
        Args:
            corrs
            mode:
                0. avg: use the average of all area match pair's self-sampson dist 
        Returns:
            alpha-thd-dict
            {alpha: thd}
        """
        assert type(alpha_list) == list
        thd = 0
        alpha_thd_dict = {}
        if mode == 0: # average
            self.self_sd_list = []
            logger.info(f"calc inlier threshold by average mode")
            for i, corrs in enumerate(rejecting_all_corrs):
                if len(corrs) < 100: continue
                F_temp = self.calc_F(corrs)
                temp_corrs_np = np.array(corrs)
                temp_self_sd = self.calc_sampson(F_temp, temp_corrs_np)
                self.self_sd_list.append(temp_self_sd)
                logger.info(f"the {i} match pair calced self sampson dist = {temp_self_sd}")
            
            # if len(self.self_sd_list) == 0: return 0
            for alpha in alpha_list:
                # print(alpha,"\n",np.array(self.self_sd_list).mean())
                if len(self.self_sd_list) == 0: 
                    thd = 0
                else:
                    thd = np.array(self.self_sd_list).mean() * alpha # NOTE alpha
                alpha_thd_dict[alpha] = thd
        
        else:
            logger.warning(f"Unspported threshold mode: {mode}")
            raise NotImplementedError

        return alpha_thd_dict

    def _calc_all_area_match_sampson_dists(self, F, corrs_list):
        """
        """
        logger.info(f"calc all match area corrs sampson dist...")
        sampson_dist_list = []

        for i, corrs in enumerate(corrs_list):
            corrs_np = np.array(corrs)
            temp_sd = self.calc_sampson(F, corrs_np)
            logger.info(f"{i} sampson distance = {temp_sd}")
            sampson_dist_list.append(temp_sd)

        return sampson_dist_list

    """ COMMON Part=============================================================================================================================
    """
    def tune_area_size(self, area, src_W, src_H, dst_W, dst_H):
        """
        Args:
            area: [u_min, u_max, v_min, v_max]
        Returns:
            area
        """
        W_ratio = dst_W / src_W
        H_ratio = dst_H / src_H

        u_min, u_max, v_min, v_max = area

        u_min_ = u_min * W_ratio
        u_max_ = u_max * W_ratio
        v_min_ = v_min * H_ratio
        v_max_ = v_max * H_ratio

        area_ = [u_min_, u_max_, v_min_, v_max_]

        return area_

    def tune_area_list_size(self, area_list, src_W, src_H, dst_W, dst_H):
        """
        """
        rt_area_list = []

        for area in area_list:
            area_ = self.tune_area_size(area, src_W, src_H, dst_W, dst_H)
            rt_area_list.append(area_)
        
        return rt_area_list

    def calc_F(self, corrs):
        """
        Args:
            corrs - [[u0, v0, u1, v1]s]
        """
        corrs = np.array(corrs)
        corrs_num = len(corrs)
        if corrs_num < 8: 
            logger.warning(f"corrs num is {corrs_num}, too small to calc F")
            return None
        corrs_F0 = corrs[:, :2]
        corrs_F1 = corrs[:, 2:]
        logger.info(f"achieve corrs with shape {corrs_F0.shape} == {corrs_F1.shape} to calc F")
        F, mask = cv2.findFundamentalMat(corrs_F0, corrs_F1, method=cv2.FM_RANSAC,ransacReprojThreshold=1, confidence=0.99)
        logger.info(f" calc F as \n {F}")

        return F

    def stack_corrs(self, corr_list):
        """
        Args:
            corrs_list: [[corrs], [corrs], ...]
        Returns:
            rt_corrs: N x 4
        """
        valid_idxs = []
        for i, corrs in enumerate(corr_list):
            if len(corrs) >= self.valid_inside_area_match_num:
                valid_idxs.append(i)
        
        if len(valid_idxs) == 0:
            return []

        rt_corrs = np.array(corr_list[valid_idxs[0]])

        for j in valid_idxs[1:]:
            rt_corrs = np.vstack((rt_corrs, np.array(corr_list[j])))

        return rt_corrs

    def norm_pts(self, pts):
        """
        Args:
            pts: Nx2 [u, v]
        """
        norm_pts = pts.copy()
        N = pts.shape[0]
        W, H = self.eval_from_size_W, self.eval_from_size_H

        for i in range(N):
            norm_pts[i, 0] = (pts[i,0] / W) - 0.5
            norm_pts[i, 1] = (pts[i,1] / H) - 0.5

        return norm_pts 

    def norm_pts_NT(self, pts):
        """ use 8-pts algorithm normalization
        """
        norm_pts = pts.copy()
        N = pts.shape[0]
        
        mean_pts = np.mean(pts, axis=0)
        
        mins_mean_pts = pts - mean_pts

        pts_temp = np.mean(np.abs(mins_mean_pts), axis=0)
        pts_temp+=1e-5

        norm_pts = mins_mean_pts / pts_temp

        logger.info(f"after norm pts shape = {norm_pts.shape}")

        return norm_pts

    def calc_sampson(self, F, corrs):
        """ calc sampson distance as geo consistancy
        Args:
            corrs: nd.array: Nx4
        """
        assert len(corrs.shape) == 2 and corrs.shape[1] == 4, f"invalid shape {corrs.shape}"
        uv0, uv1 = corrs[:, :2], corrs[:, 2:]
        uv0_norm = self.norm_pts_NT(uv0)
        uv1_norm = self.norm_pts_NT(uv1)
        uv0_h, uv1_h = Homo_2d_pts(uv0_norm), Homo_2d_pts(uv1_norm) # N x 3
        samp_dist = 0

        for i in range(corrs.shape[0]):
            samp_dist += self.calc_sampson_1_pt(F, uv0_h[i,:], uv1_h[i,:])
        
        samp_dist /= corrs.shape[0]

        return samp_dist
    
    def calc_sampson_1_pt(self, F, uv0H, uv1H):
        """
        Args:
            uviH: 1 x 3
            F: 3 x 3
        Returns:
            (uv1H^T * F * uv0H)^2 / [(F*uv0H)_0^2 + (F*uv0H)_1^2 + (F^T*uv1H)_0^2 + (F^T*uv1H)_1^2]
        """
        uv0H = uv0H.reshape((1,3))
        uv1H = uv1H.reshape((1,3))

        assert uv0H.shape[0] == 1 and uv0H.shape[1] == 3, f"invalid shape {uv0H.shape}"
        assert uv1H.shape[0] == 1 and uv1H.shape[1] == 3, f"invalid shape {uv1H.shape}"

        # logger.debug(f"calc sampson dist use:\n{uv0H}\n{uv1H}")

        up = np.matmul(uv1H, np.matmul(F, uv0H.reshape(3,1)))[0][0]
        up = up**2
        Fx0 = np.matmul(F, uv0H.T)
        FTx1 = np.matmul(F.T, uv0H.T)
        # logger.info(f"Fx1 = {Fx1}\nFTx0 = {FTx0}")
        down = Fx0[0,0]**2 + Fx0[1,0]**2 + FTx1[0,0]**2 + FTx1[1, 0]**2

        # logger.debug(f"calc sampson dist use {up} / {down}")
        
        dist = up / (down + 1e-5)


        return dist

    def draw_area_match_res(self, temp_corrs, name):
        """
        Args:
            temp_corrs: list [[u0, v0, u1, v1]s] in crop from size
        """
        temp_corrs_crop = self.tune_corrs_size(temp_corrs, self.eval_from_size_W, self.eval_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
        plot_matches_lists_ud(self.ori_img0, self.ori_img1, temp_corrs_crop, self.out_path, name)
    
    def draw_match_areas(self, matched_area0s, matched_area1s, name=""):
        """ draw match areas in crop from size

        """
        # matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
        # matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)

        l_label = len(matched_area0s)
        label_color_dict = {}

        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, l_label)))

        for i in range(l_label):
            c = cmap(i)
            label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
        
        W0, H0 = self.ori_img0.shape[1], self.ori_img0.shape[0]
        W1, H1 = self.ori_img1.shape[1], self.ori_img1.shape[0]
        H_s, W_s = max(H0, H1), W0+W1
        out = 255 * np.ones((H_s, W_s, 3), np.uint8)

        object_img0 = self.ori_img0.copy()
        object_img1 = self.ori_img1.copy()

        out[:H0, :W0, :] = object_img0
        out[:H1, W0:, :] = object_img1 

        for i in range(l_label):
            # patch0 = matched_area0s[i]
            # patch1 = matched_area1s[i]
            patch0 = [int(x) for x in matched_area0s[i]]
            patch1 = [int(x) for x in matched_area1s[i]]

            patch1_s = [patch1[0]+W0, patch1[1]+W0, patch1[2], patch1[3]]

            cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(label_color_dict[i]), 2)
            cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), label_color_dict[i], 2)

            line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
            line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

            cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=label_color_dict[i], thickness=1, lineType=cv2.LINE_AA)

        if self.datasetName != "MegaDepth" and self.datasetName != "YFCC":
            out = cv2.resize(out, (self.eval_from_size_W*2, self.eval_from_size_H))        
        cv2.imwrite(os.path.join(self.out_path, "geo_match_area_" + name+ ".jpg"), out)

    def draw_match_areas_with_corrs(self, matched_area0s, matched_area1s, corrs, name=""):
        """
        Args:
            matched_area0s: [[u_min, u_max, v_min, v_max]s] in crop from size
            matched_area1s: [[u_min, u_max, v_min, v_max]s] in crop from size
            corrs: [[u0, v0, u1, v1]s] in eval from size
        """
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
            corrs_crop = self.tune_corrs_size_diff(corrs, self.eval_from_size_W0, self.eval_from_size_W1, self.eval_from_size_H0, self.eval_from_size_H1, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1)
        # elif self.datasetName == "YFCC":
        #     corrs_crop = self.tune_corrs_size_diff(corrs, self.eval_from_size_W, self.eval_from_size_W, self.eval_from_size_H, self.eval_from_size_H, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1)
        else:
            corrs_crop = self.tune_corrs_size(corrs, self.eval_from_size_W, self.eval_from_size_H, self.crop_from_size_W, self.crop_from_size_H)

        l_label = len(matched_area0s)
        label_area_color_dict = {}

        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, l_label)))

        for i in range(l_label):
            c = cmap(i)
            label_area_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]

        W0, H0 = self.ori_img0.shape[1], self.ori_img0.shape[0]
        W1, H1 = self.ori_img1.shape[1], self.ori_img1.shape[0]
        H_s, W_s = max(H0, H1), W0+W1
        out = 255 * np.ones((H_s, W_s, 3), np.uint8)

        object_img0 = self.ori_img0.copy()
        object_img1 = self.ori_img1.copy()

        out[:H0, :W0, :] = object_img0
        out[:H1, W0:, :] = object_img1 
        # # get paint image
        # W, H = self.crop_from_size_W, self.crop_from_size_H
        # H_s, W_s = H, W*2
        # out = 255 * np.ones((H_s, W_s, 3), np.uint8)

        # object_img0 = self.ori_img0.copy()
        # object_img1 = self.ori_img1.copy()

        # out[:H, :W, :] = object_img0
        # out[:H, W:, :] = object_img1 

        # draw areas
        for i in range(l_label):
            # patch0 = matched_area0s[i]
            # patch1 = matched_area1s[i]
            patch0 = [int(x) for x in matched_area0s[i]]
            patch1 = [int(x) for x in matched_area1s[i]]

            patch1_s = [patch1[0]+W0, patch1[1]+W0, patch1[2], patch1[3]]

            cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(label_area_color_dict[i]), 2)
            cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), label_area_color_dict[i], 2)

            line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
            line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

            cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=label_area_color_dict[i], thickness=1, lineType=cv2.LINE_AA)

        # draw corrs
        corr_color = np.zeros((len(corrs_crop), 3), dtype=np.uint8)
        corr_color[:, 1] = 255

        for match, c in zip(corrs_crop, corr_color):
            c = c.tolist()
            u0, v0, u1, v1 = match
            # print(u0)
            u0 = int(u0)
            v0 = int(v0)
            u1 = int(u1) + W0
            v1 = int(v1)
            cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)
        
        # save
        if self.datasetName != "MegaDepth" or self.datasetName != "YFCC":
            out = cv2.resize(out, (self.eval_from_size_W*2, self.eval_from_size_H))
        cv2.imwrite(os.path.join(self.out_path, "geo_match_area_corrs_" + name+ ".jpg"), out)

    def match_area_pair_mind_size(self, area0, area1, name=""):
        """ TODO test-Done
        Func: 
            0. tune area (self.area_from_size) size to self.crop_from_size # NOTE move to load part
            1. crop self.crop_size from ori img (self.crop_from_size)
            2. use PMer to match
            3. recover corrs to (self.crop_from_size)
            4. put corrs to (self.eval_from_size) for eval
        Returns:
            rt_corrs: [[u0, v0, u1, v1]s] in eval_from_size
        """
        rt_corrs = []

        crop0, scale0, offset0 = img_crop_fix_aspect_ratio(self.ori_img0, area0, self.crop_size_W, self.crop_size_H)
        crop1, scale1, offset1 = img_crop_fix_aspect_ratio(self.ori_img1, area1, self.crop_size_W, self.crop_size_H)

        # cv2.imwrite(os.path.join(self.out_path, name+"crop0.jpg"), crop0)
        # cv2.imwrite(os.path.join(self.out_path, name+"crop1.jpg"), crop1)

        corr_num = int(self.std_match_num)
        self.matcher.set_corr_num_init(corr_num)
        try:
            self.matcher.match(crop0, crop1)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []

        temp_corrs = self.matcher.return_matches()
        temp_corrs = np.array(temp_corrs)
        if temp_corrs.shape[0] == 0: 
            logger.warning("No valid matches")
            return []

        # recover to crop_from_size
        scale_zip = [scale0[0], scale0[1], scale1[0], scale1[1]]
        offset_zip = [offset0[0], offset0[1], offset1[0], offset1[1]]

        rt_corrs = recover_corrs_offset_scales(temp_corrs, offset_zip, scale_zip) # corrs are in the crop_from_size
        
        # self.draw_area_match_res(rt_corrs, name)
        
        # recover to eval size
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC":
        # if self.datasetName == "MegaDepth":
            rt_corrs = self.tune_corrs_size_diff(rt_corrs, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1, self.eval_from_size_W0, self.eval_from_size_W1, self.eval_from_size_H0, self.eval_from_size_H1)
        # elif self.datasetName == "YFCC":
            # rt_corrs = self.tune_corrs_size_diff(rt_corrs, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1, self.eval_from_size_W, self.eval_from_size_W, self.eval_from_size_H, self.eval_from_size_H)
        else:
            rt_corrs = self.tune_corrs_size(rt_corrs, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)

        logger.info(f"achieve {np.array(rt_corrs).shape} corrs for {name}")

        return rt_corrs
    
    def tune_corrs_size(self, corrs, src_W, src_H, dst_W, dst_H):
        """
        """
        rt_corrs = []
        W_ratio = dst_W / src_W
        H_ratio = dst_H / src_H

        for corr in corrs:
            u0, v0, u1, v1 = corr
            u0_ = u0 * W_ratio
            v0_ = v0 * H_ratio
            u1_ = u1 * W_ratio
            v1_ = v1 * H_ratio

            rt_corrs.append([u0_, v0_, u1_, v1_])
        
        return rt_corrs

    def tune_corrs_size_diff(self, corrs, src_W0, src_W1, src_H0, src_H1, dst_W0, dst_W1, dst_H0, dst_H1):
        """
        """
        rt_corrs = []
        W0_ratio = dst_W0 / src_W0
        H0_ratio = dst_H0 / src_H0
        W1_ratio = dst_W1 / src_W1
        H1_ratio = dst_H1 / src_H1

        for corr in corrs:
            u0, v0, u1, v1 = corr
            u0_ = u0 * W0_ratio
            v0_ = v0 * H0_ratio
            u1_ = u1 * W1_ratio
            v1_ = v1 * H1_ratio

            rt_corrs.append([u0_, v0_, u1_, v1_])
        
        return rt_corrs

        

        
        
        

        














