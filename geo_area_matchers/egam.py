import os
import numpy as np
import cv2
from loguru import logger
from collections import defaultdict
import copy
from omegaconf import OmegaConf

from utils.load import (
    load_cv_img_resize, 
)

from utils.geo import (
    recover_corrs_offset_scales, 
    cal_corr_F_and_mean_sd, 
    calc_sampson_dist,
    calc_E_from_corrs,
    recover_F_from_E_K,
    list_of_corrs2corr_list,
    is_corr_inside_area,
    calc_merge_size_of_areas,
    filter_corrs_by_F,
    tune_corrs_size_diff,
    tune_corrs_size,
    
)

from utils.img_process import (
    img_crop_fix_aspect_ratio,
    img_crop_with_padding_improve_resolution,
    img_crop_with_padding_expand_square,
)


from utils.vis import (
    plot_matches_lists_lr,
    plot_matches_lists_ud,
    draw_matched_area_list,
)

from utils.common import test_dir_if_not_create

from .abstract_gam import AbstractGeoAreaMatcher

class EGeoAreaMatcher(AbstractGeoAreaMatcher):
    """ Simplified GAM
        - It estimates essential matrix to calc Sampson Distance to perform GR.
    Inputs:
        matched_area{0/1}s
    Outputs:
        refined_corrs
        inlier_area_idxs
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
        alpha_list,
        adaptive_size_thd,
        valid_inside_area_match_num,
        reject_out_area_flag,
        crop_mode,
        sac_mode,
        occ_size,
        common_occ_flag,
        sampler_name="",
        datasetName="ScanNet", 
        verbose=0) -> None:
        """
        """
        self._name = "E-GeoAreaMatcher"
        self.datasetName = datasetName

        ## area matches come from the area_from_size
        self.area_from_size_W, self.area_from_size_H = area_from_size_W, area_from_size_H
        ## to get point matches within areas, crop area with crop_size from original image with crop_from_size
        self.crop_size_W, self.crop_size_H = crop_size_W, crop_size_H
        self.crop_from_size_W, self.crop_from_size_H = crop_from_size_W, crop_from_size_H
        ## eval MMA, PoseAUC, etc. on eval_from_size
        self.eval_from_size_W, self.eval_from_size_H = eval_from_size_W, eval_from_size_H

        # other configs
        self.std_match_num = std_match_num

        self.sampler_name = sampler_name
        
        ## params for Sampson Distance
        self.alpha_list = OmegaConf.to_container(alpha_list)
        assert type(self.alpha_list) == list, f"alpha_list should be a list, but {self.alpha_list} type is {type(self.alpha_list)}"
        ## params for adpative collection, the total area size thd to determine whether to perform adaptive collection
        self.adaptive_size_thd = adaptive_size_thd

        self.valid_inside_area_match_num = valid_inside_area_match_num
        
        # flags
        self.draw_verbose = verbose
        ## determine whether to reject out-original-area matches (comes from crop areas)
        self.reject_out_area_flag = reject_out_area_flag
        ## crop mode
        # 0: crop directly but mind the aspect ratio
        # 1: crop is padding to improve size
        # 2: crop is padding but expand square
        self.crop_mode = crop_mode
        ## estimator mode
        ### "MAGSAC": MAGSAC++ is used to estimate E
        ### "RANSAC": RANSAC is used to estimate E
        self.sac_mode = sac_mode

        # holders
        self.ori_img_coors = [] # point matches in the original (entire) images
        self.all_input_matched_area0s = None
        self.all_input_matched_area1s = None
        
        if self.sampler_name == "":
            pass
        elif self.sampler_name == "GridFill":
            self.occ_size = occ_size
            self.common_occ_flag = common_occ_flag
        else:
            raise NotImplementedError

        self.initialized = False

    def name(self) -> str:
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
        self.PMatcher = point_matcher

    def load_ori_corrs(self, ori_corrs):
        """
        """
        self.ori_img_coors = ori_corrs

    def doubtful_area_match_predict(self, doubt_match_pairs):
        """
        """
        assert self.initialized, f"{self.name} not initialized"
        logger.error(f"{self.name} has no doubtful area match predicter")
        raise NotImplementedError

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

        # draw
        if self.draw_verbose == 1:
            self.draw_ori_img()
            test_dir_if_not_create(self.out_path)

    def geo_area_matching_refine(self, 
        matched_areas0, 
        matched_areas1):
        """ 
        Returns:
            alpha_corrs_dict: {
                alpha0: [[inside-area-corrs], ...],
                alpha1: [[inside-area-corrs], ...],
                ... NOTE: corrs in eval-from-size
            }
            alpha_inlier_idxs_dict: {
                alpha0: [inlier-area-idxs, ...], # index corresponding to input area list
                alpha1: [inlier-area-idxs, ...],
                ...
            }
            [Optional] alpha_sampled_corrs: { 
                alpha0: [sampled-corrs, ...],
                alpha1: [sampled-corrs, ...],
                ...
            }
        """
        assert self.initialized, f"{self.name} not initialized, run .init_gam first"

        if self.sampler_name == "":
            alpha_corrs_dict, alpha_inlier_idxs_dict = self.geo_area_match_rejection(matched_areas0, matched_areas1)
            return alpha_corrs_dict, alpha_inlier_idxs_dict, None

        elif self.sampler_name == "GridFill":
            alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs = self.geo_area_match_rejection_sampler(matched_areas0, matched_areas1)
            return alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs
        
        else:
            raise NotImplementedError
    

    """=============Rejection Part=================="""

    # Main Function without AC
    def geo_area_match_rejection_sampler(self, matched_area0s, matched_area1s):
        """ Perform GAM
        Flow:
            1. for each area matches:
                - match points - put into eval-from-size
                - normalize points with K
                - estimate E using MAGSAC++
                - get mask to filter inlier points
                - recover inlier points to original images
                - recover F and calculate Sampson Distance
            2. calc all-areas-mean Sampson Distance and derive thd
            3. calc geometry consistency of every area matches
        Returns:
            alpha-corr-dict = {
                alpha0: [[inlier-area0-corrs], [inlier-area1-corrs], ...],
                alpha1: [[inlier-area0-corrs], [inlier-area1-corrs], ...],
                ... NOTE: corrs are masked inliers and in eval-from-size
            }
            alpha-inlier-idxs = {
                alpha0: [inlier-area0-idxs, inlier-area1-idxs, ...],
                alpha1: [inlier-area0-idxs, inlier-area1-idxs, ...],
                ...
            }
            if no area matches, return original matched points
        """
        logger.info(f"Start GAM with Sampler Rejection")
        assert self.sampler_name != "", f"sampler_name is empty"

        # holder
        alpha_corrs_dict = defaultdict(list)
        alpha_inlier_idxs_dict = defaultdict(list)
        alpha_sampled_corrs_dict = defaultdict(list)
        
        # if no area matches, return directly the original matched points
        if len(matched_area0s) == 0 or len(matched_area1s) == 0:
            logger.info(f"No area matches, return directly")
            
            if len(self.ori_img_coors) == 0:
                logger.error(f"No matched points in the original images")
                return None, None, None

            # fill the original matched points
            for alpha in self.alpha_list:
                alpha_corrs_dict[alpha] = [self.ori_img_coors]
                alpha_inlier_idxs_dict[alpha] = []

            return alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_corrs_dict

        # load all area matches: put into crop-from-size
        self.tune_all_area_matches_to_crop_from_size(matched_area0s, matched_area1s)

        # draw areas before GAM 
        if self.draw_verbose == 1:
            self.draw_match_areas(self.all_input_matched_area0s, self.all_input_matched_area1s, "before_GAM")

        # for each area matches, calc inside point matches and Sampson Distance
        inlier_mkpts_list = [] # list of corrs
        F_list = []
        self_sampson_dist_list = []
        for idx, (area0, area1) in enumerate(zip(self.all_input_matched_area0s, self.all_input_matched_area1s)):
            logger.info(f"Start SD calc for area {idx}")
            temp_inlier_corrs, temp_F, temp_self_sampson_dist = self.inside_area_match_calculation(area0, area1, idx)
            
            if temp_inlier_corrs is None:
                inlier_mkpts_list.append(None)
                F_list.append(None)
                self_sampson_dist_list.append(None)
                continue
            else:
                inlier_mkpts_list.append(temp_inlier_corrs)
                F_list.append(temp_F)
                self_sampson_dist_list.append(temp_self_sampson_dist)

        # derive thd by take the mean of all areas Sampson Distance
        # mind the None
        self_sampson_dist_list_unorder = [dist for dist in self_sampson_dist_list if dist is not None]
        if len(self_sampson_dist_list_unorder) == 0:
            logger.error(f"No valid Sampson Distance, return directly")
            return None, None, None
        
        mean_sd = np.mean(self_sampson_dist_list_unorder)

        # derive thd
        gam_thd_dict = {}
        for alpha in self.alpha_list:
            gam_thd_dict[alpha] = mean_sd * alpha
        
        # for each area matches, calc geometry consistency
        geo_cons_list = []
        for idx, temp_F in enumerate(F_list):
            if temp_F is None:
                geo_cons_list.append(None)
                continue

            # calc geometry consistency
            geo_cons = self.calc_geo_consistency(temp_F, inlier_mkpts_list)
            geo_cons_list.append(geo_cons)

        # for each alpha, filter inliers
        for alpha in self.alpha_list:
            logger.info(f"Start filtering inliers for alpha {alpha}")
            inlier_idxs = []
            for idx, geo_cons in enumerate(geo_cons_list):
                if geo_cons is None:
                    continue
                if geo_cons <= gam_thd_dict[alpha]:
                    inlier_idxs.append(idx)

            alpha_inlier_idxs_dict[alpha] = inlier_idxs
            # fill inlier corrs
            alpha_corrs_dict[alpha] = [inlier_mkpts_list[idx] for idx in inlier_idxs]

        # draw areas after GAM
        if self.draw_verbose == 1:
            for alpha in self.alpha_list:
                temp_area0s = [self.all_input_matched_area0s[idx] for idx in alpha_inlier_idxs_dict[alpha]]
                temp_area1s = [self.all_input_matched_area1s[idx] for idx in alpha_inlier_idxs_dict[alpha]]
                self.draw_match_areas(temp_area0s, temp_area1s, f"after_GAM_alpha_{alpha}")

        # till now, we have got the inlier corrs for each alpha
        # next, we focus on how to sample the inliers to achieve spatially even distribution
        # we use sampler to achieve this

        for alpha in self.alpha_list:
            sampler = self._load_sampler()
            sampler.load_ori_imgs(self.color0, self.color1)
            sampler.name = f"{alpha}"
            
            cur_corrs_list = copy.deepcopy(alpha_corrs_dict[alpha])
            cur_corrs_list.append(self.ori_img_coors)
            sampler.load_corrs_from_GAM(cur_corrs_list)

            sampled_corrs, sampled_corrs_rand = sampler.sample()
            if sampled_corrs is None:
                alpha_sampled_corrs_dict[alpha] = self.ori_img_coors
            else:
                alpha_sampled_corrs_dict[alpha] = sampled_corrs_rand
        
        return alpha_corrs_dict, alpha_inlier_idxs_dict, alpha_sampled_corrs_dict


    # Main Function with Adaptive Collection and no sampler
    def geo_area_match_rejection(self, matched_area0s, matched_area1s):
        """ Perform GAM
        Flow:
            1. for each area matches:
                - match points - put into eval-from-size
                - normalize points with K
                - estimate E using MAGSAC++
                - get mask to filter inlier points
                - recover inlier points to original images
                - recover F and calculate Sampson Distance
            2. calc all-areas-mean Sampson Distance and derive thd
            3. calc geometry consistency of every area matches
        Returns:
            alpha-corr-dict = {
                alpha0: [[inlier-area0-corrs], [inlier-area1-corrs], ...],
                alpha1: [[inlier-area0-corrs], [inlier-area1-corrs], ...],
                ... NOTE: corrs are masked inliers and in eval-from-size
            }
            alpha-inlier-idxs = {
                alpha0: [inlier-area0-idxs, inlier-area1-idxs, ...],
                alpha1: [inlier-area0-idxs, inlier-area1-idxs, ...],
                ...
            }
            if no area matches, return original matched points
        """
        logger.info(f"Start GAM Rejection")

        # holder
        alpha_corrs_dict = defaultdict(list)
        alpha_inlier_idxs_dict = defaultdict(list)
        
        # if no area matches, return directly the original matched points
        if len(matched_area0s) == 0 or len(matched_area1s) == 0:
            logger.info(f"No area matches, return directly")
            
            if len(self.ori_img_coors) == 0:
                logger.error(f"No matched points in the original images")
                # initialize as empty
                for alpha in self.alpha_list:
                    alpha_corrs_dict[alpha] = []
                    alpha_inlier_idxs_dict[alpha] = []
                return alpha_corrs_dict, alpha_inlier_idxs_dict

            # fill the original matched points
            for alpha in self.alpha_list:
                alpha_corrs_dict[alpha] = [self.ori_img_coors]
                alpha_inlier_idxs_dict[alpha] = []

            return alpha_corrs_dict, alpha_inlier_idxs_dict

        # load all area matches: put into crop-from-size
        self.tune_all_area_matches_to_crop_from_size(matched_area0s, matched_area1s)

        # draw areas before GAM 
        if self.draw_verbose == 1:
            self.draw_match_areas(self.all_input_matched_area0s, self.all_input_matched_area1s, "before_GAM")

        # for each area matches, calc inside point matches and Sampson Distance
        inlier_mkpts_list = [] # list of corrs
        F_list = []
        self_sampson_dist_list = []
        for idx, (area0, area1) in enumerate(zip(self.all_input_matched_area0s, self.all_input_matched_area1s)):
            logger.info(f"Start SD calc for area {idx}")
            temp_inlier_corrs, temp_F, temp_self_sampson_dist = self.inside_area_match_calculation(area0, area1, idx)
            
            if temp_inlier_corrs is None:
                inlier_mkpts_list.append(None)
                F_list.append(None)
                self_sampson_dist_list.append(None)
                continue
            else:
                inlier_mkpts_list.append(temp_inlier_corrs)
                F_list.append(temp_F)
                self_sampson_dist_list.append(temp_self_sampson_dist)

        # derive thd by take the mean of all areas Sampson Distance
        # mind the None
        self_sampson_dist_list_unorder = [dist for dist in self_sampson_dist_list if dist is not None]
        if len(self_sampson_dist_list_unorder) == 0:
            logger.error(f"No valid Sampson Distance, return directly")
            for alpha in self.alpha_list:
                alpha_corrs_dict[alpha] = [self.ori_img_coors]
                alpha_inlier_idxs_dict[alpha] = []
            return alpha_corrs_dict, alpha_inlier_idxs_dict
        
        mean_sd = np.mean(self_sampson_dist_list_unorder)

        # derive thd
        gam_thd_dict = {}
        for alpha in self.alpha_list:
            gam_thd_dict[alpha] = mean_sd * alpha
        
        # for each area matches, calc geometry consistency
        geo_cons_list = []
        for idx, temp_F in enumerate(F_list):
            if temp_F is None:
                geo_cons_list.append(None)
                continue

            # calc geometry consistency
            geo_cons = self.calc_geo_consistency(temp_F, inlier_mkpts_list)
            geo_cons_list.append(geo_cons)

        # for each alpha, filter inliers
        for alpha in self.alpha_list:
            logger.info(f"Start filtering inliers for alpha {alpha}")
            inlier_idxs = []
            for idx, geo_cons in enumerate(geo_cons_list):
                if geo_cons is None:
                    continue
                if geo_cons <= gam_thd_dict[alpha]:
                    inlier_idxs.append(idx)

            alpha_inlier_idxs_dict[alpha] = inlier_idxs
            # fill inlier corrs
            alpha_corrs_dict[alpha] = [inlier_mkpts_list[idx] for idx in inlier_idxs]

        # draw areas after GAM
        if self.draw_verbose == 1:
            for alpha in self.alpha_list:
                temp_area0s = [self.all_input_matched_area0s[idx] for idx in alpha_inlier_idxs_dict[alpha]]
                temp_area1s = [self.all_input_matched_area1s[idx] for idx in alpha_inlier_idxs_dict[alpha]]
                self.draw_match_areas(temp_area0s, temp_area1s, f"after_GAM_alpha_{alpha}")
        
        # Adaptive Collection
        for alpha in self.alpha_list:
            temp_alpha_area0s = [self.all_input_matched_area0s[idx] for idx in alpha_inlier_idxs_dict[alpha]]
            temp_alpha_area1s = [self.all_input_matched_area1s[idx] for idx in alpha_inlier_idxs_dict[alpha]]

            # 0. calc total area cover size
            if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
                size_ratio0 = self.calc_size_cover_ratio(temp_alpha_area0s, self.crop_from_size_W0, self.crop_from_size_H0)
                size_ratio1 = self.calc_size_cover_ratio(temp_alpha_area1s, self.crop_from_size_W1, self.crop_from_size_H1)
            elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
                size_ratio0 = self.calc_size_cover_ratio(temp_alpha_area0s, self.crop_from_size_W, self.crop_from_size_H)
                size_ratio1 = self.calc_size_cover_ratio(temp_alpha_area1s, self.crop_from_size_W, self.crop_from_size_H)
            else:
                raise NotImplementedError
            
            # 1. if the total area size is samller than the thd, perform adaptive collection
            if size_ratio0 <= self.adaptive_size_thd or size_ratio1 <= self.adaptive_size_thd:
                logger.info(f"Start adaptive collection for alpha {alpha}")

                # 1.1. calc F and mean SD for all corr inliers
                temp_alpha_corrs = list_of_corrs2corr_list(alpha_corrs_dict[alpha])
                temp_alpha_F, temp_alpha_mean_sd = cal_corr_F_and_mean_sd(temp_alpha_corrs)

                # 1.2 perform collection on original image corrs
                temp_ori_corrs = self.ori_img_coors
                if len(temp_ori_corrs) == 0:
                    logger.error(f"No original image corrs")
                    continue
                
                filtered_ori_corrs = filter_corrs_by_F(temp_ori_corrs, temp_alpha_F, temp_alpha_mean_sd)
                
                # logger.success(f"achieve {len(filtered_ori_corrs)} corrs after adaptive collection for alpha {alpha}")

                # append to the alpha corrs
                if len(filtered_ori_corrs) > 0:
                    alpha_corrs_dict[alpha].append(filtered_ori_corrs)

        return alpha_corrs_dict, alpha_inlier_idxs_dict


    """=============Rejection Utils Part=================="""

    def _load_sampler(self):
        """
        """
        assert self.sampler_name != "", f"sampler_name is empty"

        if self.sampler_name == "GridFill":
            from .MatchSampler import GridFillSampler

            sampler_cfg = {
                "W0": self.eval_from_size_W0,
                "H0": self.eval_from_size_H0,
                "W1": self.eval_from_size_W1,
                "H1": self.eval_from_size_H1,
                "out_path": self.out_path,
                "sample_num": self.std_match_num,
                "draw_verbose": self.draw_verbose,
                "occ_size": self.occ_size,
                "common_occ_flag": self.common_occ_flag,
            }

            sampler = GridFillSampler(sampler_cfg)
            
        else:
            raise NotImplementedError
        
        return sampler

    def calc_size_cover_ratio(self, areas, ori_W, ori_H):
        """
        Args:
            areas: [[u_min, u_max, v_min, v_max]s]
        Returns:
            size_cover_ratio
        """
        ori_size = ori_H * ori_W
        merge_size = calc_merge_size_of_areas(areas, ori_W, ori_H)
        size_ratio = merge_size / ori_size
        return size_ratio

    def calc_geo_consistency(self, F, inlier_corrs_list):
        """ calc geometry consistency
        Funs:
            0. remove None
            1. turn list of corrs to corr list
            2. calc Sampson Distance
        """
        # remove None
        non_None_corrs_list = [corrs for corrs in inlier_corrs_list if corrs is not None]

        corrs = list_of_corrs2corr_list(non_None_corrs_list)
        geo_cons = calc_sampson_dist(F, corrs)
        
        return geo_cons

    def inside_area_match_calculation(self, area0, area1, idx):
        """ Calc point matches, Sampson Distance, and inlier mask for area matches
        Args:
            area0, area1: [u_min, u_max, v_min, v_max]
        Returns:
            inlier_corrs: in eval-from-size, if None, means no valid matches
            F: Fundamental Matrix
            self_sampson_dist: self Sampson Distance
        """

        # crop areas from original images and match, NOTE: output in eval-from-size
        if self.crop_mode == 0:
            temp_corrs = self.match_area_pair_mind_size(area0, area1, f"rejecting_{idx}") # corrs in eval_from_size
        elif self.crop_mode == 1:
            temp_corrs = self.match_area_pair_padding(area0, area1, f"rejecting_{idx}")
        elif self.crop_mode == 2:
            temp_corrs = self.match_area_pair_padding_expand_square(area0, area1, f"rejecting_{idx}")
        elif self.crop_mode == 3:
            temp_corrs = self.match_area_pair_mask(area0, area1, f"rejecting_{idx}")
        else:
            raise NotImplementedError

        if len(temp_corrs) <= self.valid_inside_area_match_num:
            logger.warning(f"{len(temp_corrs)} Less than {self.valid_inside_area_match_num} matches for area {idx}, skip")
            return None, None, None

        # logger.success(f"achieve {len(temp_corrs)} corrs for area {idx}")

        if self.reject_out_area_flag:
            temp_corrs = self.reject_corrs_outside_area(temp_corrs, area0, area1)
            if len(temp_corrs) <= self.valid_inside_area_match_num:
                logger.warning(f"{len(temp_corrs)} Less than {self.valid_inside_area_match_num} inliers for area {idx}, skip")
                return None, None, None

        # calc E
        E, inlier_corrs = calc_E_from_corrs(temp_corrs, self.K0, self.K1, sac_mode=self.sac_mode)
        F = recover_F_from_E_K(E, self.K0, self.K1)

        if len(inlier_corrs) <= self.valid_inside_area_match_num: # set as param
            logger.warning(f"{len(inlier_corrs)} Less than {self.valid_inside_area_match_num} inliers for area {idx}, skip")
            return None, None, None

        # logger.success(f"achieve {len(inlier_corrs)} inliers for area {idx}")

        # calc self Sampson Distance
        inlier_corrs_np = np.array(inlier_corrs)
        self_sampson_dist = calc_sampson_dist(F, inlier_corrs_np)

        return inlier_corrs, F, self_sampson_dist
    
    def reject_corrs_outside_area(self, corrs, area0, area1):
        """
        Args:
            corrs in eval_from_size
            areas in eval_from_size
        """
        refined_corrs = []
        for corr in corrs:
            if is_corr_inside_area(corr, area0, area1):
                refined_corrs.append(corr)
            else:
                # logger.warning(f"corr {corr} is outside area {area0} and {area1}")
                pass
        
        return refined_corrs

    def match_area_pair_padding_expand_square(self, area0, area1, name=""):
        """use padding to match, but expand area to square first
        """
        rt_corrs = []

        crop0, scale0, offset0 = img_crop_with_padding_expand_square(self.ori_img0, area0, self.crop_size_W, self.crop_size_H)
        crop1, scale1, offset1 = img_crop_with_padding_expand_square(self.ori_img1, area1, self.crop_size_W, self.crop_size_H)

        corr_num = int(self.std_match_num)
        self.PMatcher.set_corr_num_init(corr_num)
        try:
            self.PMatcher.match(crop0, crop1)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []

        temp_corrs = self.PMatcher.return_matches()
        temp_corrs = np.array(temp_corrs)

        # draw the matches on area matches
        if self.draw_verbose == 1:
            # TODO:
            plot_matches_lists_lr(crop0, crop1, temp_corrs, self.out_path, name=f"area_point_{name}")

        if temp_corrs.shape[0] == 0: return []

        # recover to crop_from_size
        scale_zip = [scale0[0], scale0[1], scale1[0], scale1[1]]
        offset_zip = [offset0[0], offset0[1], offset1[0], offset1[1]] # [u0, v0, u1, v1]

        rt_corrs = recover_corrs_offset_scales(temp_corrs, offset_zip, scale_zip) # corrs are in the crop_from_size

        # recover to eval size
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
            rt_corrs = tune_corrs_size_diff(rt_corrs, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1, self.eval_from_size_W0, self.eval_from_size_W1, self.eval_from_size_H0, self.eval_from_size_H1)
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            rt_corrs = tune_corrs_size(rt_corrs, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)
        else:
            raise NotImplementedError

        return rt_corrs

    def match_area_pair_padding(self, area0, area1, name=""):
        """use padding to match
        Args:
            area0/1: [u_min, u_max, v_min, v_max] in crop_from_size
        Returns:
            rt_corrs: [[u0, v0, u1, v1]s] in eval_from_size
        """
        rt_corrs = []

        crop0, scale0, offset0 = img_crop_with_padding_improve_resolution(self.ori_img0, area0, self.crop_size_W, self.crop_size_H)
        crop1, scale1, offset1 = img_crop_with_padding_improve_resolution(self.ori_img1, area1, self.crop_size_W, self.crop_size_H)

        corr_num = int(self.std_match_num)
        self.PMatcher.set_corr_num_init(corr_num)
        try:
            self.PMatcher.match(crop0, crop1)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []

        temp_corrs = self.PMatcher.return_matches()
        temp_corrs = np.array(temp_corrs)

        if temp_corrs.shape[0] == 0: return []

        # recover to crop_from_size
        scale_zip = [scale0[0], scale0[1], scale1[0], scale1[1]]
        offset_zip = [offset0[0], offset0[1], offset1[0], offset1[1]]

        rt_corrs = recover_corrs_offset_scales(temp_corrs, offset_zip, scale_zip) # corrs are in the crop_from_size

        # recover to eval size
        rt_corrs = tune_corrs_size(rt_corrs, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)

        return rt_corrs

    def match_area_pair_mind_size(self, area0, area1, name=""):
        """ test-Done
        Func: 
            0. tune area (self.area_from_size) size to self.crop_from_size # NOTE Has been done in area load part
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

        corr_num = int(self.std_match_num)
        self.PMatcher.set_corr_num_init(corr_num)
        try:
            self.PMatcher.match(crop0, crop1)
        except AssertionError:
            logger.exception("Some Error Happened during match")
            return []

        temp_corrs = self.PMatcher.return_matches()
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
        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
        # if self.datasetName == "MegaDepth":
            rt_corrs = tune_corrs_size_diff(rt_corrs, self.crop_from_size_W0, self.crop_from_size_W1, self.crop_from_size_H0, self.crop_from_size_H1, self.eval_from_size_W0, self.eval_from_size_W1, self.eval_from_size_H0, self.eval_from_size_H1)
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            rt_corrs = tune_corrs_size(rt_corrs, self.crop_from_size_W, self.crop_from_size_H, self.eval_from_size_W, self.eval_from_size_H)
        else:
            raise NotImplementedError

        logger.info(f"achieve {np.array(rt_corrs).shape} corrs for {name}")

        return rt_corrs

    def tune_all_area_matches_to_crop_from_size(self, matched_area0s, matched_area1s):
        """ Load all area matches to crop-from-size
        Args:
            self.all_area0_matches, self.all_area1_matches in the crop-from-size
        """
        assert matched_area0s is not None, f"all_area0_matches is None"
        assert matched_area1s is not None, f"all_area1_matches is None"
        assert len(matched_area0s) == len(matched_area1s), f"len(all_area0_matches) != len(all_area1_matches)"

        if self.datasetName == "MegaDepth" or self.datasetName == "YFCC" or self.datasetName == "ETH3D":
            self.all_input_matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W0, self.crop_from_size_H0)
            self.all_input_matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W1, self.crop_from_size_H1)
        elif self.datasetName == "ScanNet" or self.datasetName == "KITTI" or self.datasetName == "MatterPort3D":
            self.all_input_matched_area0s = self.tune_area_list_size(matched_area0s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
            self.all_input_matched_area1s = self.tune_area_list_size(matched_area1s, self.area_from_size_W, self.area_from_size_H, self.crop_from_size_W, self.crop_from_size_H)
        else:
            raise NotImplementedError
        
        logger.info(f"Load all area matches to crop-from-size")

        return self.all_input_matched_area0s, self.all_input_matched_area1s


    """=============Utils Part=================="""

    def draw_match_areas(self, matched_area0s, matched_area1s, name=""):
        """ areas in crop from size
        """
        draw_matched_area_list(self.ori_img0, self.ori_img1, matched_area0s, matched_area1s, self.out_path, name, "")

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

    def load_ori_img_corrs_in_eval_size(self, corrs):
        """
        """

    def draw_ori_img(self):
        cv2.imwrite(os.path.join(self.out_path, "Geo_ori0.jpg"), self.ori_img0)
        cv2.imwrite(os.path.join(self.out_path, "Geo_ori1.jpg"), self.ori_img1)

