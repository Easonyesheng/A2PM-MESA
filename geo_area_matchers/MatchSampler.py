
import sys
sys.path.append('../')

import os
import numpy as np
import cv2
import random
from loguru import logger

from utils.geo import (
    cal_corr_F_and_mean_sd_rt_sd,
    list_of_corrs2corr_list,
)

from utils.vis import (
    plot_matches_lists_lr,
)

class BasicMatchSampler(object):
    """Basic Sampler for the MatchSampler
    Basic Flow:

    """
    dft_configs = {
        "W0": 640,
        "H0": 480,
        "W1": 640,
        "H1": 480,
        "out_path": "",
        "sample_num": 1000,
        "draw_verbose": 0,
    }

    def __init__(self, configs) -> None:
        """
        """
        self.configs = {**self.dft_configs, **configs}
        self.W0 = self.configs["W0"]
        self.H0 = self.configs["H0"]
        self.W1 = self.configs["W1"]
        self.H1 = self.configs["H1"]
        self.sample_num = self.configs["sample_num"]
        
        self.total_corrs_list = None
        self.inside_area_corrs = None
        self.global_corrs = None
        self.sampled_corrs = None
        self.sampled_corrs_rand = None
        self.name = ""

        self.img0 = None
        self.img1 = None

        self.out_path = self.configs["out_path"]
        self.draw_verbose = self.configs["draw_verbose"]

    def load_corrs_from_GAM(self, corrs_list):
        """ Load the correspondences from the GAM
        Args:
            corrs_list (list of list): list of correspondences
                the last one is the global correspondences
        """
        self.total_corrs_list = corrs_list
        self.inside_area_corrs = corrs_list[:-1]
        self.global_corrs = corrs_list[-1]

    def load_ori_imgs(self, img0, img1):
        """ Load the original images in eval from size
        Args:
            img0 (np.ndarray): original image 0
            img1 (np.ndarray): original image 1
        """

        # if gray image, convert to color image
        if len(img0.shape) == 2:
            self.img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        else:
            self.img0 = img0

        if len(img1.shape) == 2:
            self.img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            self.img1 = img1

    def draw_before_sample(self):
        """
        """
        assert self.img0 is not None, "Please load the original images first"
        assert self.img1 is not None, "Please load the original images first"
        assert self.total_corrs_list is not None, "Please load the correspondences first"

        if self.draw_verbose:
            temp_all_corrs = list_of_corrs2corr_list(self.total_corrs_list)
            plot_matches_lists_lr(self.img0, self.img1, temp_all_corrs, self.out_path, name=f"{self.name}_before_sample_all")
            
            # random sample
            if len(temp_all_corrs) <= self.sample_num:
                temp_all_corrs_meet_num = temp_all_corrs
            else:
                temp_all_corrs_meet_num = random.sample(temp_all_corrs, self.sample_num)
            plot_matches_lists_lr(self.img0, self.img1, temp_all_corrs_meet_num, self.out_path, name=f"{self.name}_before_sample_{self.sample_num}")

    def draw_after_sample(self, sampled_corrs):
        assert self.img0 is not None, "Please load the original images first"
        assert self.img1 is not None, "Please load the original images first"
        assert self.sampled_corrs is not None, "Please sample the correspondences first"

        if self.draw_verbose:
            plot_matches_lists_lr(self.img0, self.img1, self.sampled_corrs, self.out_path, name=f"{self.name}_after_sample_all")
            plot_matches_lists_lr(self.img0, self.img1, self.sampled_corrs_rand, self.out_path, name=f"{self.name}_after_sample_{self.sample_num}_rand")

    def sample(self):
        """ Sample the correspondences
        Args:
            num_samples (int): number of samples
        """
        raise NotImplementedError



class GridFillSampler(BasicMatchSampler):
    """
    """

    def __init__(self, configs) -> None:
        super().__init__(configs)

        # specific params
        self.occ_size = self.configs["occ_size"]
        self.common_occ_flag = self.configs["common_occ_flag"]

        self.occ_img0 = np.zeros((self.H0, self.W0), dtype=np.uint8)
        self.occ_img1 = np.zeros((self.H1, self.W1), dtype=np.uint8)

    def sample(self):
        """
        Returns:
            sampled_corrs (list of corr): sampled correspondences
        """
        assert self.inside_area_corrs is not None, "Please load the correspondences first"
        assert self.global_corrs is not None, "Please load the correspondences first"

        num_samples = self.sample_num
        
        self.draw_before_sample()

        # calc F and Sampson Distance for every correspondence inside the area
        temp_inside_corrs = list_of_corrs2corr_list(self.inside_area_corrs)
        if len(temp_inside_corrs) <= 10:
            logger.error(f"Too few correspondences inside the area, only {len(temp_inside_corrs)}")
            return None, None

        F, mean_sd, rt_sd_list = cal_corr_F_and_mean_sd_rt_sd(temp_inside_corrs)

        # sort the correspondences by Sampson Distance (smaller is at the front)
        rt_sd_list = np.array(rt_sd_list)
        sorted_idx = np.argsort(rt_sd_list)
        sorted_inside_area_corrs = [temp_inside_corrs[idx] for idx in sorted_idx]

        # fill the occ img with inside area correspondences
        sampled_corrs_inside = self.fill_occ_img(sorted_inside_area_corrs)
        logger.info(f"Sampled {len(sampled_corrs_inside)} correspondences inside the area")

        # fill the occ img with global correspondences
        sampled_corrs_global = self.fill_occ_img(self.global_corrs)
        logger.info(f"Sampled {len(sampled_corrs_global)} correspondences outside the area")

        # fuse
        sampled_corrs = sampled_corrs_inside + sampled_corrs_global
        logger.info(f"Sampled {len(sampled_corrs)} correspondences in total")

        self.sampled_corrs = sampled_corrs

        # random sample self.sample_num correspondences
        if len(sampled_corrs) <= num_samples:
            sampled_corrs_rand = sampled_corrs
        else:
            sampled_corrs_rand = random.sample(sampled_corrs, num_samples)
        
        self.sampled_corrs_rand = sampled_corrs_rand

        self.draw_after_sample(sampled_corrs)

        return sampled_corrs, sampled_corrs_rand

    def fill_occ_img(self, sorted_corrs):
        """
        Args:
            sorted_corrs (list of corr): sorted correspondences
        Returns:
            sampled_corrs (list of corr): sampled correspondences
        """
        sorted_corrs_np = np.array(sorted_corrs)
        sampled_corrs = []

        for corr in sorted_corrs_np:
            corr_int = corr.astype(np.int64)
            u0, v0 = corr_int[:2]
            u1, v1 = corr_int[2:]
            
            if u0 < self.occ_size or u0 >= self.W0-self.occ_size or v0 < self.occ_size or v0 >= self.H0-self.occ_size:
                continue

            if u1 < self.occ_size or u1 >= self.W1-self.occ_size or v1 < self.occ_size or v1 >= self.H1-self.occ_size:
                continue

            if self.common_occ_flag:
                try:
                    if self.occ_img0[v0, u0] == 0 and self.occ_img1[v1, u1] == 0:
                        self.occ_img0[v0-self.occ_size:v0+self.occ_size, u0-self.occ_size:u0+self.occ_size] = 1
                        self.occ_img1[v1-self.occ_size:v1+self.occ_size, u1-self.occ_size:u1+self.occ_size] = 1
                        sampled_corrs.append(corr)
                    else:
                        continue
                except IndexError as e:
                    logger.error(f"IndexError: {corr}, W0 {self.W0}, H0 {self.H0}, W1 {self.W1}, H1 {self.H1}")
            else:
                if self.occ_img0[v0, u0] == 0 or self.occ_img1[v1, u1] == 0:
                    self.occ_img0[v0-self.occ_size:v0+self.occ_size, u0-self.occ_size:u0+self.occ_size] = 1
                    self.occ_img1[v1-self.occ_size:v1+self.occ_size, u1-self.occ_size:u1+self.occ_size] = 1
                    sampled_corrs.append(corr)
                else:
                    continue
        
        return sampled_corrs
        
            





        