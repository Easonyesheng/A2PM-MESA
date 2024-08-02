'''
Author: EasonZhang
Date: 2024-06-19 21:05:36
LastEditors: EasonZhang
LastEditTime: 2024-07-18 17:06:50
FilePath: /SA2M/hydra-mesa/metric/instance_eval.py
Description: the evaluator for instance-level metrics, including
    - area matching metrics TODO:
        - area overlap ratio (AOR)
    - point matching metrics TODO:
        - MMA w/ depth
        - MMA w/o depth
    - pose estimation metrics
        - pose error

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import os
import numpy as np
from loguru import logger
import random
from typing import List, Optional, Any, Tuple

from utils.geo import (
    tune_corrs_size_diff,
    nms_for_corrs,
    compute_pose_error_simp,
    calc_area_match_performence_eff_MC,
    assert_match_reproj,
    assert_match_qFp,
)

from utils.vis import plot_matches_with_mask_ud

from utils.common import test_dir_if_not_create

class InstanceEval(object):
    """ params are loaded when using
    """

    def __init__(self,
        sample_mode,
        eval_corr_num,
        sac_mode,
        out_path,
        draw_verbose=False,
        ) -> None:
        self.eval_info = None
        self.sample_mode = sample_mode
        self.eval_corr_num = eval_corr_num
        self.sac_mode = sac_mode
        self.out_path = os.path.join(out_path, 'ratios')
        self.draw_verbose = draw_verbose
        test_dir_if_not_create(self.out_path)

    def init_data_loader(self, dataloader, eval_W, eval_H):
        """
        """
        try:
            self.eval_info = dataloader.get_eval_info(eval_W, eval_H)
            self.instance_name = f'{dataloader.scene_name}_{dataloader.image_name0}_{dataloader.image_name1}'
        except Exception as e:
            logger.error(f'Error: {e}')
    
    def eval_area_overlap_ratio(self,
        areas0,
        areas1,
        pre_name,
        ):
        """
        Args:
            areas0/1: list of [u_min, u_max, v_min, v_max] 
        """
        assert self.eval_info is not None, 'Error: eval_info is not loaded, init dataloader pls'
        assert self.eval_info['dataset_name'] in ['ScanNet'], 'Error: dataset not supported'

        if len(areas0) < 1 or len(areas1) < 1:
            logger.error(f'Error: areas0/1 is empty')
            return

        try:
            acr, aor = calc_area_match_performence_eff_MC(
                    areas0,
                    areas1,
                    self.eval_info['image0'],
                    self.eval_info['image1'],
                    self.eval_info['K0'],
                    self.eval_info['K1'],
                    self.eval_info['P0'],
                    self.eval_info['P1'],
                    self.eval_info['depth0'],
                    self.eval_info['depth1'],
                    self.eval_info['depth_factor'],
                    )
            mean_aor = np.mean(aor)
        except Exception as e:
            logger.error(f'Error: {e}')
            raise e
        
        # write to file
        name_file = os.path.join(self.out_path, f'{pre_name}_ameval_names.txt')
        aor_file = os.path.join(self.out_path, f'{pre_name}_aor.txt')
        acr_file = os.path.join(self.out_path, f'{pre_name}_acr.txt')

        # read names
        if not os.path.exists(name_file):
            exist_names = []
        else:
            with open(name_file, 'r') as f:
                exist_names = f.readlines()
                exist_names = [name.strip() for name in exist_names]
        
        if self.instance_name in exist_names:
            pass
        else:
            with open(name_file, 'a') as f:
                f.write(f'{self.instance_name}\n')
            with open(aor_file, 'a') as f:
                f.write(f'{mean_aor}\n')
            with open(acr_file, 'a') as f:
                f.write(f'{acr}\n')

    def eval_point_match(self,
        corrs,
        pre_name,
        thds=[1,3,5],
        ):
        """
        Args:
            corrs: all corrs should be in the eval size
        """
        assert self.eval_info is not None, 'Error: eval_info is not loaded, init dataloader pls'
        assert self.sample_mode in ['random', 'grid'], f'Error: sample_mode {sample_mode} not supported'

        eval_num = self.eval_corr_num
        sample_mode = self.sample_mode

        corr_num = len(corrs)
        if corr_num < 10:
            logger.error(f'Error: not enough corrs for pose estimation')
            return []

        if corr_num > eval_num:
            if sample_mode == 'random':
                corrs = random.sample(corrs, eval_num)
            elif sample_mode == 'grid':
                corrs = np.array(corrs)
                corrs = nms_for_corrs(corrs, r=3)
                if len(corrs) > eval_num:
                    corrs = random.sample(corrs, eval_num)

        good_ratios = []
        masks = []

        try:
            dataset_name = self.eval_info['dataset_name']
            pose0 = self.eval_info['P0']
            pose1 = self.eval_info['P1']
            K0 = self.eval_info['K0']
            K1 = self.eval_info['K1']
            image0 = self.eval_info['image0'] # imgs are in eval size
            image1 = self.eval_info['image1']
            if dataset_name == "ScanNet":
                depth0 = self.eval_info['depth0']
                depth1 = self.eval_info['depth1']
                depth_factor = self.eval_info['depth_factor']
                for thd in thds:
                    mask, bad_ratio, gt_pts = assert_match_reproj(corrs,
                                        depth0, depth1, depth_factor,
                                        K0, K1,
                                        pose0, pose1,
                                        thd, 0)
                    good_ratio = (100 - bad_ratio)/100
                    good_ratios.append(good_ratio)
                    masks.append(mask)

            elif dataset_name in ['MegaDepth', 'YFCC', 'ETH3D']:
                for thd in thds:
                    mask, bad_ratio = assert_match_qFp(corrs, K0, K1, pose0, pose1, thd)
                    good_ratio = (1 - bad_ratio)
                    good_ratios.append(good_ratio)
                    masks.append(mask)
            else:
                raise NotImplementedError(f"dataset {dataset_name} not supported")

            # draw match images with masks for corrs
            if self.draw_verbose:
                # get the upper path of the out_path
                up_out_path = os.path.dirname(self.out_path)
                pm_out_path = os.path.join(up_out_path, self.instance_name, 'pm')
                test_dir_if_not_create(pm_out_path)
                for i, thd in enumerate(thds):
                    plot_matches_with_mask_ud(
                        image0, image1,
                        masks[i], corrs, pm_out_path, pre_name+f"_mma_{thd}")

        except Exception as e:
            logger.error(f'Error: {e}')
            raise e
            
        
        logger.success(f'point matches good ratios: {good_ratios}')
        
        # write to file
        name_file = os.path.join(self.out_path, f'{pre_name}_mma_names.txt')
        mma_file = os.path.join(self.out_path, f'{pre_name}_mmas.txt')

        # read names
        if not os.path.exists(name_file):
            exist_names = []
        else:
            with open(name_file, 'r') as f:
                exist_names = f.readlines()
                exist_names = [name.strip() for name in exist_names]

        if self.instance_name in exist_names:
            pass
        else:
            with open(name_file, 'a') as f:
                f.write(f'{self.instance_name}\n')
            with open(mma_file, 'a') as f:
                for i, _ in enumerate(thds):
                    f.write(f'{good_ratios[i]} ')
                f.write('\n')

        return good_ratios


    def eval_pose_error(self, 
        corrs,
        pre_name,
        ):
        """
        Args:
            corrs: list in eval size
        Returns:
            pose error: [R_err, t_err]
        """
        assert self.eval_info is not None, 'Error: eval_info is not loaded, init dataloader pls'
        assert self.sample_mode in ['random', 'grid'], f'Error: sample_mode {sample_mode} not supported'

        corr_num = len(corrs)
        if corr_num < 10:
            logger.error(f'Error: not enough corrs for pose estimation')
            errs = [180, 180]

        eval_num = self.eval_corr_num
        sample_mode = self.sample_mode
        if corr_num > eval_num:
            if sample_mode == 'random':
                corrs = random.sample(corrs, eval_num)
            elif sample_mode == 'grid':
                corrs = np.array(corrs)
                corrs = nms_for_corrs(corrs, r=3)
                if len(corrs) > eval_num:
                    corrs = random.sample(corrs, eval_num)
        
        try:
            dataset_name = self.eval_info['dataset_name']
            pose0 = self.eval_info['P0']
            pose1 = self.eval_info['P1']
            K0 = self.eval_info['K0']
            K1 = self.eval_info['K1']
            if dataset_name == "MegaDepth":
                gt_pose = np.matmul(pose1, np.linalg.inv(pose0))
            else:
                gt_pose = pose1.I @ pose0

            errs = compute_pose_error_simp(corrs, K0, K1, gt_pose, pix_thd=0.5, conf=0.9999, sac_mode=self.sac_mode)

        except Exception as e:
            logger.error(f'Error: {e}')
            if e is KeyError:
                raise e
            else:
                errs = [180, 180]
        
        # write to file
        name_file = os.path.join(self.out_path, f'{pre_name}_pose_err_names.txt')
        pose_err_file = os.path.join(self.out_path, f'{pre_name}_pose_errs.txt')

        # read names
        if not os.path.exists(name_file):
            exist_names = []
        else:
            with open(name_file, 'r') as f:
                exist_names = f.readlines()
                exist_names = [name.strip() for name in exist_names]

        if self.instance_name in exist_names:
            pass
        else:
            with open(name_file, 'a') as f:
                f.write(f'{self.instance_name}\n')
            with open(pose_err_file, 'a') as f:
                f.write(f'{errs[0]} {errs[1]}\n')

        return errs


    @staticmethod
    def tune_corrs_size(corrs, 
        src_W0=None, 
        src_W1=None,
        src_H0=None,
        src_H1=None,
        dst_W0=None,
        dst_W1=None,
        dst_H0=None,
        dst_H1=None):
        """
        """
        return tune_corrs_size_diff(corrs, src_W0, src_W1, src_H0, src_H1, dst_W0, dst_W1, dst_H0, dst_H1)
        