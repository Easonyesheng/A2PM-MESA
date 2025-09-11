'''
Author: EasonZhang
Date: 2024-06-12 20:31:50
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-11 12:44:11
FilePath: /A2PM-MESA/scripts/test_a2pm.py
Description: test hydra-powered a2pm

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''
import sys
sys.path.append('..')

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from point_matchers.abstract_point_matcher import AbstractPointMatcher
from dataloader.abstract_dataloader import AbstractDataloader
from area_matchers.abstract_am import AbstractAreaMatcher
from geo_area_matchers.abstract_gam import AbstractGeoAreaMatcher
from metric.instance_eval import InstanceEval
from utils.common import validate_type
from utils.geo import list_of_corrs2corr_list

import random
# fix random seed
random.seed(2)

import torch
# fix random seed
torch.manual_seed(2)

@hydra.main(version_base=None, config_path="../conf")
def test(cfg: DictConfig) -> None:
    """
    Test A2PM
    """

    # set full error
    os.environ["HYDRA_FULL_ERROR"] = '1'
    
    OmegaConf.resolve(cfg)

    if cfg.verbose==0:
        logger.remove()
        logger.add(sys.stdout, level="SUCCESS")
    elif cfg.verbose==1:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    else:
        raise NotImplementedError(f"verbose {cfg.verbose} not supported")

    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # load point matcher
    pmer = hydra.utils.instantiate(cfg.point_matcher)
    validate_type(pmer, AbstractPointMatcher)

    # load dataloader
    dataloader = hydra.utils.instantiate(cfg.dataset)
    validate_type(dataloader, AbstractDataloader)

    # load evaler
    evaler = hydra.utils.instantiate(cfg.evaler)
    evaler.init_data_loader(dataloader, cfg.eval_from_size_W, cfg.eval_from_size_H)

    # test pmer
    if cfg.dataset_name in ['ScanNet', 'KITTI']: # no need padding
        img0, img1, _, _ = dataloader.load_images(cfg.crop_size_W, cfg.crop_size_H)
        pmer.set_corr_num_init(cfg.match_num)
        ori_corrs = pmer.match(img0, img1, None, None)
        ori_corrs = dataloader.tune_corrs_size_to_eval(ori_corrs, cfg.crop_size_W, cfg.crop_size_H, cfg.eval_from_size_W, cfg.eval_from_size_H)

    elif cfg.dataset_name in ['MegaDepth', 'YFCC', 'ETH3D']: # need padding
        color = pmer.name() in ['Mast3rMatcher', 'Dust3rMatcher'] # mast3r needs color image
        
        if color: logger.warning(f"using color image for mast3r")
        
        img0, mask0, match_in_W0, match_in_H0,\
        img1, mask1, match_in_W1, match_in_H1 = dataloader.load_images(cfg.crop_size_W, cfg.crop_size_H, PMer=True, color=color)
        pmer.set_corr_num_init(cfg.match_num)
        
        if color:
            assert img0.shape[2] == 3 and img1.shape[2] == 3, f"img0 and img1 should have 3 channels for mast3r, but got {img0.shape[2]} and {img1.shape[2]}"
        
        ori_corrs = pmer.match(img0, img1, mask0, mask1)
        ori_corrs = dataloader.tune_corrs_size_to_eval(ori_corrs,\
                                                match_in_W0, match_in_H0,\
                                                match_in_W1, match_in_H1) # eval size is the same as the original image size
    else:
        raise NotImplementedError(f"dataset {cfg.dataset_name} not supported")

    # test amer
    amer = hydra.utils.instantiate(cfg.area_matcher)
    logger.info(f"amer: {amer.name()}")
    validate_type(amer, AbstractAreaMatcher)
    area_matches0, area_matches1 = amer.area_matching(dataloader, cfg.out_path)

    logger.success(f"area matching done, area_matches len: {len(area_matches0)}")

    if cfg.test_area_acc:
        # test area accuracy
        evaler.eval_area_overlap_ratio(area_matches0, area_matches1, 'am')

    # test gam
    gamer = hydra.utils.instantiate(cfg.geo_area_matcher)
    validate_type(gamer, AbstractGeoAreaMatcher)
    logger.info(f"gamer: {gamer.name()}")
    gamer.init_gam(
        dataloader=dataloader,
        point_matcher=pmer,
        ori_corrs=ori_corrs,
        out_path=cfg.out_path
    )
    alpha_corrs_dict, alpha_inlier_idxs_dict, _ = gamer.geo_area_matching_refine(area_matches0, area_matches1)

    logger.success(f"geo area matching done")
    for alpha in alpha_corrs_dict.keys():
        logger.success(f"for alpha: {alpha}, areas num: {len(alpha_inlier_idxs_dict[alpha])}")

    if cfg.test_area_acc:
        # test area accuracy
        for alpha in alpha_inlier_idxs_dict.keys():
            # get inlier area for each alpha
            inlier_areas0 = [area_matches0[i] for i in alpha_inlier_idxs_dict[alpha]]
            inlier_areas1 = [area_matches1[i] for i in alpha_inlier_idxs_dict[alpha]]
            evaler.eval_area_overlap_ratio(inlier_areas0, inlier_areas1, f'am+gam-{alpha}')

    # test point matching accuracy
    if cfg.test_pm_acc:
        thds = cfg.pm_acc_thds

        # for pmer
        logger.success(f"ori corrs matching accuracy from pmer {pmer.name()} are: ")
        evaler.eval_point_match(ori_corrs, 'pm', thds)

        # for a2pmer
        for alpha in alpha_corrs_dict.keys():
            logger.success(f"for alpha: {alpha} of {cfg.name}, matching accuracies are: ")
            corrs = list_of_corrs2corr_list(alpha_corrs_dict[alpha])
            evaler.eval_point_match(corrs, f'a2pm-{alpha}', thds)


    # test pose error
    if cfg.test_pose_err:
        # for pmer
        logger.success(f"ori corrs pose error from pmer {pmer.name()} are: ")
        pose_err = evaler.eval_pose_error(ori_corrs, 'pm')
        
        # for a2pmer
        for alpha in alpha_corrs_dict.keys():
            logger.success(f"for alpha: {alpha} of {cfg.name}, pose errors are: ")
            corrs = list_of_corrs2corr_list(alpha_corrs_dict[alpha])
            pose_err = evaler.eval_pose_error(corrs, f'a2pm-{alpha}')

if __name__ == "__main__":
    test()
    pass