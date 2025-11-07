'''
Author: EasonZhang
Date: 2024-06-12 20:31:50
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-07 15:12:36
FilePath: /SA2M/hydra-mesa/scripts/demo.py
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
from utils.common import validate_type
from utils.geo import list_of_corrs2corr_list

import random
# fix random seed
random.seed(2)

import torch
# fix random seed
torch.manual_seed(2)

# @hydra.main(version_base=None, config_path="../conf", config_name="a2pm_mesa_egam_dkm_scannet")
# @hydra.main(version_base=None, config_path="../conf", config_name="a2pm_mesa_egam_spsg_scannet")
@hydra.main(version_base=None, config_path="../conf")
def test(cfg: DictConfig) -> None:
    """
    Test point matcher
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
    elif cfg.verbose==2:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    else:
        raise NotImplementedError(f"verbose {cfg.verbose} not supported")

    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # load point matcher
    pmer = hydra.utils.instantiate(cfg.point_matcher)
    validate_type(pmer, AbstractPointMatcher)
    pmer.set_corr_num_init(cfg.match_num)

    # load dataloader
    dataloader = hydra.utils.instantiate(cfg.dataset)
    validate_type(dataloader, AbstractDataloader)

    img0, img1, _, _ = dataloader.load_images(cfg.crop_size_W, cfg.crop_size_H)
    ori_corrs = pmer.match(img0, img1, None, None)

    # test amer
    amer = hydra.utils.instantiate(cfg.area_matcher)
    logger.info(f"amer: {amer.name()}")
    validate_type(amer, AbstractAreaMatcher)
    area_matches0, area_matches1 = amer.area_matching(dataloader, cfg.out_path) # here is the area matches from the area matcher, areas are represented by 4 coordinate list: [u_min, u_max, v_min, v_max]

    logger.success(f"area matching done, area_matches len: {len(area_matches0)}")

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
        # get inlier area matches
        inlier_area_matches0 = [area_matches0[i] for i in alpha_inlier_idxs_dict[alpha]]
        inlier_area_matches1 = [area_matches1[i] for i in alpha_inlier_idxs_dict[alpha]]

        # NOTE: description about how to use the matched areas
        # areas are represented by 4 coordinate list: [u_min, u_max, v_min, v_max]
        # the matched areas are stored in inlier_area_matches0 and inlier_area_matches1
        # for example, inlier_area_matches0[0]=[u0_min, u0_max, v0_min, v0_max] is matched with inlier_area_matches1[0]=[u1_min, u1_max, v1_min, v1_max]
        # you can choose a proper alpha by modifying the `alpha_list` in the config file of the geo area matcher you used
        # such as `alpha_list: [0.1, 0.2, 0.3, 0.4, 0.5]` in L17 of `conf/geo_area_matcher/egam.yaml`


    for alpha in alpha_corrs_dict.keys():
        # draw
        corrs = list_of_corrs2corr_list(alpha_corrs_dict[alpha])
        logger.success(f"alpha: {alpha}, corrs num: {len(corrs)}")
        #TODO: draw corrs
 
if __name__ == "__main__":
    test()
    pass