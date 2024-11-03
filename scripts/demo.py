'''
Author: EasonZhang
Date: 2024-06-12 20:31:50
LastEditors: EasonZhang
LastEditTime: 2024-11-03 15:05:05
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
    area_matches0, area_matches1 = amer.area_matching(dataloader, cfg.out_path)

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


    for alpha in alpha_corrs_dict.keys():
        # draw
        corrs = list_of_corrs2corr_list(alpha_corrs_dict[alpha])
        logger.success(f"alpha: {alpha}, corrs num: {len(corrs)}")
        #TODO: draw corrs
 
if __name__ == "__main__":
    test()
    pass