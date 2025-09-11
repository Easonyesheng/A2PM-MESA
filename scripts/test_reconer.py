'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2025-09-09 12:56:32
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-10 10:48:50
FilePath: /A2PM-MESA/scripts/test_reconer.py
Description: test reconer 
'''

import sys
sys.path.append('..')

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger


from dataloader.abstract_dataloader import AbstractDataloader
from reconer.abstract_3dr import Abstract3dr
from utils.common import validate_type


import random
# fix random seed
random.seed(2)

import torch
# fix random seed
torch.manual_seed(2)

@hydra.main(version_base=None, config_path="../conf")
def test(cfg: DictConfig) -> None:
    """
    Test reconer
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

    # load dataloader
    dataloader = hydra.utils.instantiate(cfg.dataset)
    validate_type(dataloader, AbstractDataloader)

    # load reconer
    reconer = hydra.utils.instantiate(cfg.reconer)
    validate_type(reconer, Abstract3dr)


    # outdir
    outdir = os.path.join(cfg.out_path, 'test_reconer')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    reconer.get_outdir(outdir)

    # init dataloader for reconer
    reconer.init_dataloader(dataloader)

    # recon scene
    scene, corres = reconer.recon_scene()

    # vis corres
    corres = reconer.save_corres(corres)
    reconer.vis_corres(corres, name='test_recon_corres')

    # save scene
    pts3d, _, _ = reconer.save_3d_info(scene)
    # save as ply
    reconer.save_as_ply(pts3d, name='test_recon_pts3d')


if __name__ == "__main__":
    test()