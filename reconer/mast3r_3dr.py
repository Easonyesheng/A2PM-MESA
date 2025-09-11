'''
Author: EasonZhang
Date: 2024-09-03 15:07:47
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-10 11:26:54
FilePath: /3DGS/mast3r-main/reconers/mast3r_3dr.py
Description: TBD

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''
import sys
sys.path.append("..")  # noqa
import os
import numpy as np
from .mast3r.mast3r.model import AsymmetricMASt3R
from .mast3r.mast3r.demo import get_reconstructed_scene
from loguru import logger
import cv2

from .mast3r_param import SceneParams
from .abstract_3dr import Abstract3dr

from utils.common import test_dir_if_not_create, norm_conf

class Mast3r3dr(Abstract3dr):
    def __init__(self, 
        model_path, 
        match_conf_thd,
        draw_verbose=False,
        orimg_recon_size=512,
        device='cuda'):
        super().__init__(draw_verbose)

        self.model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
        self.device = device
        self.orimg_recon_size = orimg_recon_size
        self.match_conf_thd = match_conf_thd

    def get_outdir(self, outdir):
        outdir = os.path.join(outdir, 'recon')
        self.outdir = outdir
    
    def name(self):
        return 'MASt3R_3DR'

    def init_dataloader(self, dataloader):
        # img paths
        img0_path, img1_path = dataloader.img0_path, dataloader.img1_path
        self.img_lists = [img0_path, img1_path]

    def init_aim_dataloader(self, dataloader, idx):
        # img paths
        img0_path, img1_path = dataloader.get_area_imgs(idx)
        self.img_lists = [img0_path, img1_path]
        
    def calc_corres_shape(self, ):
        # read one image to get shape
        img0 = cv2.imread(self.img_lists[0])
        ih, iw = img0.shape[0], img0.shape[1]
        assert ih % 32 == 0 and iw % 32 == 0, f"img shape {img0.shape} not support"
        # resize with long side = orimg_recon_size
        if ih >= iw:
            scale = self.orimg_recon_size / ih
            iw = int(iw * scale)
            ih = self.orimg_recon_size
        else:
            scale = self.orimg_recon_size / iw
            ih = int(ih * scale)
            iw = self.orimg_recon_size
        
        # if we 
    def recon_scene(self):
        try:
            # scene params
            scene_params = SceneParams(
                outdir=self.outdir,
                filelist=self.img_lists,
                model=self.model,
                image_size=self.orimg_recon_size,
            )
        except Exception as e:
            print(f"Error: {e}")
            return None

        # get reconstructed scene
        # images are resized to with long side = 512 and center-cropped to %16==0
        self.scene, self.corres = get_reconstructed_scene(**scene_params.params)
        
        return self.scene, self.corres

    def recon_orimgs(self):
        self.outdir = os.path.join(self.outdir, 'orimgs')
        test_dir_if_not_create(self.outdir)
        scene, corres = self.recon_scene()
        self.save_3d_info(scene)
        self.save_corres(corres)
        return scene

    def save_aim_pts3d_depth_conf(self, dataloader, idx):
        """
        """
        self.init_aim_dataloader(dataloader, idx)
        self.outdir = os.path.join(self.outdir, f'area_{idx}')
        test_dir_if_not_create(self.outdir)
        scene, corres = self.recon_scene()

        self.save_3d_info(scene)

    def save_corres(self, corres, save_flag=True):
        # logger.info(f"len(corres): {len(corres)}")
        # logger.info(f"corres[0]: {corres[0]}")
        # logger.info(f"{corres[0][1].shape}")
        # logger.info(f"corres[1]: {corres[1]}")
        # logger.info(f"{corres[1][1].shape}")
        matched_pts0 = corres[0][1].cpu().numpy()
        matched_confs0 = corres[0][2].cpu().numpy()
        matched_pts1 = corres[1][1].cpu().numpy()
        normed_confs = norm_conf(matched_confs0)

        # filter out invalid matches
        valid_idx = matched_confs0 > self.match_conf_thd
        matched_pts0 = matched_pts0[valid_idx]
        matched_pts1 = matched_pts1[valid_idx]

        # save as [[u0, v0, u1, v1], ...]
        corres = np.concatenate([matched_pts0, matched_pts1], axis=1)
        logger.info(f"corres shape: {corres.shape}")

        if save_flag:
            logger.info(f"save to {os.path.join(self.outdir, 'corres.npy')}")
            np.save(os.path.join(self.outdir, 'corres.npy'), corres)

        return corres


    def save_3d_info(self, scene, save_flag=False, save_name=""):
        # save pts3d, depths, confs
        pts3d_d, depths_d, confs_d = scene.get_dense_pts3d()

        pts3d_d = [p.cpu().numpy() for p in pts3d_d]
        pts3d_d = np.array(pts3d_d) # 2x(H*W)x3

        depths_d = [d.cpu().numpy() for d in depths_d]
        depths_d = np.array(depths_d) # 2x(H*W)

        confs_d = [c.cpu().numpy() for c in confs_d]
        confs_d = np.array(confs_d) # 2x(H*W)

        if save_flag:
            logger.info(f"pts3d_d shape: {pts3d_d.shape}")
            logger.info(f"depths_d shape: {depths_d.shape}")
            logger.info(f"confs_d shape: {confs_d.shape}")
            logger.info(f"save to {os.path.join(self.outdir, 'pts3d.npy')}, {os.path.join(self.outdir, 'depths.npy')}, {os.path.join(self.outdir, 'confs.npy')}")
            # save
            np.save(os.path.join(self.outdir, f'{save_name}_pts3d.npy'), pts3d_d)
            np.save(os.path.join(self.outdir, f'{save_name}_depths.npy'), depths_d)
            np.save(os.path.join(self.outdir, f'{save_name}_confs.npy'), confs_d)
            
            im_poses = scene.get_im_poses()
            im_poses = [p.cpu().numpy() for p in im_poses]
            im_poses = np.array(im_poses)
            np.save(os.path.join(self.outdir, 'im_poses.npy'), im_poses)

            intrins = scene.intrinsics
            intrins = [i.cpu().numpy() for i in intrins]
            intrins = np.array(intrins)
            np.save(os.path.join(self.outdir, 'intrins.npy'), intrins)
        
        return pts3d_d, depths_d, confs_d