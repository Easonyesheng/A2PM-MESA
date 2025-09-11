'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2025-09-09 11:34:00
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-10 11:00:16
FilePath: /A2PM-MESA/reconer/abstract_3dr.py
Description: abstract class for 3d reconer
'''


import numpy as np
import abc

import open3d as o3d
import os
import cv2
from loguru import logger

from utils.common import test_dir_if_not_create


class Abstract3dr(abc.ABC):
    def __init__(self, draw_verbose) -> None:
        self.draw_verbose = draw_verbose
        pass

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def init_dataloader(self, dataloader):
        """get all the necessary info from dataloader
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recon_scene(self):
        """ Main Func
        Returns:
            pixel-wise depth map
            pixel-wise 3d point cloud
            two view poses
        """
        raise NotImplementedError

    def set_outpath(self, outpath: str):
        """Run after init_dataloader 
        """
        self.out_path = os.path.join(outpath, f"{self.scene_name}_{self.name0}_{self.name1}", 'am')

        if self.draw_verbose == 1:
            test_dir_if_not_create(self.out_path)

    
    def save_as_ply(self, pts3d, name=""):
        """ save pts3d as ply
        Args:
            pts3d: np.array, shape Nx3
        """
        pcd = o3d.geometry.PointCloud()
        try:
            pcd.points = o3d.utility.Vector3dVector(pts3d)
        except Exception as e:
            logger.error(f"Error in save_as_ply: {e}")
            logger.warning(f"pts3d shape: {pts3d.shape}, pts3d.type: {type(pts3d)}")
            raise e


        save_path = os.path.join(self.out_path, f"{name}.ply")
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"save ply to {save_path}")


    def vis_corres(self, corres, name=""):
        """ used after save_corres
        Args:
            corres: np.array, shape Nx4, [[u0, v0, u1, v1], ...]
        """
        # load images
        img0 = cv2.imread(self.img_lists[0])
        img1 = cv2.imread(self.img_lists[1])
        logger.info(f"img0 shape: {img0.shape}, img1 shape: {img1.shape}")

        # suppose corres is [[u0, v0, u1, v1], ...] in the original image size
        pts0 = corres[:, :2]
        pts1 = corres[:, 2:]
        pts0 = pts0.astype(np.int32)
        pts1 = pts1.astype(np.int32)
        logger.info(f"pts0 shape: {pts0.shape}, pts1 shape: {pts1.shape}")
       
        concat_img = np.concatenate([img0, img1], axis=1)
        for i in range(pts0.shape[0]):
            color = (0, 255, 0)
            cv2.circle(concat_img, (pts0[i, 0], pts0[i, 1]), 3, color, -1)
            cv2.circle(concat_img, (pts1[i, 0] + img0.shape[1], pts1[i, 1]), 3, color, -1)
            cv2.line(concat_img, (pts0[i, 0], pts0[i, 1]), (pts1[i, 0] + img0.shape[1], pts1[i, 1]), color, 1)
        
        save_path = os.path.join(self.outdir, f'corres_vis{name}.png')
        cv2.imwrite(save_path, concat_img)
        logger.info(f"save corres vis to {save_path}")