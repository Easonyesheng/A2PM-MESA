'''
Author: EasonZhang
Date: 2024-06-13 23:07:56
LastEditors: EasonZhang
LastEditTime: 2024-06-28 19:42:21
FilePath: /SA2M/hydra-mesa/utils/load.py
Description: load utils

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''
import numpy as np
import yaml
import cv2
from loguru import logger
import os
import glob
from itertools import combinations

def load_cv_img_resize(img_path, W, H, mode=0):
    """
    """
    assert os.path.exists(img_path), f"img path {img_path} not exists"
    img = cv2.imread(img_path, mode)
    logger.info(f"load img from {img_path} with size {img.shape} resized to {W} x {H}")
    if mode == 1:
        H_ori, W_ori, _ = img.shape
    else:
        H_ori, W_ori = img.shape
    scale_u = W / W_ori
    scale_v = H / H_ori
    # print(f"ori W, H: {W_ori} x {H_ori},  with scale: {scale_u} , {scale_v}")
    img = cv2.resize(img, (W, H), cv2.INTER_AREA) # type: ignore
    return img, [scale_u, scale_v]

def load_cv_depth(depth_path):
    """ for ScanNet Dataset
    """
    assert os.path.exists(depth_path), f"depth path {depth_path} not exists"
    logger.info(f"load depth from {depth_path}")
    return cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

def load_K_txt(intri_path, scale=[1, 1]):
    """For ScanNet K
    Args:
        scale = [scale_u, scale_v]
    """
    assert os.path.exists(intri_path), f"intri path {intri_path} not exists"
    K = np.loadtxt(intri_path)
    fu = K[0,0] * scale[0]
    fv = K[1,1] * scale[1]
    cu = K[0,2] * scale[0]
    cv = K[1,2] * scale[1]
    K_ = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])

    logger.info(f"load K from {intri_path} with scale {scale} is \n {K_}")
    return np.matrix(K_)

def load_pose_txt(pose_path):
    """For ScanNet pose: cam2world
        txt file with 
        P =
            |R t|
            |0 1|
            
    Returns:
        P : np.mat
    """
    assert os.path.exists(pose_path), f"pose path {pose_path} not exists"
    P = np.loadtxt(pose_path)
    P = np.matrix(P)
    logger.info(f"load pose is \n{P}")
    return P

