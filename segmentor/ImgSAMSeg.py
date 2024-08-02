
'''
Author: EasonZhang
Date: 2023-06-15 17:06:04
LastEditors: EasonZhang
LastEditTime: 2024-07-23 10:47:51
FilePath: /SA2M/hydra-mesa/segmentor/ImgSAMSeg.py
Description: A script to segment the image using SAM

Copyright (c) 2023 by EasonZhang, All Rights Reserved. 
'''

import sys
sys.path.append("..")

import os

import argparse
import numpy as np
from loguru import logger
logger.remove()#删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
handler_id = logger.add(sys.stderr, level="INFO")#添加一个可以修改控制的handler
import cv2

from segmentor.SAMSeger import SAMSeger

# current file path
current_path = os.path.dirname(os.path.abspath(__file__))

SAM_configs = {
    "W": 640,
    "H": 480,
    "sam_model_type": "vit_h",
    "sam_model_path": f"{current_path}/../SAM/sam_vit_h_4b8939.pth",
    "save_folder": "",
    "points_per_side": 16,
}

def SMASeg(args):
    """
    """

    SAM_configs["save_folder"] = args.save_folder
    SAM_configs["W"] = args.W
    SAM_configs["H"] = args.H

    sam_seger = SAMSeger(configs=SAM_configs)

    img_path = args.img_path
    img = sam_seger.img_loader(path=img_path)
    sam_res = sam_seger.segment(img_path=img_path, save_name=args.save_name, save_img_flag=False)

def args_achieve():
    """
    """
    parser = argparse.ArgumentParser(description="A script to segment the image using SAM")
    parser.add_argument("--img_path", type=str, default="/data0/zys/A2PM/data/ScanData/scene0000_00/color/12.jpg", help="The path of the image to be segmented")
    parser.add_argument("--save_folder", type=str, default="/data0/zys/A2PM/testAG/res", help="The folder to save the segmented image")
    parser.add_argument("--save_name", type=str, default="SAMRes", help="The name of the segmented image")
    parser.add_argument("--W", type=int, default=640, help="The width of the image")
    parser.add_argument("--H", type=int, default=480, help="The height of the image")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = args_achieve()
    SMASeg(args)