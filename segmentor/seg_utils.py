'''
Author: EasonZhang
Date: 2024-07-19 20:42:13
LastEditors: EasonZhang
LastEditTime: 2024-09-11 15:16:03
FilePath: /SA2M/hydra-mesa/segmentor/seg_utils.py
Description: TBD

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import cv2
import numpy as np
import os.path as osp

class MaskViewer(object):
    """ Mask Visualization
    """
    def __init__(self, save_path) -> None:
        """
        """
        self.save_path = save_path

    def draw_single_mask(self, mask, bbox, name):
        """
        """
        mask_show = mask.astype(np.uint8) * 255
        # to color img
        mask_show = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2BGR)
        # draw bbox
        cv2.rectangle(mask_show, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 0, 255), 2)
        
        cv2.imwrite(osp.join(self.save_path, f"{name}.jpg"), mask_show)
        

    def draw_multi_masks_in_one(self, area_info_list, W, H, name="", key="mask"):
        """
        """
        masks_show = np.zeros((H, W, 3), dtype=np.uint8)
        exsit_colors = []

        for area_info in area_info_list:
            mask = area_info[key].astype(np.uint8)
            mask = cv2.resize(mask, (W, H))
            color = np.random.randint(0, 255, size=3)
            while tuple(color.tolist()) in exsit_colors:
                color = np.random.randint(0, 255, size=3)
            masks_show[mask > 0] = color
            color = tuple(color.tolist())
            
            if key == "mask":
                bbox = area_info["area_bbox"]
                # turn color to scalar
                # draw bbox
                cv2.rectangle(masks_show, (bbox[0], bbox[2]), (bbox[1], bbox[3]), color, 2)
            exsit_colors.append(color)
        
        cv2.imwrite(osp.join(self.save_path, f"{name}.png"), masks_show)
