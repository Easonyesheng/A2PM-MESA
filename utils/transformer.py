'''
Author: EasonZhang
Date: 2023-09-22 17:04:24
LastEditors: EasonZhang
LastEditTime: 2023-12-27 14:56:40
FilePath: /A2PM/utils/transformer.py
Description: including some functions for transforming
    - transform SEEM segmentation results to SAM format
        - SAM format dict:
            - "segmentation" : segmentation binary mask
            - "bbox" : bounding box of the segmentation: [x1, y1, x2, y2]

Copyright (c) 2023 by EasonZhang, All Rights Reserved. 
'''

import numpy as np
import cv2
import os
from loguru import logger


class SEEM2SAM(object):
    """ one folder a time
    """
    cfg_dft = {
        "root_path": "",
        "floder_name": "",
        "out_path": "",
    }

    def __init__(self, cfg={}):
        """
        """
        self.cfg = {**self.cfg_dft, **cfg}
        self.root_path = self.cfg["root_path"]
        self.floder_name = self.cfg["floder_name"]
        self.out_path = self.cfg["out_path"]

    def load_seem_seg_img_name_list(self):
        """ load all png images in the folder
        
        """
        img_name_list = []
        for file_ in os.listdir(f"{self.root_path}/{self.floder_name}"):
            if file_.endswith(".png"):
                img_name_list.append(file_)
        return img_name_list

    def load_seem_seg_img(self, img_name):
        """ load png image
        
        """
        img = cv2.imread(f"{self.root_path}/{self.floder_name}/{img_name}", -1)
        logger.info(f"load {img_name} with shape {img.shape}")
        return img

    def trans_png2npy(self, img, img_name="", save=True):
        """ transform segmentation png image to dict and save as npy file
        Args:
            img (np.ndarray): segmentation png image
                - each pixel value is the class index
                - get the same-values pixels as the segmentation
                - get the bounding box of the segmentation
        Returns:
            a list of dict: segmentation dict
                - "segmentation" : segmentation binary mask
                - "bbox" : bounding box of the segmentation: [x, y, w, h]
        """
        # get all class index
        class_index_list = np.unique(img)

        # for each class index, get the segmentation and bounding box
        segmentation_dicts = []
        for class_index in class_index_list:
            # if class_index == 0:
            #     continue
            # get the segmentation
            segmentation = np.zeros_like(img)
            segmentation[img == class_index] = 1

            segmentation = self.get_connection_area(segmentation)

            # get the bounding box
            bbox = self.get_bbox(segmentation) # [x, y, w, h]

            # # draw the bounding box
            # save_path = f"/data2/zys/A2PM/testAGC/{class_index}.jpg"
            # color = np.random.randint(0, 255, (3))
            # x1 = bbox[0]
            # y1 = bbox[1]
            # x2 = bbox[0] + bbox[2]
            # y2 = bbox[1] + bbox[3]
            # # covert segmentation to color image
            # segmentation_show = np.zeros_like(segmentation)
            # segmentation_show = cv2.cvtColor(segmentation_show, cv2.COLOR_GRAY2BGR)
            # segmentation_show[segmentation == 1] = color
            # cv2.rectangle(segmentation_show, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)
            # cv2.imwrite(save_path, segmentation_show)

            # save as dict
            segmentation_dict = {
                "segmentation": segmentation,
                "bbox": bbox,
            }
            segmentation_dicts.append(segmentation_dict)

        if save:
            save_path = f"{self.out_path}/{self.floder_name}"
            logger.info(f"save segmentation dicts to {save_path}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.save(f"{save_path}/{img_name[:-4]}.npy", segmentation_dicts)

        return segmentation_dicts

    def get_connection_area(self, segmentation):
        """ get the connection area of the segmentation, only save the biggest one
        Args:
            segmentation (np.ndarray): segmentation binary mask
        """

        bin_img = self._get_bin_img(segmentation)

        # get the connection area
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

        # get the biggest area
        max_area = 0
        max_area_idx = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_area_idx = i
        
        # return the biggest area img
        max_area_img = np.zeros_like(bin_img)
        max_area_img[labels == max_area_idx] = 1

        return max_area_img
    
    def _get_bin_img(self, img):
        """ get binary image
        """
        H, W = img.shape[0], img.shape[1]
        bin_img = np.zeros((H, W), dtype=np.uint8)

        bin_img[np.where(img==1)] = 255
        
        _, bin_img_rt = cv2.threshold(bin_img, 10, 255, cv2.THRESH_BINARY)

        kernel_close = np.ones((13,13), np.uint8)
        close = cv2.morphologyEx(bin_img_rt, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((5,5),np.uint8) 
        opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel_open)

        # cv2.imwrite(os.path.join(self.out_path, "bin_img_" + str(pix_val)+"_"+str(name) + ".jpg"), opening)
        return opening

    def get_bbox(self, segmentation):
        """ get the bounding box of the segmentation
        Args:
            segmentation (np.ndarray): segmentation binary mask
        Returns:
            bbox (list): bounding box of the segmentation: [u, v, w, h]
        """
        # get the index of the segmentation
        index = np.where(segmentation == 1)
        # u is the width-axis, v is the height-axis
        u_min, u_max = np.min(index[1]), np.max(index[1])
        v_min, v_max = np.min(index[0]), np.max(index[0])

        w = u_max - u_min
        h = v_max - v_min

        bbox = [u_min, v_min, w, h]

        return bbox


    def show_res(self, img_name, segmentation_dicts):
        """ show the segmentation results
        Args:
            img (np.ndarray): original image
            segmentation_dicts (list): segmentation dicts
        """
        H, W = segmentation_dicts[0]["segmentation"].shape
        img = np.zeros((H, W, 3), dtype=np.uint8)

        for segmentation_dict in segmentation_dicts:
            segmentation = segmentation_dict["segmentation"]
            bbox = segmentation_dict["bbox"]

            # show the segmentation
            color = np.random.randint(0, 255, (3))
            img[segmentation == 1] = color

            # show the bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (int(color[0]), int(color[1]), int(color[2])), 2)

        cv2.imwrite(f"{self.out_path}/{self.floder_name}/{img_name}", img)
        logger.info(f"save {img_name} with shape {img.shape}")

    def run(self, show_idx_end=-1):
        """
        """
        img_name_list = self.load_seem_seg_img_name_list()
        for i, img_name in enumerate(img_name_list):
            img = self.load_seem_seg_img(img_name)
            seg_dicts = self.trans_png2npy(img, img_name)
            if i <= show_idx_end:
                self.show_res(img_name, seg_dicts)
