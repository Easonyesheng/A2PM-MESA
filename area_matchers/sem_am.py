'''
Author: EasonZhang
Date: 2024-06-20 11:43:03
LastEditors: EasonZhang
LastEditTime: 2024-06-20 22:13:12
FilePath: /SA2M/hydra-mesa/area_matchers/sem_am.py
Description: semantic area matcher, part of SGAM

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''


import numpy as np
import os
import math
import cv2
from loguru import logger
import matplotlib
from sklearn.metrics import hamming_loss
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt

from dataloader.abstract_dataloader import AbstractDataloader
from .abstract_am import AbstractAreaMatcher
from utils.vis import (
    paint_semantic,
)
from utils.common import test_dir_if_not_create, validate_type
from utils.img_process import (
    patch_adjust_with_square_min_limit,
    img_crop_without_Diffscale,
)
from utils.geo import adopt_K, calc_euc_dist_2d



class SemAreaMatcher(AbstractAreaMatcher):
    """SemAMer
    Work Flow
        - classify into object & overlap area
        - hand-crafted descriptor & extractor
    Main Func
        FindMatchArea()
    """
    def __init__(self, 
        semantic_mode,
        datasetName, 
        W,
        H,
        connected_thd,
        radius_thd_up,
        radius_thd_down,
        desc_type,
        small_label_filted_thd_on_bound,
        small_label_filted_thd_inside_area,
        combined_obj_dist_thd,
        leave_multi_obj_match,
        obj_desc_match_thd, 
        same_overlap_dist,
        label_list_area_thd,
        overlap_radius,
        overlap_desc_dist_thd,
        inv_overlap_pyramid_ratio,
        output_patch_size,
        draw_verbose=0) -> None:
        logger.info(f"AreaMatcher by raw semantic is initilaized")

        self.semantic_mode = semantic_mode
        self.datasetName = datasetName
        self.W = W
        self.H = H
        self.size = [W, H]
        self.draw_verbose = draw_verbose
        

        # specific for SAM    
        self.connected_thd = connected_thd
        self.radius_thd = [radius_thd_up, radius_thd_down]
        self.desc_type = desc_type # 2 - bin desc with order

        self.small_label_filted_thd_on_bound = small_label_filted_thd_on_bound
        self.small_label_filted_thd_inside_area = small_label_filted_thd_inside_area
        self.combined_obj_dist_thd = combined_obj_dist_thd
        self.obj_desc_match_thd = obj_desc_match_thd
        self.leave_multi_obj_match = leave_multi_obj_match
        self.same_overlap_dist = same_overlap_dist  # overlap area gather distance 
        self.label_list_area_thd = label_list_area_thd
        self.overlap_radius = overlap_radius
        self.overlap_desc_dist_thd = overlap_desc_dist_thd
        self.overlap_pyramid_ratio = 1/inv_overlap_pyramid_ratio
        self.output_patch_size = output_patch_size
        

        if datasetName == "ScanNet":
            self.semantic_mode = semantic_mode
            if self.semantic_mode == "":
                logger.warning("No semantic mode is specified")

            if self.semantic_mode == "GT":
            # for ScanNet GT semantic segmentation
                self.non_obj_dict = {
                    "wall": 1,
                    "floor": 3,
                    "window": 16,
                    "window1": 0,
                    "ceiling": 41,
                    "curtain": 21,
                    "whiteboard": 52,
                    "shower_walls": 128,
                    "board": 69,
                    "closet_wall": 1165,
                    "cabinets": 7,
                    "doors": 5,
                    "bathroom_stall": 125,
                    "bar": 145,
                    "bathroom_stall_door": 1167,
                    "decoration": 250,
                    "rail": 95,
                    "floor_mat": 140,
                    "ledge": 193,
                    "person": 80,
                    "windowsill": 141,
                    "closet": 57,
                    "stair_rail": 1171,
                    "shower_curtain_rod": 170,
                    "tube": 1172,
                    "closet_doors": 276,
                    "stairs": 59,
                    "blinds": 86,
                    "plunger": 300,
                    "ladder": 122,
                    "pipe": 107,
                    "rack": 87,
                    "shower floor": 417,
                    "shower door": 188,
                    "curtains": 21,
                    "hand rail": 1192,
                    "pantry wall": 1204,
                    "mirror doors": 1208,
                    "dart board": 1215,
                    "closet ceiling": 1225,
                    "dollhouse": 282,
                    "bath walls": 1243,
                    "cabinet door": 385,
                    "yoga mat": 1267,
                    "garage door": 976,
                    "closet wardrobe": 99,
                    "curtain rod": 513,
                    "glass doors": 649,
                    "door wall": 947,
                    "sliding door": 569,
                    "closet doorframe": 1345,
                    "closet floor": 1347
                }
            elif self.semantic_mode == "SEEM":
                self.non_obj_dict = {
                    # "background": 0,
                }
        elif datasetName == "MatterPort3D":
            self.non_obj_dict = {"bg":0}
        elif datasetName == "KITTI" or datasetName == "YFCC":
            self.non_obj_dict = {}
        else:
            raise NotImplementedError

        self.matched_obj_label = []
        self.matched_obj_patch0 = []
        self.matched_obj_patch1 = []

        # self.draw_ori_img()
        # self.get_label_list() after connected area 
    
    def name(self):
        return "SemAM"

    def init_dataloader(self, dataloader):
        """
        """
        validate_type(dataloader, AbstractDataloader)
        sem_path0, sem_path1 = dataloader.get_sem_paths()

        self.name0 = sem_path0.split('/')[-1].split('.')[0]
        self.name1 = sem_path1.split('/')[-1].split('.')[0]

        self.scene_name = dataloader.scene_name

        self.sem0, self.sem1 = dataloader.load_semantics(self.W, self.H)

        self.color0, self.color1, self.scale0, self.scale1 = dataloader.load_images(self.W, self.H)


    def rt_names(self):
        """
        """
        return self.name0, self.name1


    def get_label_list_collect(self):
        """ use collect
        Achieve main sem. label, get rid of filtered noise
        """
        logger.info(f"use collection to get lable list")
        self.label_list = []
        self.label_list0 = []
        self.label_list1 = []
        W, H = self.size

        # img0
        temp_sem_list = np.squeeze(self.sem0.reshape((-1,1))).tolist()
        temp_stas_dict = collections.Counter(temp_sem_list)
        # logger.debug(f" img0 got collections: {temp_stas_dict}")

        for label in temp_stas_dict.keys():
            if temp_stas_dict[label] > self.label_list_area_thd:
                if label not in self.label_list:
                    self.label_list.append(label)
                if label not in self.label_list0:
                    self.label_list0.append(label)
        
        # logger.debug(f" img0 got labels: {self.label_list0}")

        # img1
        temp_sem_list = np.squeeze(self.sem1.reshape((-1,1))).tolist()
        temp_stas_dict = collections.Counter(temp_sem_list)

        for label in temp_stas_dict.keys():
            if temp_stas_dict[label] > self.label_list_area_thd:
                if label not in self.label_list:
                    self.label_list.append(label)
                if label not in self.label_list1:
                    self.label_list1.append(label)
        
        # logger.debug(f" img1 got labels: {self.label_list1}")

        self.label_size = len(self.label_list)
        return self.label_list
    
    def static_connected_area_upspeed(self, sem, label_list, name=0):
        """ use opencv
        Returns:
            sem_connect_dict = {
                label: [[connect_area]s]
                    connect_area: [[u, v]s]
            }
        """
        logger.info(f"static connected area starting...")
        sem_connect_dict = {}
        
        for label in label_list:
            temp_bin_img = self._get_bin_img(sem, label, name)
            temp_connect_num, connect_label_img = cv2.connectedComponents(temp_bin_img)
            if label not in sem_connect_dict.keys():
                sem_connect_dict.update({label:[]})
            for connect_label_id in range(1, temp_connect_num+1):
                temp_where_set = np.where(connect_label_img == connect_label_id)
                N = temp_where_set[0].shape[0]
                if N < self.connected_thd:
                    continue
                area_coord_list = self._convert_where_to_uvlist(temp_where_set)
                sem_connect_dict[label].append(area_coord_list)
            
        return sem_connect_dict
    
    def _convert_where_to_uvlist(self, where_set):
        """ 
        Args:
            np.where return: (array([v]), array([u]))
        Returns:
            [[v, u]s]
        """

        rt = []

        for i in range(where_set[0].shape[0]):
            rt.append([where_set[0][i], where_set[1][i]])
        
        return rt

    def _get_bin_img(self, img, pix_val, name=0):
        """
        """
        H, W = img.shape[0], img.shape[1]
        bin_img = np.zeros((H, W), dtype=np.uint8)

        bin_img[np.where(img==pix_val)] = 255
        
        _, bin_img_rt = cv2.threshold(bin_img, 10, 255, cv2.THRESH_BINARY)

        kernel_close = np.ones((13,13), np.uint8)
        close = cv2.morphologyEx(bin_img_rt, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((5,5),np.uint8) 
        opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel_open)

        # cv2.imwrite(os.path.join(self.out_path, "bin_img_" + str(pix_val)+"_"+str(name) + ".jpg"), opening)
        return opening
        
    def get_sem_connected(self):
        """ get connected area 
        Returns:
            dict: {
                label: [[connected-coors: [u,v],...]]
            }
        """
        # speed up
        self.sem0_sem_connected = self.static_connected_area_upspeed(self.sem0, self.label_list0, 0)
        self.sem1_sem_connected = self.static_connected_area_upspeed(self.sem1, self.label_list1, 1)

        return self.sem0_sem_connected, self.sem1_sem_connected
  
    def clean_the_connect_dict(self):
        """clean the invalid connect part
        """
        new_sem0_connected = {}
        new_sem1_connected = {}

        for k in self.sem0_sem_connected.keys():
            temp_list = [x for x in self.sem0_sem_connected[k] if x != []]
            if len(temp_list) != 0:
                new_sem0_connected.update({k:temp_list})

        for k in self.sem1_sem_connected.keys():
            temp_list = [x for x in self.sem1_sem_connected[k] if x != []]
            if len(temp_list) != 0:
                new_sem1_connected.update({k:temp_list})

        self.sem0_sem_connected = new_sem0_connected
        self.sem1_sem_connected = new_sem1_connected

    def find_obj_patch(self, WHRatioThd=4, areaThd=3600, desc_type=0):
        """ find objects 
            patch representation: [u-min, u-max, v-min, v-max, label] 
            Object patch s.t.:
                1. not in non-obj-list
                2. wid-height-ratio should be close to 1
                3. area size should be big enough
        Workflow:
            1. non-obj-list screen
            2. wid-height-ratio screen
            3. construct desc
            4. return
        Args:
            desc_type: 
                0 - set
                1 - bin
        Returns:
            self.sem0_obj_patch = [[u_min, u_max, v_min, v_max, k, desc], ]
            self.sem1_obj_patch = [[u_min, u_max, v_min, v_max, k, desc], ]
        """
        W, H = self.size

        # sem 0
        self.sem0_obj_patch = []
        for k in self.sem0_sem_connected.keys():
            if k in self.non_obj_dict.values():
                # logger.info(f"label {k} is not an object")
                continue
            for i in range(len(self.sem0_sem_connected[k])):
                temp_sem_area = self.sem0_sem_connected[k][i]
                area_size = len(temp_sem_area)
                if area_size < areaThd: continue
                u_max = v_max = 0
                u_min = v_min = 1e5
                for vu in temp_sem_area:
                    v_temp, u_temp = vu

                    if v_temp > v_max: v_max = v_temp
                    if v_temp < v_min: v_min = v_temp
                    
                    if u_temp > u_max: u_max = u_temp
                    if u_temp < u_min: u_min = u_temp
                temp_wid = (u_max - u_min)
                temp_height = (v_max - v_min)
                tempWHRatio = max(temp_wid, temp_height) / (min(temp_wid, temp_height)+1e-5)
                if tempWHRatio > WHRatioThd: 
                    logger.info(f"label {k} has abnormal ratio.")
                    continue
                # construct desc
                if desc_type == 0:
                    logger.warning(f"Not valid desc type = {desc_type}")
                    raise NotImplementedError()
                elif desc_type == 1:
                    temp_desc = self.construct_bin_desc_along_bound_multiscale([u_min, u_max, v_min, v_max], k, self.sem0)
                elif desc_type == 2:
                    temp_desc = self.construct_bin_desc_with_order_along_bound_multiscale([u_min, u_max, v_min, v_max], k, self.sem0)
                else:
                    raise NotImplementedError(f"desc type error")

                self.sem0_obj_patch.append([u_min, u_max, v_min, v_max, k, temp_desc])
        
        objs_info = []
        for obj in self.sem0_obj_patch:
            # logger.info(f"sem0 label {obj[-2]} got desc {obj[-1]}")
            objs_info.append(obj[-2])
        logger.info(f"Achieve object area:\n{objs_info} for sem0")

        # sem 1
        self.sem1_obj_patch = []
        for k in self.sem1_sem_connected.keys():
            if k in self.non_obj_dict.values():
                continue
            for i in range(len(self.sem1_sem_connected[k])):
                temp_sem_area = self.sem1_sem_connected[k][i]
                area_size = len(temp_sem_area)
                if area_size < areaThd: continue
                u_max = v_max = 0
                u_min = v_min = 1e5
                for vu in temp_sem_area:
                    v_temp, u_temp = vu
                    if v_temp > v_max: v_max = v_temp
                    if v_temp < v_min: v_min = v_temp
                    
                    if u_temp > u_max: u_max = u_temp
                    if u_temp < u_min: u_min = u_temp
                temp_wid = (u_max - u_min)
                temp_height = (v_max - v_min)
                tempWHRatio = max(temp_wid, temp_height) / min(temp_wid, temp_height)+1e-5
                if tempWHRatio > WHRatioThd: 
                    continue
                # construct desc
                if desc_type == 0:
                    logger.warning(f"Not valid desc type = {desc_type}")
                    raise NotImplementedError()
                elif desc_type == 1:
                    temp_desc = self.construct_bin_desc_along_bound_multiscale([u_min, u_max, v_min, v_max], k, self.sem1)
                elif desc_type == 2:
                    temp_desc = self.construct_bin_desc_with_order_along_bound_multiscale([u_min, u_max, v_min, v_max], k, self.sem0)
                else:
                    raise NotImplementedError(f"desc type error")

                self.sem1_obj_patch.append([u_min, u_max, v_min, v_max, k, temp_desc])

        objs_info = []
        for obj in self.sem1_obj_patch:
            # logger.info(f"sem1 label {obj[-2]} got desc {obj[-1]}")
            objs_info.append(obj[-2])
        logger.info(f"Achieve object area:\n{objs_info} for sem1")

        return self.sem0_obj_patch, self.sem1_obj_patch

    def construct_bin_desc_along_bound(self, area_bound, label, sem, radius=2):
        """ construct the binary descriptor along the area bound
        Returns:
            bin_desc: [lx,] np.ndarray 
        """
        sorted_labels = sorted(self.label_list)
        l = len(sorted_labels)
        bin_desc = np.zeros((l))
        desc_dict = {}

        u_min, u_max, v_min, v_max = area_bound
        
        for i in range(u_min, u_max, radius):
            temp_list0 = self._get_label(sem, [i, v_min], label, radius-1)
            temp_list1 = self._get_label(sem, [i, v_max], label, radius-1)
            for label0, label1 in zip(temp_list0, temp_list1):
                if label0 not in desc_dict.keys():
                    desc_dict.update({label0:1})
                else:
                    desc_dict[label0] += 1
                if label1 not in desc_dict.keys():
                    desc_dict.update({label1:1})
                else:
                    desc_dict[label1] += 1
                
        for i in range(v_min, v_max, radius):
            temp_list0 = self._get_label(sem, [u_min, i], label, radius-1)
            temp_list1 = self._get_label(sem, [u_max,i], label, radius-1)
            for label0, label1 in zip(temp_list0, temp_list1):
                if label0 not in desc_dict.keys():
                    desc_dict.update({label0:1})
                else:
                    desc_dict[label0] += 1
                if label1 not in desc_dict.keys():
                    desc_dict.update({label1:1})
                else:
                    desc_dict[label1] += 1
        
        for k in desc_dict.keys():
            if desc_dict[k] > self.small_label_filted_thd_on_bound and k in sorted_labels:
                bin_desc[sorted_labels.index(k)] = 1

        return bin_desc

    def construct_bin_desc_along_bound_multiscale(self, area_bound, label, sem, radius=2, ms_list=[1.2, 1.4, 1.6]):
        """ get desc with multi scale: expand the area radius
           -----------
           | ------  |
           | |    |  |
           | |    |  |
           | ------  |
           -----------
        Returns:
            desc: [0or1, ...] list
        """
        desc_final_list = np.zeros((self.label_size)) == 1
        W, H = self.size

        u_min, u_max, v_min, v_max = area_bound

        raw_u_r = (u_max - u_min) // 2
        raw_v_r = (v_max - v_min) // 2

        raw_u_center = (u_max + u_min) // 2
        raw_v_center = (v_max + v_min) // 2

        for scale in ms_list:
            ms_u_r = raw_u_r * scale
            ms_v_r = raw_v_r * scale
            u_max_ms = min(int(raw_u_center + ms_u_r), W-radius)
            u_min_ms = max(int(raw_u_center - ms_u_r), 0)
            v_max_ms = min(int(raw_v_center + ms_v_r), H-radius)
            v_min_ms = max(int(raw_v_center - ms_v_r), 0)

            # label of bound
            bound_label = [1,1,1,1]
            if v_min_ms == v_min:
                bound_label[0] = 0
            if u_max_ms == u_max:
                bound_label[1] = 0
            if v_max_ms == v_max:
                bound_label[2] = 0
            if u_min_ms == u_min:
                bound_label[3] = 0
            
            temp_desc = self.construct_bin_desc_along_bound([u_min_ms, u_max_ms, v_min_ms, v_max_ms], label, sem, radius) == 1
            # logger.info(f"construct {len(desc_final_list)} length desc for label {label} with scale {scale}")
            desc_final_list = desc_final_list | temp_desc


        # logger.info(f"construct {len(desc_final_list)} length desc for label {label} Finally:\n {desc_final_list}")

        return (desc_final_list * 1).tolist()

    def construct_bin_desc_with_order_along_bound(self, area, label, bound_label, sem, radius=2):
        """ construct desc with order
        Args:
            bound_label: [1,1,1,1] 1 is needed to be stastic -- up - right - down - left
        """
        W, H = self.size
        sorted_labels = sorted(self.label_list)
        l = len(sorted_labels)
        bin_desc = np.zeros((l*4))

        u_min, u_max, v_min, v_max = area

        # up
        if bound_label[0] == 1:
            offset = 0
            temp_v_s = max(0, v_min-radius)
            temp_v_e = min(H, v_min+radius)
            # bound_patch = sem[u_min:u_max, temp_v_s:temp_v_e ]
            bound_patch = sem[temp_v_s:temp_v_e, u_min:u_max]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound:
                    bin_desc[self.label_list.index(k)+offset] = 1
        elif bound_label[0] == 0:
            offset = 0
            temp_v_s = max(0, v_min-radius)
            temp_v_e = min(H, v_min+radius)
            # bound_patch = sem[u_min:u_max, temp_v_s:temp_v_e ]
            bound_patch = sem[temp_v_s:temp_v_e, u_min:u_max]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)
            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound and k != label:
                    bin_desc[self.label_list.index(k)+offset] = 1

        # right
        if bound_label[1] == 1:
            offset = l
            temp_u_s = max(0, u_max-radius)
            temp_u_e = min(W, u_max+radius)
            # bound_patch = sem[temp_u_s:temp_u_e, v_min:v_max]
            bound_patch = sem[v_min:v_max, temp_u_s:temp_u_e]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound:
                    bin_desc[self.label_list.index(k)+offset] = 1
        if bound_label[1] == 0:
            offset = l
            temp_u_s = max(0, u_max-radius)
            temp_u_e = min(W, u_max+radius)
            # bound_patch = sem[temp_u_s:temp_u_e, v_min:v_max]
            bound_patch = sem[v_min:v_max, temp_u_s:temp_u_e]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound and k != label:
                    bin_desc[self.label_list.index(k)+offset] = 1

        # down
        if bound_label[2] == 1:
            offset = l*2
            temp_v_s = max(0, v_max-radius)
            temp_v_e = min(H, v_max+radius)
            # bound_patch = sem[u_min:u_max, temp_v_s:temp_v_e]
            bound_patch = sem[temp_v_s:temp_v_e, u_min:u_max]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound:
                    bin_desc[self.label_list.index(k)+offset] = 1
        if bound_label[2] == 0:
            offset = l*2
            temp_v_s = max(0, v_max-radius)
            temp_v_e = min(H, v_max+radius)
            # bound_patch = sem[u_min:u_max, temp_v_s:temp_v_e]
            bound_patch = sem[temp_v_s:temp_v_e, u_min:u_max]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound and k != label:
                    bin_desc[self.label_list.index(k)+offset] = 1

        # left  
        if bound_label[3] == 1:
            offset = l*3
            temp_u_s = max(0, u_min-radius)
            temp_u_e = min(W, u_min+radius)
            # bound_patch = sem[temp_u_s:temp_u_e, v_min:v_max]
            bound_patch = sem[v_min:v_max, temp_u_s:temp_u_e]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound:
                    bin_desc[self.label_list.index(k)+offset] = 1
        if bound_label[3] == 0:
            offset = l*3
            temp_u_s = max(0, u_min-radius)
            temp_u_e = min(W, u_min+radius)
            # bound_patch = sem[temp_u_s:temp_u_e, v_min:v_max]
            bound_patch = sem[v_min:v_max, temp_u_s:temp_u_e]
            temp_patch = np.squeeze(bound_patch.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            for k in temp_stas_dict.keys():
                if k in self.label_list and temp_stas_dict[k] > self.small_label_filted_thd_on_bound and k != label:
                    bin_desc[self.label_list.index(k)+offset] = 1

        return bin_desc

    
    def construct_bin_desc_with_order_along_bound_multiscale(self, area, label, sem, radius=2, ms_list=[1, 1.6, 2.2]):
        """ get desc with multi scale
           -----------
           | ------  |
           | |    |  |
           | |    |  |
           | ------  |
           -----------
        """

        desc_final_list = np.zeros((self.label_size*4)) == 1
        W, H = self.size

        u_min, u_max, v_min, v_max = area

        raw_u_r = (u_max - u_min) // 2
        raw_v_r = (v_max - v_min) // 2

        raw_u_center = (u_max + u_min) // 2
        raw_v_center = (v_max + v_min) // 2

        for scale in ms_list:
            ms_u_r = raw_u_r * scale
            ms_v_r = raw_v_r * scale
            u_max_ms = min(int(raw_u_center + ms_u_r), W-radius)
            u_min_ms = max(int(raw_u_center - ms_u_r), 0)
            v_max_ms = min(int(raw_v_center + ms_v_r), H-radius)
            v_min_ms = max(int(raw_v_center - ms_v_r), 0)

            # label of bound
            bound_label = [1,1,1,1]
            if v_min_ms == v_min:
                bound_label[0] = 0
            if u_max_ms == u_max:
                bound_label[1] = 0
            if v_max_ms == v_max:
                bound_label[2] = 0
            if u_min_ms == u_min:
                bound_label[3] = 0
                
            temp_desc = self.construct_bin_desc_with_order_along_bound([u_min_ms, u_max_ms, v_min_ms, v_max_ms], label, bound_label, sem, radius) == 1
            # logger.info(f"construct {len(desc_final_list)} length desc for label {label} with scale {scale}")
            desc_final_list = desc_final_list | temp_desc

        # logger.info(f"construct {len(desc_final_list)} length desc for label {label} Finally:\n {desc_final_list}")

        return (desc_final_list * 1).tolist()

    def _get_label(self, sem, loc, label, radius=2):
        """ achieve label in sem[loc] -- a local window
        Args:
            loc: [u, v]
        """
        W, H = self.size
        u, v = loc
        label_list = []

        for i in range(-radius, radius):
            u_temp = min(max(0, u+i), W-1)
            v_temp = min(max(0, v+i), H-1)
            label_temp = sem[v_temp, u_temp]
            if label_temp != label and label_temp not in label_list:
            # if label_temp not in label_list:
                label_list.append(label_temp)

        return label_list

    def _compare_obj_desc_bin(self, desc0, desc1):
        """ calc the hamming distance of bin descs
        Args:
            desc0: [Nx1] ndarray
        Returns:
            hamming loss
        """
        h_loss = hamming_loss(desc0, desc1)
        return h_loss

    def combine_same_label_obj_patch(self):
        """ TODO the combination of desc need fix - DONE
            combine the object patch with the same label
        Args:
            self.semx_obj_patch: [[u_min, u_max, v_min, v_max, label, desc], ]
        Returns:
            self.semx_obj_patch
        """
        # sem0
        label_num_dict = {}
        label_patch_dict = {}
        combined_obj_patch = []

        # stastic label-num
        for objP in self.sem0_obj_patch:
            temp_label = objP[-2]
            if temp_label not in label_num_dict.keys():
                label_num_dict.update({temp_label:1})
            else:
                label_num_dict[temp_label] += 1
            if temp_label not in label_patch_dict.keys():
                label_patch_dict.update({temp_label: [objP]})
            else:
                label_patch_dict[temp_label].append(objP)
        
        # check label to combine
        for label_k in label_num_dict.keys():
            if label_num_dict[label_k] == 1:
                combined_obj_patch.append(label_patch_dict[label_k][0])
            else:
                # check desc and combine the patches with same desc 
                temp_need_combine_lists = []

                while len(label_patch_dict[label_k]) > 0:
                    if self.desc_type == 0:
                        ref_desc = sorted(label_patch_dict[label_k][0][-1])
                    elif self.desc_type == 1 or self.desc_type == 2:
                        ref_desc = label_patch_dict[label_k][0][-1]

                    if len(label_patch_dict[label_k]) == 1:
                        combined_obj_patch.append(label_patch_dict[label_k].pop(0))
                    else:
                        temp_need_comb_list = []
                        pop_idx = []
                        for i, ps in enumerate(label_patch_dict[label_k]):
                            if i == 0: continue

                            if self.desc_type == 0:
                                tar_desc = sorted(ps[-1])
                            elif self.desc_type == 1 or self.desc_type == 2:
                                tar_desc = ps[-1]
                                
                            if tar_desc == ref_desc:
                                temp_need_comb_list.append(ps)
                                pop_idx.append(i)
                        # pop
                        for i,idx in enumerate(pop_idx):
                            label_patch_dict[label_k].pop(idx-i)
                        temp_need_comb_list.append(label_patch_dict[label_k].pop(0))
                        temp_need_combine_lists.append(temp_need_comb_list)

                logger.info(f"Wanna combine {label_num_dict[label_k]} patches for label {label_k}")
                for list_ in temp_need_combine_lists:
                    combined_list = []
                    combined_list = self.combine_single_label_patches(list_, self.sem0)
                    combined_obj_patch+=combined_list

        self.sem0_obj_patch = combined_obj_patch
        
        # sem1
        label_num_dict = {}
        label_patch_dict = {}
        combined_obj_patch = []

        for objP in self.sem1_obj_patch:
            temp_label = objP[-2]
            if temp_label not in label_num_dict.keys():
                label_num_dict.update({temp_label:1})
            else:
                label_num_dict[temp_label] += 1
            if temp_label not in label_patch_dict.keys():
                label_patch_dict.update({temp_label: [objP]})
            else:
                label_patch_dict[temp_label].append(objP)
        
        for label_k in label_num_dict.keys():
            if label_num_dict[label_k] == 1:
                combined_obj_patch.append(label_patch_dict[label_k][0])
            else:
                # TODO if the desc is same, than combine - DONE
                # check desc and combine the patches with same desc 
                temp_need_combine_lists = []

                while len(label_patch_dict[label_k]) > 0:
                    ref_desc = sorted(label_patch_dict[label_k][0][-1])
                    if len(label_patch_dict[label_k]) == 1:
                        combined_obj_patch.append(label_patch_dict[label_k].pop(0))
                    else:
                        temp_need_comb_list = []
                        pop_idx = []
                        for i, ps in enumerate(label_patch_dict[label_k]):
                            if i == 0: continue
                            tar_desc = sorted(ps[-1])
                            if tar_desc == ref_desc:
                                temp_need_comb_list.append(ps)
                                pop_idx.append(i)
                        # pop
                        for i,idx in enumerate(pop_idx):
                            label_patch_dict[label_k].pop(idx-i)
                        temp_need_comb_list.append(label_patch_dict[label_k].pop(0))
                        temp_need_combine_lists.append(temp_need_comb_list)

                logger.info(f"Wanna combine {label_num_dict[label_k]} patches for label {label_k}")
                for list_ in temp_need_combine_lists:
                    combined_list = []
                    combined_list = self.combine_single_label_patches(list_, self.sem1)
                    combined_obj_patch+=combined_list

        self.sem1_obj_patch = combined_obj_patch

        return self.sem0_obj_patch, self.sem1_obj_patch

    def combine_single_label_patches(self, patches, sem):
        """
        workflow:
            1. calc centers 
            2. first as main role and stastic center dist
            3. sort and combine
            4. pop and push
        Args:
            patches: [[u_min, u_max, v_min, v_max, label, desc], ...]
        Returns:
            combined patches: [[u_min, u_max, v_min, v_max, label, desc], ...]
        """
        combined_patches = patches[:]
        cannot_combine_patches = []
        logger.info(f"got {len(combined_patches)} patches to combinne")

        while (len(combined_patches)>1):
            centers = []
            centers = [[(x[0]+x[1])/2 , (x[2]+x[3])/2] for x in combined_patches]
            dists = []
            for i in range(1, len(centers)):
                dists.append(calc_euc_dist_2d(centers[0], centers[i]))
                logger.info(f"got dist = {dists}")
            dists_sorted = sorted(dists)
            if dists_sorted[0] > self.combined_obj_dist_thd:
                cannot_combine_patches.append(combined_patches.pop(0))
            else:
                close_idx = dists.index(dists_sorted[0])
                combined_patch_temp = self.combine_two_patches(combined_patches.pop(0), combined_patches.pop(close_idx), sem)
                combined_patches.insert(0, combined_patch_temp)
        
        logger.info(f"After combination got {len(combined_patches)} + {len(cannot_combine_patches)} patches")

        return combined_patches + cannot_combine_patches

    def combine_two_patches(self, patch0, patch1, sem):
        """ fro object area combine
        """
        rt_patch = []
        rt_patch = [min(patch0[0], patch1[0]), max(patch0[1], patch1[1]), min(patch0[2], patch1[2]), max(patch0[3], patch1[3])]

        assert patch0[4] == patch1[4]
        rt_patch.append(patch0[4])

        # reconstruct desc
        if self.desc_type == 0:
            logger.warning(f"Not valid desc type = {self.desc_type}")
            raise NotImplementedError()
        elif self.desc_type == 1:
            desc = self.construct_bin_desc_along_bound_multiscale(rt_patch[:4], rt_patch[4], sem)
        elif self.desc_type == 2:
            desc = self.construct_bin_desc_with_order_along_bound_multiscale(rt_patch[:4], rt_patch[4], self.sem0)
        else:
            raise NotImplementedError()

        rt_patch.append(desc)

        return rt_patch
    
    def _find_match_obj_leave_multi_candi(self, obj_area, candi_objs, single_desc_thd=0.8, desc_type=0, sizeRatioThd=4, multi_candi_dist_thd=0.1):
        """ find single match obj 
        Args:
            obj_area: [u_min, u_max, v_min, v_max, label, desc]
            candi_objs: [objp0, objp,...]
            desc_type: 
                0 - set desc
                1 - bin desc
            sizeRatioThd: big/small size threshold
        Returns:
            Matched Flag: true-found
            matched info: [matched_area1]
            matched area idx: matched_idx
            multi_candi_list = [[possible_area_idx]s]
        """
        label_src = obj_area[-2]
        desc_src = obj_area[-1]
        same_label_patch_idx = []
        multi_candi_list = []

        for i, objP in enumerate(candi_objs):
            temp_label = objP[-2]
            if temp_label == label_src:
                same_label_patch_idx.append(i)
        
        if len(same_label_patch_idx) == 0:
            logger.info(f"label {label_src} has no candis")
            return False, [], -1, []
        elif len(same_label_patch_idx) == 1:
            logger.info(f"label {label_src} has one candis")
            desc_dst = candi_objs[same_label_patch_idx[0]][-1]

            if desc_type == 0:
                raise NotImplementedError(f"Error desc type: {desc_type}")
            elif desc_type == 1 or desc_type == 2:
                same_ratio = 1 - self._compare_obj_desc_bin(desc_src, desc_dst)
            else:
                raise NotImplementedError(f"Error desc type: {desc_type}")
            logger.info(f"label same ratio={same_ratio}")
            if same_ratio > single_desc_thd:
                src_size = (obj_area[1]-obj_area[0])*(obj_area[3]-obj_area[2])
                dst_size = (candi_objs[same_label_patch_idx[0]][1] - candi_objs[same_label_patch_idx[0]][0])*(candi_objs[same_label_patch_idx[0]][3]-candi_objs[same_label_patch_idx[0]][2])
                sizeRatio = max(src_size, dst_size)/(min(src_size, dst_size)+1e-5)
                if sizeRatio > sizeRatioThd: 
                    logger.info(f"size ratio out: {sizeRatio} > {sizeRatioThd}")
                    return False, [], -1, []
                return True, candi_objs[same_label_patch_idx[0]][:4], same_label_patch_idx[0], []
            else:
                logger.info(f"obj same ratio: {same_ratio} < thd: {single_desc_thd}")
                return False, [], -1, []
        elif len(same_label_patch_idx) > 1:
            logger.info(f"label {label_src} has {len(same_label_patch_idx)} candis")

            for idx in same_label_patch_idx:
                desc_dst = candi_objs[idx][-1]
                src_size = (obj_area[1]-obj_area[0])*(obj_area[3]-obj_area[2])
                dst_size = (candi_objs[idx][1] - candi_objs[idx][0])*(candi_objs[idx][3]-candi_objs[idx][2])
                sizeRatio = max(src_size, dst_size)/(min(src_size, dst_size)+1e-5)
                if sizeRatio > sizeRatioThd: 
                    logger.info(f"size ratio out: {sizeRatio} > {sizeRatioThd}")
                    continue

                multi_candi_list.append(idx)
            
            if len(multi_candi_list) > 0:
                return False, [], -1, multi_candi_list
            else:
                return False, [], -1, []
        else:
            return False, [], -1, []
    
    def _find_match_obj(self, obj_area, candi_objs, single_desc_thd=0.8, desc_type=0, sizeRatioThd=4, multi_candi_dist_thd=0.1):
        """ find single match obj 
        Args:
            obj_area: [u_min, u_max, v_min, v_max, label, desc]
            candi_objs: [objp0, objp,...]
            desc_type: 
                0 - set desc
                1 - bin desc
            sizeRatioThd: big/small size threshold
        Returns:
            Matched Flag: true-found
            matched info: [matched_area1]
            matched area idx: matched_idx
            multi_candi_list = [[possible_area_idx]s]
        """
        label_src = obj_area[-2]
        desc_src = obj_area[-1]
        same_label_patch_idx = []
        multi_candi_list = []

        for i, objP in enumerate(candi_objs):
            temp_label = objP[-2]
            if temp_label == label_src:
                same_label_patch_idx.append(i)
        
        if len(same_label_patch_idx) == 0:
            # logger.info(f"label {label_src} has no candis")
            return False, [], -1, []
        elif len(same_label_patch_idx) == 1:
            # logger.info(f"label {label_src} has one candis")
            desc_dst = candi_objs[same_label_patch_idx[0]][-1]

            if desc_type == 0:
                raise NotImplementedError(f"Error desc type: {desc_type}")
            elif desc_type == 1 or desc_type == 2:
                same_ratio = 1 - self._compare_obj_desc_bin(desc_src, desc_dst)
            else:
                raise NotImplementedError(f"Error desc type: {desc_type}")
            # logger.info(f"label same ratio={same_ratio}")
            if same_ratio > single_desc_thd:
                src_size = (obj_area[1]-obj_area[0])*(obj_area[3]-obj_area[2])
                dst_size = (candi_objs[same_label_patch_idx[0]][1] - candi_objs[same_label_patch_idx[0]][0])*(candi_objs[same_label_patch_idx[0]][3]-candi_objs[same_label_patch_idx[0]][2])
                sizeRatio = max(src_size, dst_size)/(min(src_size, dst_size)+1e-5)
                if sizeRatio > sizeRatioThd: 
                    logger.info(f"size ratio out: {sizeRatio} > {sizeRatioThd}")
                    return False, [], -1, []
                return True, candi_objs[same_label_patch_idx[0]][:4], same_label_patch_idx[0], []
            else:
                # logger.info(f"obj same ratio: {same_ratio} < thd: {single_desc_thd}")
                return False, [], -1, []
        elif len(same_label_patch_idx) > 1:
            # logger.info(f"label {label_src} has {len(same_label_patch_idx)} candis")
            max_ratio = 0
            max_ratio_idx = 0
            same_ratio_dist = {}
            for idx in same_label_patch_idx:
                desc_dst = candi_objs[idx][-1]
                src_size = (obj_area[1]-obj_area[0])*(obj_area[3]-obj_area[2])
                dst_size = (candi_objs[idx][1] - candi_objs[idx][0])*(candi_objs[idx][3]-candi_objs[idx][2])
                sizeRatio = max(src_size, dst_size)/(min(src_size, dst_size)+1e-5)
                if sizeRatio > sizeRatioThd: 
                    logger.info(f"size ratio out: {sizeRatio} > {sizeRatioThd}")
                    continue

                if desc_type == 0:
                    raise NotImplementedError(f"Error desc type: {desc_type}")
                elif desc_type == 1 or desc_type == 2:
                    same_ratio = 1 - self._compare_obj_desc_bin(desc_src, desc_dst)
                else:
                    raise NotImplementedError(f"Error desc type: {desc_type}")
                
                same_ratio_dist.update({idx:same_ratio})

                if same_ratio > max_ratio:
                    max_ratio = same_ratio
                    max_ratio_idx = idx

            for i in same_ratio_dist.keys():
                if same_ratio_dist[i] >= max_ratio - multi_candi_dist_thd:
                    logger.info(f"doubted candi catch with same ratio dist = {same_ratio_dist[i]}")
                    multi_candi_list.append(i)
            logger.info(f"for label {label_src}, {len(multi_candi_list)} possible candi.s exist")

            # logger.info(f"max label same ratio = {max_ratio}")
            
            if max_ratio > single_desc_thd:
                return True, candi_objs[max_ratio_idx][:4], max_ratio_idx, multi_candi_list
            else:
                logger.info(f"obj max same ratio: {max_ratio} < thd: {single_desc_thd}")
                return False, [], -1, multi_candi_list
        else:
            return False, [], -1, []

    """Main Func for Object Area Match********************************
    """
    def match_object_patch(self):
        """ Main Flow of Object Area Matching
            based object found in ins0 to find the corresponding object in ins1
            Done_TODO debug 0131_00 52,92 

        Args:
            self.sem0_obj_patch -match to-> self.sem1_obj_patch
            [[u_min, u_max, v_min, v_max, k, desc], ...]

        Returns:
            self.matched_obj_label: [label0, ...]
            self.matched_obj_patch0: [[matched_patch in sem0: u_min, u_max, v_min, v_max]], [...], ...]
            self.matched_obj_patch1: [[matched_patch in sem1]], [...], ...]
            doubt_obj_match_area0/1: {
                label: [[area]s]
            }
        """
        self.matched_obj_label = []
        self.matched_obj_patch0 = []
        self.matched_obj_patch1 = []

        self.get_sem_connected()
        self.clean_the_connect_dict()
        self.find_obj_patch(desc_type=self.desc_type)

        # combination
        self.combine_same_label_obj_patch()

        # dual-direction match

        # doubt match pair
        doubt_obj_match_area0 = {} # {label:[area_idxs]}
        doubt_obj_match_area1 = {} # {label:[area_idxs]}

        temp_matched_01_dict0 = {} # {label:[[idx0,idx1]s]
        for i, objP in enumerate(self.sem0_obj_patch):
            temp_match1_idx = -1
            src_area = objP[:4]
            temp_label = objP[-2]
            Mflag, dst_area, temp_match1_idx, multi_candi_list01 = self._find_match_obj(objP, self.sem1_obj_patch, single_desc_thd=self.obj_desc_match_thd, desc_type=self.desc_type)
            
            # save doubt area
            if len(multi_candi_list01) > 1:
                if temp_label not in doubt_obj_match_area0.keys():
                    doubt_obj_match_area0.update({temp_label:[i]})
                else:
                    if i not in doubt_obj_match_area0[temp_label]:
                        doubt_obj_match_area0[temp_label].append(i)
                for idx_doubt in multi_candi_list01:
                    if temp_label not in doubt_obj_match_area1.keys():
                        doubt_obj_match_area1.update({temp_label:[idx_doubt]})
                    else:
                        if idx_doubt not in doubt_obj_match_area1[temp_label]:
                            doubt_obj_match_area1[temp_label].append(idx_doubt)
                logger.info(f"achieve {len(multi_candi_list01)} doubted candis for label {temp_label} in area0 to area1")
                continue # leave doubted area to GeoAM
                
            if not Mflag: 
                logger.info(f"match not found for label {temp_label} in sem0 to sem1")
                continue
            else:
                logger.info(f"match found for label {temp_label} in sem0 to sem1")
                if temp_label in temp_matched_01_dict0.keys():
                    temp_matched_01_dict0[temp_label].append([i, temp_match1_idx])
                else:
                    temp_matched_01_dict0.update({temp_label:[[i, temp_match1_idx]]})

        temp_matched_01_dict1 = {} # {label:[[idx0,idx1]s]
        for i, objP in enumerate(self.sem1_obj_patch):
            temp_match0_idx = -1
            src_area = objP[:4]
            temp_label = objP[-2]
            Mflag, dst_area, temp_match0_idx, multi_candi_list10 = self._find_match_obj(objP, self.sem0_obj_patch, single_desc_thd=self.obj_desc_match_thd, desc_type=self.desc_type)

            # save doubt area
            if len(multi_candi_list10) > 1:
                if temp_label not in doubt_obj_match_area1.keys():
                    doubt_obj_match_area1.update({temp_label:[i]})
                else:
                    if i not in doubt_obj_match_area1[temp_label]:
                        doubt_obj_match_area1[temp_label].append(i)
                for idx_doubt in multi_candi_list10:
                    if temp_label not in doubt_obj_match_area0.keys():
                        doubt_obj_match_area0.update({temp_label:[idx_doubt]})
                    else:
                        if idx_doubt not in doubt_obj_match_area0[temp_label]:
                            doubt_obj_match_area0[temp_label].append(idx_doubt)
                logger.info(f"achieve {len(multi_candi_list10)} doubted candis for label {temp_label} in area1 to area0")
                continue
            
            if not Mflag: 
                logger.info(f"match not found for label {temp_label} in sem1 to sem0")
                continue
            else:
                logger.info(f"match found for label {temp_label} in sem1 to sem0")
                if temp_label in temp_matched_01_dict1.keys():
                    temp_matched_01_dict1[temp_label].append([temp_match0_idx, i])
                else:
                    temp_matched_01_dict1.update({temp_label:[[temp_match0_idx, i]]})
        
        logger.info(f"get dual-direct match dict: ")
        for k0 in temp_matched_01_dict0.keys():
            logger.info(f"match 0->1 get label {k0} with matches: {temp_matched_01_dict0[k0]}")
        for k1 in temp_matched_01_dict1.keys():
            logger.info(f"match 1->0 get label {k1} with matches: {temp_matched_01_dict1[k1]}")

        for k0 in temp_matched_01_dict0.keys():
            if k0 not in temp_matched_01_dict1.keys():
                continue
            for matches01 in temp_matched_01_dict0[k0]:
                if matches01 in temp_matched_01_dict1[k0]:
                    self.matched_obj_label.append(k0)
                    self.matched_obj_patch0.append(self.sem0_obj_patch[matches01[0]][:4])
                    self.matched_obj_patch1.append(self.sem1_obj_patch[matches01[1]][:4])
        
        
        logger.info(f"Found {len(self.matched_obj_label)} matched object areas.")
        
        for label_d in doubt_obj_match_area0.keys():
            logger.info(f"for label {label_d}, achieve doubt area matches:\n img0: {doubt_obj_match_area0[label_d]} \n img1: {doubt_obj_match_area1[label_d]}")


        return doubt_obj_match_area0, doubt_obj_match_area1

    def achieve_obj_match_scale(self):
        """
        Args:
            self.matched_obj_patch0/1
        """
        logger.info(f"calc obj matched patch scale from {len(self.matched_obj_patch0)} matches")
        W, H = self.size

        self.obj_centers0 = []
        self.obj_centers1 = []
        self.obj_scale0 = []
        self.obj_scale1 = []

        for i, area0 in enumerate(self.matched_obj_patch0):
            u_min0, u_max0, v_min0, v_max0 = area0
            u_min1, u_max1, v_min1, v_max1 = self.matched_obj_patch1[i]
            if u_min0 <= 10 or u_min1 <= 10 or v_min0 <= 10 or v_min1 <= 10: continue
            if u_max0 >= W-10 or u_max1 >= W-10 or v_max0 >= H-10 or v_max1 >= H-10: continue
            W_len0 = u_max0 - u_min0
            H_len0 = v_max0 - v_min0
            avg_len0 = (W_len0 + H_len0) / 2
            W_len1 = u_max1 - u_min1
            H_len1 = v_max1 - v_min1
            avg_len1 = (W_len1 + H_len1) / 2

            self.obj_centers0.append([(u_max0+u_min0)/2, (v_max0+v_min0)/2])
            self.obj_centers1.append([(u_max1+u_min1)/2, (v_max1+v_min1)/2])

            # only get bigger scale
            if avg_len0 > avg_len1*1.2:
                self.obj_scale1.append(1.0)
                W_ratio = W_len0 / W_len1
                H_ratio = H_len0 / H_len1
                ratio_f = max(W_ratio, H_ratio)
                self.obj_scale0.append(ratio_f)
            elif avg_len1 > avg_len0*1.2:
                self.obj_scale0.append(1.0)
                W_ratio = W_len1 / W_len0
                H_ratio = H_len1 / H_len0
                ratio_f = max(W_ratio, H_ratio)
                self.obj_scale1.append(ratio_f)
            else:
                self.obj_scale0.append(1.0)
                self.obj_scale1.append(1.0)
        
        return self.obj_scale0, self.obj_scale1

    def achieve_obj_patches(self):
        """
        Args:
            self.matched_obj_patch0&1 -> crop ori img
        Returns:
            crop_patch_matches: [[patch_img0, patch_img1], [...], ...]
            offsets: [[u0_offset, v0_offset, u1_offset, v1_offset]]
        """
        W, H = self.size
        crop_patch_matches = []
        offsets = []
        scales = []
        for patch0, patch1 in zip(self.matched_obj_patch0, self.matched_obj_patch1):
            patch0 = patch_adjust_with_square_min_limit(patch0, W, H, self.radius_thd[1])
            patch1 = patch_adjust_with_square_min_limit(patch1, W, H, self.radius_thd[1])

            # patch_img0, scales0 = img_crop_with_resize(self.color0, patch0, [self.output_patch_size, self.output_patch_size])
            # patch_img1, scales1 = img_crop_with_resize(self.color1, patch1, [self.output_patch_size, self.output_patch_size])

            patch_img0, scales0, offsets0 = img_crop_without_Diffscale(self.color0, patch0, self.output_patch_size)
            patch_img1, scales1, offsets1 = img_crop_without_Diffscale(self.color1, patch1, self.output_patch_size)
            offsets.append([offsets0[0], offsets0[1], offsets1[0], offsets1[1]])


            # assert patch_img0.shape == patch_img1.shape, f"shape0 = {patch_img0.shape} != shape1 = {patch_img1.shape}"
            scales.append([scales0[0], scales0[1], scales1[0], scales1[1]])
            logger.info(f"achieve patch with size {patch_img0.shape[0]}")
            crop_patch_matches.append([patch_img0, patch_img1])
        
        return crop_patch_matches, offsets, scales

    # search points with multi semantic labels to decide overlap area
    """Main Func for overlap achieve"""
    def get_overlap_area_match(self, window_size=100, step=4):
        """ main func to achieve semantic overlap area
            # Use obj mask
        Returns:
            self.sem0/1_overlap_area_dict
            self.sem0/1_overlap_area_desc_dict
            self.matched_overlap_label
            self.matched_overlap_area0
            self.matched_overlap_area1
        """
        # self.draw_sem()

        if len(self.matched_obj_patch0) == 0:
            logger.warning(f"find overlap without object mask!")

        # sem0
        logger.info(f"achieve sem0 overlap...")
        self.sem0_overlap_area_dict = self.sliding_win_get_overlap_area(self.sem0, self.matched_obj_patch0, window_size, step)
        self.sem0_overlap_area_dict = self.filt_overlap_area(self.sem0_overlap_area_dict, self.matched_obj_patch0)
        self.sem0_overlap_area_dict = self.combine_overlap_area(self.sem0_overlap_area_dict)
        logger.info(f"draw sem0 overlap area...")
        self.draw_overlap_area(self.sem0_overlap_area_dict, self.color0, self.name0)

        # sem1
        logger.info(f"achieve sem1 overlap...")
        self.sem1_overlap_area_dict = self.sliding_win_get_overlap_area(self.sem1, self.matched_obj_patch1, window_size, step)
        self.sem1_overlap_area_dict = self.filt_overlap_area(self.sem1_overlap_area_dict, self.matched_obj_patch1)
        self.sem1_overlap_area_dict = self.combine_overlap_area(self.sem1_overlap_area_dict)
        logger.info(f"draw sem1 overlap area...")
        self.draw_overlap_area(self.sem1_overlap_area_dict, self.color1, self.name1)

        # refinement & construct size desc
        self.refine_overlap_area()

        # match
        logger.info(f"Starting overlap matching")
        self.match_overlap_area_from_label_size_desc()
        self.draw_overlap_match_res()

    def refine_overlap_area(self):
        """ refine the location of overlap area
            # drop the repeat label_str_area
            construct the desc
        """
        logger.info(f"refine the location of overlap areas & construct desc")

        # sem0
        l_ori = len(self.sem0_overlap_area_dict.values())
        # logger.info(f"before refine sem0 get {l_ori} areas")
        new_sem0_overlap_dict = {}
        self.sem0_overlap_area_desc_dict = {}
        for area in self.sem0_overlap_area_dict.values():
            label_str, area_new, desc0 = self.refine_single_area(area[0], self.sem0)
            if len(area_new) != 4: continue
            if label_str in new_sem0_overlap_dict.keys():
                new_sem0_overlap_dict[label_str].append(area_new)
                self.sem0_overlap_area_desc_dict[label_str].append(desc0)
                continue
            else:
                new_sem0_overlap_dict.update({label_str:[area_new]})
                self.sem0_overlap_area_desc_dict.update({label_str:[desc0]})
        l_ref = len(new_sem0_overlap_dict.keys())
        # logger.info(f"after refine sem0 get {l_ref} areas")
    
        # sem1
        l_ori = len(self.sem1_overlap_area_dict.values())
        # logger.info(f"before refine sem1 get {l_ori} areas")
        new_sem1_overlap_dict = {}
        self.sem1_overlap_area_desc_dict = {}
        for area in self.sem1_overlap_area_dict.values():
            label_str, area_new, desc1 = self.refine_single_area(area[0], self.sem1)
            if len(area_new) != 4: continue

            if label_str in new_sem1_overlap_dict.keys():
                new_sem1_overlap_dict[label_str].append(area_new)
                self.sem1_overlap_area_desc_dict[label_str].append(desc1)
                continue
            else:
                new_sem1_overlap_dict.update({label_str:[area_new]})
                self.sem1_overlap_area_desc_dict.update({label_str:[desc1]})
        l_ref = len(new_sem1_overlap_dict.keys())
        # logger.info(f"after refine sem1 get {l_ref} areas")

        self.sem0_overlap_area_dict = new_sem0_overlap_dict
        self.sem1_overlap_area_dict = new_sem1_overlap_dict

        return new_sem0_overlap_dict, new_sem1_overlap_dict

    def refine_single_area(self, area, sem):
        """ refine single area in sem, by expand with self.radius_thd[0]
        Args:
            area: [u_min, u_max, v_min, v_max]
            sem
        Returns:
            label_str: inside the area
            area_new
        """
        sorted_label_list = sorted(self.label_list)
        l_desc = len(sorted_label_list)
        W, H = self.size
        u_min, u_max, v_min, v_max = area
        radius = self.radius_thd[0]
        u_center_raw = (u_min + u_max) // 2
        v_center_raw = (v_min + v_max) // 2

        if u_center_raw < radius: u_center_raw = radius
        if W - u_center_raw < radius: u_center_raw = W - radius
        if v_center_raw < radius: v_center_raw = radius
        if H - v_center_raw < radius: v_center_raw = H - radius

        u_min_new = u_center_raw - radius
        u_max_new = u_center_raw + radius

        v_min_new = v_center_raw - radius
        v_max_new = v_center_raw + radius

        sem_patch = sem[v_min_new:v_max_new, u_min_new:u_max_new]
        total_patch_size = (v_max_new - v_min_new) * (u_max_new - u_min_new)

        temp_patch = np.squeeze(sem_patch.reshape((-1,1))).tolist()
        temp_stas_dict = collections.Counter(temp_patch)

        label_list_rt = []
        for k in temp_stas_dict.keys():
            if temp_stas_dict[k] > self.small_label_filted_thd_inside_area:
                label_list_rt.append(k)
        
        if len(label_list_rt) < 3:
            return "", [], None
        else:
            label_list_rt.sort()
            desc = np.zeros((l_desc,1))
            # construct desc
            for label in label_list_rt:
                if label not in sorted_label_list:
                    continue
                desc[sorted_label_list.index(label),0] = temp_stas_dict[label] / total_patch_size
            
            label_str_rt = "_".join([str(x) for x in label_list_rt])
            return label_str_rt, [u_min_new, u_max_new, v_min_new, v_max_new], desc   

    def match_overlap_area_from_label_size_desc(self):
        """ match overlap area from size desc
        Args:
            self.sem0_overlap_area_dict
            self.sem1_overlap_area_dict
            self.sem0_overlap_area_desc_dict
            self.sem1_overlap_area_desc_dict
        Returns:
            self.matched_overlap_label 
            self.matched_overlap_area0 
            self.matched_overlap_area1 
        """
        overlap_desc_dist_thd = self.overlap_desc_dist_thd
        label_str_list = []
        matched_overlap_area0 = []
        matched_overlap_area1 = []
        count = 0

        # desc show
        # for i, k0 in enumerate(self.sem0_overlap_area_desc_dict.keys()):
        #     for j, desc in enumerate(self.sem0_overlap_area_desc_dict[k0]):
        #         logger.info(f"area {i}.{j} with label: {k0} get desc: \n{desc}")

        # for i, k1 in enumerate(self.sem1_overlap_area_desc_dict.keys()):
        #     for j, desc in enumerate(self.sem1_overlap_area_desc_dict[k1]):
        #         logger.info(f"area {i}.{j} with label: {k1} get desc: \n{desc}")
        
        # match
        for k0 in self.sem0_overlap_area_desc_dict.keys():
            for i0, desc0 in enumerate(self.sem0_overlap_area_desc_dict[k0]):
                temp_closest_k1 = None
                temp_closest_area_index = -1
                closest_dist = 1e5
                for k1 in self.sem1_overlap_area_desc_dict.keys():
                    for i, desc1 in enumerate(self.sem1_overlap_area_desc_dict[k1]):
                        temp_dist = np.linalg.norm(desc0 - desc1)
                        # logger.info(f"sem0 {k0} to sem1 {k1} get dist: {temp_dist}")
                        if temp_dist > overlap_desc_dist_thd:
                            continue
                        else:
                            if closest_dist > temp_dist:
                                closest_dist = temp_dist
                                temp_closest_k1 = k1
                                temp_closest_area_index = i
                if temp_closest_area_index != -1:
                    label_str_list.append(k0)
                    matched_overlap_area0.append(self.sem0_overlap_area_dict[k0][i0])
                    matched_overlap_area1.append(self.sem1_overlap_area_dict[temp_closest_k1][temp_closest_area_index])
                    # logger.info(f"achieve match with {k0} and {temp_closest_k1} whose dist: {closest_dist}")

        self.matched_overlap_label = label_str_list
        self.matched_overlap_area0 = matched_overlap_area0
        self.matched_overlap_area1 = matched_overlap_area1

        return label_str_list, matched_overlap_area0, matched_overlap_area1     

    def filt_overlap_area(self, overlap_area_dict, mask, WHRatio=2, areaSize=1600, maskOverlapThd=0.99):
        """ filt small and non-square area
        Args:
            overlap_area_dict = {
                label_str: [[area]s]
            }
            mask = [[area]s]
            maskOverlapThd: overlapSize / areaSize < thd -> reserve
        Returns:
            dict = {
                label_str: [[area]s]
            }
        """
        re_dict = {}

        for label in overlap_area_dict.keys():
            for area in overlap_area_dict[label]:
                u_min, u_max, v_min, v_max = area
                u_len = u_max - u_min
                v_len = v_max - v_min
                ratio = max(u_len, v_len) / (min(u_len, v_len)+1e-5)
                if ratio > WHRatio: continue
                if u_len * v_len < areaSize: continue

                size_flag = True
                for area_mask in mask:
                    size_val = self._calc_overlap_size(area, area_mask)
                    if size_val > maskOverlapThd:
                        size_flag = False
                        break
                if not size_flag: continue

                if label not in re_dict.keys():
                    re_dict.update({label:[area]})
                else:
                    re_dict[label].append(area)
        
        return re_dict

    def combine_overlap_area(self, overlap_area_dict):
        """ combine the overlap area according to distance
        Args:
            overlap_area_dict = {
                label_str: [[area]s]
            }
        """
        combine_dist = self.overlap_radius * 2
        label_str_list = []
        areas_list = []

        for k in overlap_area_dict.keys():
            for area_tmp in overlap_area_dict[k]:
                label_str_list.append(k)
                areas_list.append(area_tmp)
        
        after_combine_label_str = []
        after_combine_areas = []

        logger.info(f"Area combination...")
        
        while len(areas_list) > 1:
            assert len(areas_list) == len(label_str_list), "Alignment Error"
            centers = [[(x[0]+x[1])/2 , (x[2]+x[3])/2] for x in areas_list]
            dists = []
            for i in range(1, len(centers)):
                dists.append(calc_euc_dist_2d(centers[0], centers[i]))
            
            dist_sorted = sorted(dists)

            if dist_sorted[0] > combine_dist:
                after_combine_areas.append(areas_list.pop(0))
                after_combine_label_str.append(label_str_list.pop(0))
            else:
                closest_idx = dists.index(dist_sorted[0])

                combined_area_temp = self._combine_overlap_area(areas_list.pop(0), areas_list.pop(closest_idx))
                areas_list.insert(0, combined_area_temp)

                combined_label_str_temp = self._combine_overlap_label_str(label_str_list.pop(0), label_str_list.pop(closest_idx))
                label_str_list.insert(0, combined_label_str_temp)


        after_combine_areas += areas_list
        after_combine_label_str += label_str_list
        assert len(after_combine_areas) == len(after_combine_label_str)

        rt_dict = {}

        for i, label_str in enumerate(after_combine_label_str):
            if label_str not in rt_dict.keys():
                rt_dict.update({label_str:[after_combine_areas[i]]})
            else:
                rt_dict[label_str].append(after_combine_areas[i])
        
        return rt_dict
    
    def _combine_overlap_area(self, area0, area1):
        """ for overlap areas to combine
        Args:
            area = [u_min, u_max, v_min, v_max]
        """
        u_min0, u_max0, v_min0, v_max0 = area0
        u_min1, u_max1, v_min1, v_max1 = area1
        
        u_min_ = min(u_min0, u_min1)
        u_max_ = max(u_max0, u_max1)
        v_min_ = min(v_min0, v_min1)
        v_max_ = max(v_max0, v_max1)
        
        return [u_min_, u_max_, v_min_, v_max_]

    def _combine_overlap_label_str(self, labels0, labels1):
        """ for overlap areas to combine label str
        Args:
            label0: "xx_xx_xx_x" 
        """
        label_list = labels0.split('_')
        labels1_list = labels1.split('_')

        for l in labels1_list:
            if l not in label_list:
                label_list.append(l)
        
        label_list.sort()
        
        rt = "_".join(label_list)

        return rt

    def _calc_overlap_size(self, bbox_src, bbox_dst):
        """ overlapSize = overlap(bboxSrc, bboxDst) / bboxSrc
        """
        u_min_s, u_max_s, v_min_s, v_max_s = bbox_src
        src_size = (u_max_s - u_min_s) * (v_max_s - v_min_s)
        assert src_size > 0, "invalid size"
        u_min_d, u_max_d, v_min_d, v_max_d = bbox_dst

        u_min_f = max(u_min_s, u_min_d)
        u_max_f = min(u_max_s, u_max_d)
        v_min_f = max(v_min_s, v_min_d)
        v_max_f = min(v_max_s, v_max_d)

        u_len = u_max_f - u_min_f
        v_len = v_max_f - v_min_f

        if u_len <= 0 or v_len <= 0:
            return 0
        else:
            return (u_len * v_len) / (src_size + 1e-5)

    def match_overlap_area_pyramid_version(self):
        """ achieve overlap area & dual match
        Returns:
            self.matched_overlap_label 
            self.matched_overlap_area0 
            self.matched_overlap_area1 
        """
        # achieve areas & descs
        self.sem0_overlap_areas, self.sem0_overlap_area_descs = self.achieve_overlap_area_pyramid_main(self.sem0, self.color0, self.obj_centers0, self.obj_scale0, pyramid_ratio=1/8, name="pyramid_sem0_overlap")
        self.sem1_overlap_areas, self.sem1_overlap_area_descs = self.achieve_overlap_area_pyramid_main(self.sem1, self.color1, self.obj_centers1, self.obj_scale1, pyramid_ratio=1/8, name="pyramid_sem1_overlap")

        # dual match
        self.matched_overlap_label = []
        self.matched_overlap_area0 = []
        self.matched_overlap_area1 = []
        overlap_desc_dist_thd = self.overlap_desc_dist_thd
        overlap0_len = len(self.sem0_overlap_areas)

        # 0 to 1
        temp_021_matched_idx0 = [-1]*overlap0_len
        
        for idx0, desc0 in enumerate(self.sem0_overlap_area_descs):
            min_dist = 1e5
            min_dist_idx = -1
            for i, desc1 in enumerate(self.sem1_overlap_area_descs):
                temp_dist = np.linalg.norm(desc0 - desc1)
                # logger.info(f"0->1:{idx0} to {i} dist = {temp_dist}")
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_dist_idx = i
            
            if min_dist > self.overlap_desc_dist_thd:
                continue

            temp_021_matched_idx0[idx0] = min_dist_idx
            # logger.info(f"0-{idx0} matched to 1-{min_dist_idx} with dist = {min_dist}")
        
        # 1 to 0
        temp_021_matched_idx1 = [-1]*overlap0_len

        for idx1, desc1 in enumerate(self.sem1_overlap_area_descs):
            min_dist = 1e5
            min_dist_idx = -1
            for idx0, desc0 in enumerate(self.sem0_overlap_area_descs):
                temp_dist = np.linalg.norm(desc0 - desc1)
                # logger.info(f"1->0:{idx0} to {idx1} dist = {temp_dist}")
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_dist_idx = idx0
            
            if min_dist > self.overlap_desc_dist_thd:
                continue
            
            temp_021_matched_idx1[min_dist_idx] = idx1
            # logger.info(f"1-{idx1} matched to 0-{min_dist_idx} with dist = {min_dist}")

        # logger.info(f"achieve match idx021-0: {temp_021_matched_idx0} \nmatched idx021-1: {temp_021_matched_idx1}")

        for idx0, idx1 in enumerate(temp_021_matched_idx0):
            if idx1 != temp_021_matched_idx1[idx0] or idx1 == -1: continue
            self.matched_overlap_area0.append(self.sem0_overlap_areas[idx0])
            self.matched_overlap_area1.append(self.sem1_overlap_areas[idx1])

        logger.info(f"achieve {len(self.matched_overlap_area0)} matched overlap areas")

        return self.matched_overlap_area0, self.matched_overlap_area1

    def achieve_overlap_area_pyramid_main(self, sem, color, obj_areas, obj_scales, pyramid_ratio=1/8, name=""):
        """
        Args:
            sem: np.ndarray
            mask: [areas]
            pyramid_ratio: resize scale
        Returns:
            self.sem0/1_overlap_areas: []
            self.sem0/1_overlap_area_descs: []
        Mainflow:
            0. filter sem -> black_list_label
            1. resize to pyramid level
            2. stastic -> candis
            3. refine in ori sem to min variance 
            4. expand to area & construct desc
        """

        overlap_areas = []
        overlap_area_descs = []

        H, W = sem.shape
        pyramid_W = int(W * pyramid_ratio)
        pyramid_H = int(H * pyramid_ratio)

        # 0. filter
        filted_sem, black_list_label = self._filter_overlap_sem(sem)

        # logger.info(f"black list label is {black_list_label}")

        # # debug
        # paint_semantic_single(sem, self.out_path, "nonfilterd_sem")
        # paint_semantic_single(filted_sem, self.out_path, "filterd_sem")

        # 1. resize
        high_pyramid_layer = cv2.resize(filted_sem, (pyramid_W, pyramid_H), interpolation=cv2.INTER_NEAREST)
        # paint_semantic_single(high_pyramid_layer, self.out_path, "pyramid_sem")

        # 2. stastic
        pyramid_sem_overlap_centers, pyramid_sem_overlap_labels, pyramid_sem_overlap_vars = self._stastic_overlap_candis_list(high_pyramid_layer, black_list_label, self.overlap_radius,pyramid_ratio)
        # logger.info(f"list stastic centers: \n{pyramid_sem_overlap_centers} \n{pyramid_sem_overlap_labels}")
        
        # 3. refine in ori sem
        ori_sem_overlap_centers, ori_sem_overlap_labels = self._refine_overlap_in_ori_sem_list(filted_sem, pyramid_sem_overlap_centers, pyramid_sem_overlap_labels, pyramid_ratio, black_list_label)
        # logger.info(f"list refined = \n{ori_sem_overlap_centers} \n{ori_sem_overlap_labels}")

        # 4. expand to area by obj scale and construct desc
        overlap_areas, overlap_area_descs = self._expand_area_from_center_form_desc(filted_sem, ori_sem_overlap_centers, obj_areas, obj_scales, self.overlap_radius, black_list_label)
        logger.info(f"achieve {len(overlap_areas)} areas after expand")

        # draw
        if self.draw_verbose == 1:
            self.draw_overlap_area_list(overlap_areas, color, name)

        return overlap_areas, overlap_area_descs

    def _expand_area_from_center_form_desc(self, sem, overlap_centers, obj_centers, obj_scales, std_radius, black_list_label):
        """
        Args:
            overlap_centers
            obj_centers: self.obj_centers0/1
            obj_scales: self.obj_scales0
            std_radius: self.overlap_radius
        Returns:
            overlap_areas
            overlap_area_descs
        """
        overlap_areas = []
        overlap_area_descs = []
        W, H = self.size

        # logger.info(f"achieve {len(obj_centers)} obj areas as scale anchor:\n {obj_centers} \n{obj_scales}")

        # achieve area

        for center in overlap_centers:
            scale = 1.0
            u_c, v_c = center
            if len(obj_centers) > 0:
                dists = [math.sqrt((u_c-obj_center[0])**2 + (v_c - obj_center[1])**2) for obj_center in obj_centers]
                min_dist = min(dists)
                min_idx = dists.index(min_dist)

                if min_dist < self.same_overlap_dist:
                    
                    scale = obj_scales[min_idx]
            
            radius = std_radius * scale

            u_t, v_t = self._refine_center(u_c, v_c, radius, W, H)

            u_min = int(u_t - radius)
            u_max = int(u_t + radius)
            v_min = int(v_t - radius)
            v_max = int(v_t + radius)

            overlap_areas.append([u_min, u_max, v_min, v_max])

            # construct desc
            sub_sem = sem[v_min:v_max, u_min:u_max]
            temp_patch = np.squeeze(sub_sem.reshape((-1,1))).tolist()
            temp_stas_dict = collections.Counter(temp_patch)

            total_valid_size = (radius*2)**2
            sorted_label_list = sorted(self.label_list)
            l_total = len(sorted_label_list)
            desc_temp = np.zeros((l_total, 1))

            for label in temp_stas_dict.keys():
                if label == black_list_label: continue
                desc_temp[sorted_label_list.index(label),0] = temp_stas_dict[label] / total_valid_size
            
            overlap_area_descs.append(desc_temp)


        return overlap_areas, overlap_area_descs

    def _refine_center(self, u, v, radius, W, H):
        """ refine area centers to adjust to shape
        Returns:
            u_rt
            v_rt
        """
        if u-radius > 0 and u+radius < W:
            u_rt = u
        elif u-radius < 0 and u+radius > W:
            u_rt = u
        elif u-radius <= 0:
            u_rt = radius
        elif u+radius >= W:
            u_rt = W-radius-1

        if v-radius > 0 and v+radius < W:
            v_rt = v
        elif v-radius < 0 and v+radius > W:
            v_rt = v
        elif v-radius <= 0:
            v_rt = radius
        elif v+radius >= W:
            v_rt = W-radius-1
        
        return u_rt, v_rt

    def _refine_overlap_in_ori_sem_list(self, sem, pyramid_overlap_centers, pyramid_overlap_labels, pyramid_ratio, black_list_label):
        """
        Returns:
            ori_overlap_centers: [centers]
            ori_overlap_labels: [labels]
        """
        ori_overlap_centers = []
        ori_overlap_labels = []
        H, W = sem.shape
        ori_radius = int(1/pyramid_ratio)

        for i, [u, v] in enumerate(pyramid_overlap_centers):
            u_ori = int(u / pyramid_ratio)
            v_ori = int(v / pyramid_ratio)
            u_ori, v_ori = self._refine_center(u_ori, v_ori, ori_radius, W, H)
            u_s = max(0, u_ori-ori_radius)
            u_e = min(W, u_ori+ori_radius)
            v_s = max(0, v_ori-ori_radius)
            v_e = min(W, v_ori+ori_radius)
            label_str_ori = pyramid_overlap_labels[i]
            
            min_var = 1e5
            min_var_u = u_ori
            min_var_v = v_ori
            min_var_label = label_str_ori
            # logger.debug(f"achieve {u_ori}, {v_ori} to refine in {u_s}~{u_e}, {v_s}~{v_e}")
            for u_ in range(u_s, u_e, 2):
                for v_ in range(v_s, v_e, 2):
                    flag, label_str, var = self._stastic_single_point(sem, u_, v_, self.overlap_radius, black_list_label, 1)
                    # logger.debug(f"achieve {flag} in {u_},{v_} wiht var={var}")
                    if not flag: continue
                    if var < min_var:
                        min_var = var
                        min_var_u = u_
                        min_var_v = v_
                        min_var_label = label_str

            if min_var == 1e5: continue

            ori_overlap_centers.append([min_var_u, min_var_v])
            ori_overlap_labels.append(min_var_label)
        
        return ori_overlap_centers, ori_overlap_labels

    def _refine_overlap_in_ori_sem(self, sem, pyramid_overlap_dict, pyramid_ratio, black_list_label):
        """Refine centers in ori size sem
        Returns:
            ori_sem_overlap_dict = {
                label_str: [centers]
            }
        """
        ori_sem_overlap_dict = {}

        H, W = sem.shape
        ori_radius = int(1/pyramid_ratio)
        for label in pyramid_overlap_dict.keys():
            for [u, v] in pyramid_overlap_dict[label]:
                u_ori = int(u / pyramid_ratio)
                v_ori = int(v / pyramid_ratio)
                u_ori, v_ori = self._refine_center(u_ori, v_ori, ori_radius, W, H)
                u_s = max(0, u_ori-ori_radius)
                u_e = min(W, u_ori+ori_radius)
                v_s = max(0, v_ori-ori_radius)
                v_e = min(W, v_ori+ori_radius)
                
                min_var = 1e5
                min_var_u = u_ori
                min_var_v = v_ori
                # logger.debug(f"achieve {u_ori}, {v_ori} to refine in {u_s}~{u_e}, {v_s}~{v_e}")
                # for u_ in tqdm(range(u_s, u_e, 2), ncols=50):
                for u_ in range(u_s, u_e, 2):
                    for v_ in range(v_s, v_e, 2):
                        flag, label_str, var = self._stastic_single_point(sem, u_, v_, self.overlap_radius, black_list_label, 1)
                        # logger.debug(f"achieve {flag} in {u_},{v_} wiht var={var}")
                        if not flag: continue
                        if var < min_var:
                            min_var = var
                            min_var_u = u_
                            min_var_v = v_

                if min_var == 1e5: continue

                if label not in ori_sem_overlap_dict.keys():
                    ori_sem_overlap_dict.update({label:[[min_var_u, min_var_v]]})
                else:
                    ori_sem_overlap_dict[label].append([min_var_u, min_var_v])

        return ori_sem_overlap_dict

    def _stastic_overlap_candis_list(self, sem, black_list_label, window_radius, resize_ratio):
        """
        Args:
            sem: np.ndarray
            window_radius: overlap area radius
        Returns:
            sem_overlap_centers = [centers]
            sem_overlap_labels = [label_strs]
            sem_overlap_vars = [vars]
        """
        H, W = sem.shape
        window_radius = int(window_radius * resize_ratio)
        logger.info(f"stastic with radius = {window_radius} in {W}x{H} sem")
        sem_overlap_centers = []
        sem_overlap_labels = []
        sem_overlap_vars = []

        for u in range(window_radius, W-window_radius):
            for v in range(window_radius, H-window_radius):
                flag, label_str, variance = self._stastic_single_point(sem, u, v, window_radius, black_list_label, resize_ratio)
                if not flag: 
                    # logger.debug(f"No valid label")
                    continue

                if len(sem_overlap_centers) == 0:
                    sem_overlap_centers.append([u, v])
                    sem_overlap_labels.append(label_str)
                    sem_overlap_vars.append(variance)
                else:
                    dists = [math.sqrt((u-c[0])**2 + (v-c[1])**2) for c in sem_overlap_centers]
                    copy_centers = sem_overlap_centers[:]
                    fuse_candi_centers = [[u,v]]
                    fuse_candi_labels = [label_str]
                    fuse_candi_vars = [variance]

                    # get all fuse candis based on dist
                    for idx_dist, dist in enumerate(dists):
                        if dist < self.same_overlap_dist * resize_ratio:
                            pop_obj = copy_centers[idx_dist]
                            pop_idx = sem_overlap_centers.index(pop_obj)
                            fuse_candi_centers.append(sem_overlap_centers.pop(pop_idx))
                            fuse_candi_labels.append(sem_overlap_labels.pop(pop_idx))
                            fuse_candi_vars.append(sem_overlap_vars.pop(pop_idx))
                    
                    if len(fuse_candi_centers) == 1:
                        sem_overlap_centers.append([u, v])
                        sem_overlap_labels.append(label_str)
                        sem_overlap_vars.append(variance)
                    else:
                        # first scan based on label num
                        label_len_candi_centers = []
                        label_len_candi_labels = []
                        label_len_candi_vars = []

                        label_lens = [len(label.split('_')) for label in fuse_candi_labels]
                        max_len = max(label_lens)
                        
                        # get max len label centers
                        for i, label in enumerate(fuse_candi_labels):
                            if len(label.split("_")) == max_len:
                                label_len_candi_centers.append(fuse_candi_centers[i])
                                label_len_candi_labels.append(fuse_candi_labels[i])
                                label_len_candi_vars.append(fuse_candi_vars[i])
                        
                        if len(label_len_candi_centers) == 1:
                            sem_overlap_centers.append(label_len_candi_centers[0])
                            sem_overlap_labels.append(label_len_candi_labels[0])
                            sem_overlap_vars.append(label_len_candi_vars[0])
                        else:
                            # second scan based on variance
                            min_var = min(label_len_candi_vars)
                            min_var_idx = label_len_candi_vars.index(min_var)

                            sem_overlap_centers.append(label_len_candi_centers[min_var_idx])
                            sem_overlap_labels.append(label_len_candi_labels[min_var_idx])
                            sem_overlap_vars.append(label_len_candi_vars[min_var_idx])
                    
        return sem_overlap_centers, sem_overlap_labels, sem_overlap_vars

    def _stastic_overlap_candis(self, sem, black_list_label, window_radius, resize_ratio):
        """
        Args:
            sem: np.ndarray
            window_radius: overlap area radius
        Returns: sem_overlap_candis_dict = 
            {
                label_str : [centers]
            }
        """
        H, W = sem.shape
        window_radius = int(window_radius * resize_ratio)
        logger.info(f"stastic with radius = {window_radius} in {W}x{H} sem")
        sem_overlap_candis_dict = {}
        sem_overlap_candis_var_dict = {}

        # for u in tqdm(range(window_radius, W-window_radius), ncols=50):
        for u in range(window_radius, W-window_radius):
            for v in range(window_radius, H-window_radius):
                flag, label_str, variance = self._stastic_single_point(sem, u, v, window_radius, black_list_label, resize_ratio)
                if not flag: continue
                if label_str in sem_overlap_candis_dict.keys():
                    # logger.debug(f"{u}, {v} got {label_str} in dict")
                    fuse_center_list = [[u, v]]
                    fuse_var_list = [variance]
                    dist = [math.sqrt((u-c[0])**2 + (v-c[1])**2) for c in sem_overlap_candis_dict[label_str]]
                    # logger.debug(f"get dist = {dist}")
                    dist_relate_copy = sem_overlap_candis_dict[label_str][:]

                    # check dist
                    for i, d in enumerate(dist):
                        if d < self.same_overlap_dist * resize_ratio * 2:
                            # close center need fuse
                            pop_obj = dist_relate_copy[i]
                            # logger.debug(f"dist {d} is small to pop {pop_obj}")
                            idx_pop = sem_overlap_candis_dict[label_str].index(pop_obj)
                            fuse_center_list.append(sem_overlap_candis_dict[label_str].pop(idx_pop))
                            fuse_var_list.append(sem_overlap_candis_var_dict[label_str].pop(idx_pop))
                            # logger.debug(f"after pop dict = {sem_overlap_candis_dict[label_str]}")

                    # fuse: save the min variance center
                    if len(fuse_center_list) > 1:
                        min_var = min(fuse_var_list)
                        min_idx = fuse_var_list.index(min_var)

                        fuse_center_list_ = [fuse_center_list[min_idx]]
                        fuse_var_list_ = [fuse_var_list[min_idx]]
                    else:
                        fuse_center_list_ = fuse_center_list
                        fuse_var_list_ = fuse_var_list
                        
                    assert len(fuse_center_list_) == 1

                    sem_overlap_candis_dict[label_str] += fuse_center_list_
                    sem_overlap_candis_var_dict[label_str] += fuse_var_list_
                    # logger.debug(f"select best {fuse_center_list_} to add, then dict = {sem_overlap_candis_dict[label_str]}")

                else:
                    sem_overlap_candis_dict.update({label_str:[[u,v]]})
                    sem_overlap_candis_var_dict.update({label_str:[variance]})
        
        return sem_overlap_candis_dict
                
    def _stastic_single_point(self, sem, u, v, window_radius, black_list_label, ratio=1):
        """
        Returns:
            flag
            label_str
            variance
        """
        H, W = sem.shape
        u_s = max(0, u-window_radius)
        u_e = min(W, u+window_radius)
        v_s = max(0, v-window_radius)
        v_e = min(H, v+window_radius)

        temp_sem_list = np.squeeze(sem[v_s:v_e, u_s:u_e].reshape((-1,1))).tolist()
        temp_stas_dict = collections.Counter(temp_sem_list)

        valid_labels = []
        valid_label_nums = []

        for label in temp_stas_dict.keys():
            # not the background nor small area
            if label != black_list_label and temp_stas_dict[label] > self.small_label_filted_thd_inside_area*ratio*ratio:
                if self.datasetName == "MatterPort3D":
                    if label in self.non_obj_dict.values():
                        continue
                valid_labels.append(label)
                valid_label_nums.append(temp_stas_dict[label])
        
        if len(valid_labels) < 3:
            return False, None, None
        
        valid_labels = sorted(valid_labels)
        label_str = self._sorted_labels_list2str(valid_labels)
        valid_num_np = np.array(valid_label_nums)
        valid_num_np = self._nparray_norm(valid_num_np)
        var = np.var(valid_num_np)
        # logger.debug(f"achieve {var} with {valid_num_np}")

        return True, label_str, var

    def _nparray_norm(self, array1d):
        """
        """
        M_ = np.max(array1d) - np.min(array1d)
        rt = (array1d - np.min(array1d)) / M_
        # logger.debug(f"after norm = {rt}")
        return rt

    def _filter_overlap_sem(self, sem):
        """
        Return:
            filted_sem
            black_list_label
        """
        max_label = max(self.label_list)
        black_list_label = max_label+100
        filted_sem = sem.copy()
        filted_sem[:] = black_list_label

        for label in self.label_list:
            filted_sem[np.where(sem==label)] = label
        
        return filted_sem, black_list_label

    def sliding_win_get_overlap_area(self, sem, mask, win_size=100, step=2):
        """ sliding window to stastic the overlap area
            if the window area includes >= 3 different labels -> the center point [u,v] is added to define the overlap area
        Args:
            sem: np.ndarray
            mask: [[u_min, u_max, v_min, v_max]]
        Returns:
            overlap_area_dict = {
                sorted_labels_list2str : [[u_min, u_max, v_min, v_max], ...]
            }
            # sorted_labels_list: [0,1,12] -> str: '0_1_12'
        """
        overlap_area_dict = {}
        W, H = self.size

        sem_ = sem.copy()
        logger.info(f"get {len(mask)} masks to avoid")
        for m in mask:
            u_min, u_max, v_min, v_max = m
            sem_[v_min:v_max, u_min:u_max] = -1

        min_var = 1e5
        for u in range(0, W, step):
            for v in range(0, H, step):
                if sem_[v, u] == -1: 
                    continue
                flag, temp_label_list, temp_var = self.check_overlap_in_single_win(sem_, [u, v], win_size)
                if not flag:
                    continue
                else:
                    # see multiple semantic
                    label_str = self._sorted_labels_list2str(temp_label_list)

                    if label_str not in overlap_area_dict.keys():
                        overlap_area_dict.update({label_str:[[u, u, v, v]]})
                    else: # if already in dict, then combine them 
                        assign_idx = 1000 # impossible num
                        min_dist = 1e5
                        # check dist to assign this overlap
                        for i, bbox in enumerate(overlap_area_dict[label_str]):
                            dist = self._calc_pt_dist2bbox(bbox, [u,v])
                            if dist < self.same_overlap_dist:
                                if dist < min_dist:
                                    min_dist = dist
                                    assign_idx = i
                        # update the bbox
                        if assign_idx == 1000:
                            # add new bbox
                            overlap_area_dict[label_str].append([u, u, v, v])
                        else:
                            # update 
                            raw_bbox = overlap_area_dict[label_str][assign_idx][:]
                            overlap_area_dict[label_str][assign_idx] = self._update_bbox(raw_bbox, [u, v])

        return overlap_area_dict

    def _update_bbox(self, bbox, pt):
        """
        """
        u_min, u_max, v_min, v_max = bbox
        u, v = pt

        u_min_ = min(u_min, u)
        u_max_ = max(u_max, u)
        v_min_ = min(v_min, v)
        v_max_ = max(v_max, v)

        return [u_min_, u_max_, v_min_, v_max_]

    def _calc_pt_dist2bbox(self, bbox, pt):
        """ calc the distance between pt and bbox center pt
        Args:
            bbox: [u_min, u_max, v_min, v_max]
            pt: [u, v]
        """
        u_min, u_max, v_min, v_max = bbox
        u, v = pt

        u_b_center, v_b_center = (u_min + u_max) / 2, (v_min+v_max)/2
        
        dist = math.sqrt((u - u_b_center)**2 + (v - v_b_center)**2)

        return dist

    def _sorted_labels_list2str(self, label_list):
        """ convert label list to str for dict key demands
        """
        assert len(label_list) > 0, "invalid label list"
        str_list = [str(x) for x in label_list]
        label_str = '_'.join(str_list)

        return label_str
        
    def check_overlap_in_single_win(self, sem, loc, win_size):
        """ # check this window, stastic the label-num pair
            TODO Speed up! - done
            if have multi semantic label: return True, sorted_label_list:[...]
            else: return False, []
            ## label area need filted by self.thd
        """
        loc_win_label_dict = {}
        temp_stas_dict = {}
        radius = win_size
        W, H = self.size
        u_, v_ = loc
        assert sem[v_, u_] != -1

        u_min_ = max(0, u_ - radius)
        u_max_ = min(W, u_ + radius)
        v_min_ = max(0, v_ - radius)
        v_max_ = min(H, v_ + radius)

        temp_patch = np.squeeze(sem[v_min_:v_max_, u_min_:u_max_].reshape((-1,1))).tolist()
        temp_stas_dict = collections.Counter(temp_patch)
        
        for label in temp_stas_dict.keys():
            if label == -1: continue
            if temp_stas_dict[label] > self.small_label_filted_thd_inside_area:
                loc_win_label_dict.update({label: temp_stas_dict[label]})
        
        if len(loc_win_label_dict.keys()) >= 3:
            var_array = np.array(list(loc_win_label_dict.values()))
            var = np.var(var_array)
            return True, sorted(list(loc_win_label_dict.keys())), var
        else:
            return False, [], 0

    def achieve_overlap_patch(self):
        """
        Args:
            self.matched_overlap_area0&1 -> crop ori img
        Returns:
            crop_patch_matches: [[patch_img0, patch_img1]s]
            offsets: [[u0_offset, v0_offset, u1_offset, v1_offset]s]
        """
        crop_patch_matches = []
        offsets = []
        scales = []

        W, H = self.size

        for area0, area1 in zip(self.matched_overlap_area0, self.matched_overlap_area1):

            # offsets.append([area0[0], area0[2], area1[0], area1[2]])
            # patch_img0, scales0 = img_crop_with_resize(self.color0, area0, [self.output_patch_size, self.output_patch_size])
            # patch_img1, scales1 = img_crop_with_resize(self.color1, area1, [self.output_patch_size, self.output_patch_size])

            patch_img0, scales0, offset0 = img_crop_without_Diffscale(self.color0, area0, self.output_patch_size)
            patch_img1, scales1, offset1 = img_crop_without_Diffscale(self.color1, area1, self.output_patch_size)
            offsets.append([offset0[0], offset0[1], offset1[0], offset1[1]])

            scales.append([scales0[0], scales0[1], scales1[0], scales1[1]])
            crop_patch_matches.append([patch_img0, patch_img1])
        
        return crop_patch_matches, offsets, scales

    def fuse_patch_corrs(self, corrs, offsets, scales):
        """ recover corrs to ori images
        Args:
            corrs: [u0, v0, u1 v1]s
            offsets: [u0_offset, v0_offset, u1_offset, v1_offset]
            scales: [u0_scale, v0_scale, u1_scale, v1_scale]
        Returns:
            ori_matches: [u0, v0, u1, v1]
        """
        uv0, uv1 = corrs[:, :2], corrs[:, 2:]
        logger.info(f"fuse with offset: {offsets} and scales: {scales}")
        # print("uv0 size:", uv0.shape)
        # print("uv1 size:", uv1.shape)

        u0_offset, v0_offset, u1_offset, v1_offset = offsets
        u0_scale, v0_scale, u1_scale, v1_scale = scales
        # print("offsets: ", u0_offset, v0_offset, u1_offset, v1_offset)

        ori_matches = []
        N = uv0.shape[0]
        assert N == uv1.shape[0]
        for i in range(N):
            u0, v0 = uv0[i]
            u1, v1 = uv1[i]
            u0_ = u0 * u0_scale
            v0_ = v0 * v0_scale
            u1_ = u1 * u1_scale
            v1_ = v1 * v1_scale
            ori_matches.append([u0_+u0_offset, v0_+v0_offset, u1_+u1_offset, v1_+v1_offset])
        
        logger.info(f"fuse {len(ori_matches)} matches")
        # print("ori match\n", ori_matches)
        return ori_matches

    def collect_doubt_obj_pair(self, doubt_areas0, doubt_areas1):
        """
        Args:
            doubt_areai: {label: [area_idxs]}
        Returns:
            doubt_area_match_pairs: [pairs]
                pair: [[area0s], [area1s]]
                    pair[0]: areas in img0
        """
        doubt_match_pairs = []

        for label in doubt_areas0.keys():
            temp_pair = []
            temp_areas0 = []
            temp_areas1 = []
            for area_idx0 in doubt_areas0[label]:
                temp_areas0.append(self.sem0_obj_patch[area_idx0][:4])

            if label not in doubt_areas1.keys(): continue

            for area_idx1 in doubt_areas1[label]:
                temp_areas1.append(self.sem1_obj_patch[area_idx1][:4])

            temp_pair.append(temp_areas0)
            temp_pair.append(temp_areas1)
            doubt_match_pairs.append(temp_pair)

        return doubt_match_pairs

    """ Main Funcs """
    def FindMatchArea(self):
        """
        Returns:
            matched_all_area0/1: [[area0/1]s]
            total_crops: [[patch0, patch1]s]
            doubt_match_pairs: 
                [[pair]s]: each pair is a set of doubt area matches
                pair: [[area0]s, [area1]s]
                area: [u_min, u_max, v_min, v_max]
        """

        # match obj area
        logger.info(f"\n***\nStarting Perfrom RawSem Area Matching\n***")
        doubt_area_dict0, doubt_area_dict1 = self.match_object_patch()

        doubt_match_pairs = self.collect_doubt_obj_pair(doubt_area_dict0, doubt_area_dict1)
        self.achieve_obj_match_scale()

        if self.draw_verbose == 1:
            logger.info(f"images are saved in {self.out_path}")
            self.draw_doubt_match_pairs(doubt_match_pairs)
            self.draw_obj_match_res()

        # match overlap area
        # self.get_overlap_area_match(window_size=self.overlap_radius)
        self.match_overlap_area_pyramid_version()

        crops_obj, offsets_obj, scales_obj = self.achieve_obj_patches()
        crops_overlap, offsets_overlap, scales_overlap = self.achieve_overlap_patch()

        total_crops = crops_obj + crops_overlap
        
        # self.draw_object_patch()

        self.draw_all_area_match(self.draw_verbose)

        return self.matched_all_area0, self.matched_all_area1, doubt_match_pairs, total_crops

    def area_matching(self, dataloader, outpath):
        """ NOTE: the doubtful areas are ignored in this version
        """
        self.init_dataloader(dataloader)
        self.set_outpath(outpath)
        
        self.get_label_list_collect()
        if self.draw_verbose == 1:
            logger.info(f"labels are {self.label_list}")
            self.draw_ori_img()
            self.draw_sem()
            
        matched_area0, matched_area1, doubt_match_pairs, total_crops = self.FindMatchArea()

        return matched_area0, matched_area1

    """ Draw Funs """

    def draw_ori_img(self):
        """
        """
        cv2.imwrite(os.path.join(self.out_path, "ori_" + self.name0 + ".jpg"), self.color0)
        cv2.imwrite(os.path.join(self.out_path, "ori_" + self.name1 + ".jpg"), self.color1)

        return self.color0, self.color1
    
    def draw_sem(self):
        """
        """
        self.color_sem0, self.color_sem1 = paint_semantic(self.sem0, self.sem1, self.out_path, self.name0, self.name1)

        return self.color_sem0, self.color_sem1

    def draw_vis_connected_after_clean(self):
        """
        """
        # draw sem0 connected area
        for k in self.sem0_sem_connected.keys():
            temp_img = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
            for area in self.sem0_sem_connected[k]:
                for [v, u] in area:
                    temp_img[v, u] = 255
            if temp_img.max() > 0:
                out_name = os.path.join(self.out_path, str(k)+"_sem_" + self.name0+ ".jpg")
                cv2.imwrite(out_name, temp_img)

        # draw sem1 connected area
        for k in self.sem1_sem_connected.keys():
            temp_img = np.zeros((self.size[1], self.size[0]), dtype=np.uint8)
            for area in self.sem1_sem_connected[k]:
                for [v, u] in area:
                    temp_img[v, u] = 255
            if temp_img.max() > 0:
                out_name = os.path.join(self.out_path, str(k)+"_sem_" + self.name1 + ".jpg")
                cv2.imwrite(out_name, temp_img)

    def draw_object_patch(self):
        """
        """
        object_img0 = self.color0.copy()
        object_img1 = self.color1.copy()

        for obj in self.sem0_obj_patch:
            cv2.rectangle(object_img0, (obj[0], obj[2]), (obj[1], obj[3]), (255, 0, 0), 2)

        for obj in self.sem1_obj_patch:
            cv2.rectangle(object_img1, (obj[0], obj[2]), (obj[1], obj[3]), (255, 0, 0), 2)
        
        cv2.imwrite(os.path.join(self.out_path, "obj" + self.name0 + ".jpg"), object_img0)
        cv2.imwrite(os.path.join(self.out_path, "obj" + self.name1 + ".jpg"), object_img1)

    def draw_obj_match_res(self):
        """Draw the match res
            draw the rectangle and line
        """
        l_label = len(self.matched_obj_label)
        label_color_dict = {}

        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, l_label)))

        for i in range(l_label):
            c = cmap(i)
            label_color_dict[self.matched_obj_label[i]] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
        
        W, H = self.size
        H_s, W_s = H, W*2
        out = 255 * np.ones((H_s, W_s, 3), np.uint8)

        object_img0 = self.color0.copy()
        object_img1 = self.color1.copy()

        out[:H, :W, :] = object_img0
        out[:H, W:, :] = object_img1 

        for i, label in enumerate(self.matched_obj_label):
            patch0 = self.matched_obj_patch0[i]
            patch1 = self.matched_obj_patch1[i]

            patch1_s = [patch1[0]+W, patch1[1]+W, patch1[2], patch1[3]]

            # logger.info(f"patch0 are {patch0[0]}, {patch0[1]}, {patch0[2]}, {patch0[3]}")
            # logger.info(f"patch1 are {patch1_s[0]}, {patch1_s[1]}, {patch1_s[2]}, {patch1_s[3]}")

            cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(label_color_dict[label]), 2)
            cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), label_color_dict[label], 2)

            line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
            line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

            cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=label_color_dict[label], thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(self.out_path, "obj_match_res_" + self.name0 + "_" + self.name1 + ".jpg"), out)

    def draw_all_area_match(self, draw_flag=0):
        """ draw all the area match 
        Args:
            self.matched_obj_patch0
            self.matched_obj_patch1
            self.matched_overlap_area0
            self.matched_overlap_area1
        """
        self.matched_all_area0 = []
        self.matched_all_area1 = []

        for i, area_obj_0 in enumerate(self.matched_obj_patch0):
            self.matched_all_area0.append(area_obj_0)
            self.matched_all_area1.append(self.matched_obj_patch1[i])
        
        for j, area_overlap_0 in enumerate(self.matched_overlap_area0):
            self.matched_all_area0.append(area_overlap_0)
            self.matched_all_area1.append(self.matched_overlap_area1[j])
        
        l_all = len(self.matched_all_area0)
        assert len(self.matched_all_area0) == len(self.matched_all_area1)

        logger.info(f"Achieve {l_all} matched patches totally")

        if l_all == 0:
            logger.info(f"No matched areas")
            return
            
        if draw_flag == 0:
            return

        label_color_dict = {}

        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, l_all)))

        for i in range(l_all):
            c = cmap(i)
            label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
        
        W, H = self.size
        H_s, W_s = H, W*2
        out = 255 * np.ones((H_s, W_s, 3), np.uint8)

        object_img0 = self.color0.copy()
        object_img1 = self.color1.copy()

        out[:H, :W, :] = object_img0
        out[:H, W:, :] = object_img1 

        for i in range(l_all):
            patch0 = self.matched_all_area0[i]
            patch1 = self.matched_all_area1[i]

            patch1_s = [patch1[0]+W, patch1[1]+W, patch1[2], patch1[3]]

            # logger.info(f"patch0 are {patch0[0]}, {patch0[1]}, {patch0[2]}, {patch0[3]}")
            # logger.info(f"patch1 are {patch1_s[0]}, {patch1_s[1]}, {patch1_s[2]}, {patch1_s[3]}")

            cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(label_color_dict[i]), 2)
            cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), label_color_dict[i], 2)

            line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
            line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

            cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=label_color_dict[i], thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(self.out_path, "all_matched_area_" + self.name0 + "_" + self.name1 + ".jpg"), out)

    def draw_ori_matches(self, ori_matches, name):
        """
        """
        plot_matches_lists_ud(self.color0, self.color1, ori_matches, self.out_path, name)

    def draw_overlap_area(self, overlap_dict, ori_img, name):
        """ draw overlap area from dict
        Args:
            overlap_dict = {
                label_str : [[area],...]
            }
        """
        out_img = ori_img.copy()

        N = len(overlap_dict.keys())
        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, N)))

        for i, label_s in enumerate(overlap_dict.keys()):
            c_raw = cmap(i)
            color = [int(c_raw[0]*255),int(c_raw[1]*255),int(c_raw[2]*255)]
            for area in overlap_dict[label_s]:
                cv2.rectangle(out_img, (area[0], area[2]), (area[1], area[3]), color, 2)
        
        cv2.imwrite(os.path.join(self.out_path, name+"_overlap.jpg"), out_img)
    
    def draw_overlap_area_list(self, overlap_areas, ori_img, name):
        """
        """
        out_img = ori_img.copy()

        N = len(overlap_areas)
        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, N)))

        for i, area in enumerate(overlap_areas):
            c_raw = cmap(i)
            color = [int(c_raw[0]*255),int(c_raw[1]*255),int(c_raw[2]*255)]
            cv2.rectangle(out_img, (area[0], area[2]), (area[1], area[3]), color, 2)
        
        cv2.imwrite(os.path.join(self.out_path, name+"_overlap.jpg"), out_img)

    def draw_overlap_match_res(self, name=""):
        """ Draw the overlap match result
        """
        l_label = len(self.matched_overlap_area0)
        if l_label == 0: 
            logger.info(f"No matched overlap is found")
            return
        label_color_dict = {}

        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, l_label)))

        for i in range(l_label):
            c = cmap(i)
            label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
        
        W, H = self.size
        H_s, W_s = H, W*2
        out = 255 * np.ones((H_s, W_s, 3), np.uint8)

        object_img0 = self.color0.copy()
        object_img1 = self.color1.copy()

        out[:H, :W, :] = object_img0
        out[:H, W:, :] = object_img1 

        for i, label in enumerate(self.matched_overlap_area0):
            patch0 = self.matched_overlap_area0[i]
            patch1 = self.matched_overlap_area1[i]

            patch1_s = [patch1[0]+W, patch1[1]+W, patch1[2], patch1[3]]

            # logger.info(f"patch0 are {patch0[0]}, {patch0[1]}, {patch0[2]}, {patch0[3]}")
            # logger.info(f"patch1 are {patch1_s[0]}, {patch1_s[1]}, {patch1_s[2]}, {patch1_s[3]}")

            cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(label_color_dict[i]), 2)
            cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), label_color_dict[i], 2)

            line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
            line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

            cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=label_color_dict[i], thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(self.out_path, "overlap_area_match_res_" + self.name0 + "_" + self.name1 + name + ".jpg"), out)

    def draw_doubt_match_pairs(self, doubt_match_pairs):
        """ draw doubt match pairs
        Args:
            doubt_mathc_pairs: 
                [[pair]s]
                pair: [[area0]s, [area1]s]
                area: [u_min, u_max, v_min, v_max]
            self.color0/1
        """
        if len(doubt_match_pairs) == 0: return

        l_all = len(doubt_match_pairs)

        label_color_dict = {}

        cmaps_='gist_ncar'
        cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, l_all)))

        for i in range(l_all):
            c = cmap(i)
            label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
        
        W, H = self.size
        H_s, W_s = H, W*2
        out = 255 * np.ones((H_s, W_s, 3), np.uint8)

        object_img0 = self.color0.copy()
        object_img1 = self.color1.copy()

        out[:H, :W, :] = object_img0
        out[:H, W:, :] = object_img1 

        for i in range(l_all):
            
            pair_temp = doubt_match_pairs[i]
            assert len(pair_temp) == 2, f"invalid pair with len {len(pair_temp)}"

            for patch0 in pair_temp[0]:
                cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(label_color_dict[i]), 2)
                
            for patch1 in pair_temp[1]:
                patch1_s = [patch1[0]+W, patch1[1]+W, patch1[2], patch1[3]]
                cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), tuple(label_color_dict[i]), 2)

        cv2.imwrite(os.path.join(self.out_path, "doubted_matched_area_" + self.name0 + "_" + self.name1 + ".jpg"), out)
