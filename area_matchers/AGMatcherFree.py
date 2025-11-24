'''
Author: EasonZhang
Date: 2023-06-28 21:21:13
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-24 15:06:13
FilePath: /SA2M/hydra-mesa/area_matchers/AGMatcherFix.py
Description: Given two images, construct their area graphs and match them. [Graphical Area Matching in Paper]
             First use Self-Energy for Graph-cut, then use Global-Energy for Refinement.

Copyright (c) 2023 by EasonZhang, All Rights Reserved. 
'''

import sys
sys.path.append("..")


import os
import os.path as osp
import numpy as np
import copy
import torch
import cv2
from loguru import logger
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from typing import Any

from .AreaGrapher import AreaGraph
from .AGConfig import areagraph_configs
from .AGUtils import GraphCutSolver
from .CoarseAreaMatcher import CoarseAreaMatcher
from .AGBasic import AGNode
from utils.vis import draw_matched_area, draw_matched_area_list, draw_matched_area_with_mkpts
from utils.common import test_dir_if_not_create
from utils.geo import calc_areas_iou

dft_config = {
    "matcher_name": "ASpan",
    "mast3r_weight_path": "", # path to mast3r weights
    "datasetName": "ScanNet",
    "out_path": "/data2/zys/A2PM/testAG/GCRes",
    "level_num": 4,
    "level_step": [560, 390, 256, 130, 0],
    "global_energy_weights": [10, 1, 1, 1],
    "adj_weight": 0.01,
    "stop_match_level": 3,
    "W": 640,
    "H": 480,
    "coarse_match_thd": 0.1,
    "patch_size": 16,
    "keep_area_pts_ratio": 0.3,
    "area_overlap_strong_thd": 0.5,
    "similar_area_dist_thd": 10,
    "area_w": 480,
    "area_h": 480,
    "show_flag": 1,
    "coarse_match_num_thd": 30,
    "sigma_thd": 0.1, # activity < sigma_thd set to 0
    "iou_fusion_thd": 0.8,
    "global_refine": 1, # 1 is refine, 0 is refine by self energy
    "candi_energy_thd": 0.7, # for self energy
    "global_energy_candi_range": 0.1, # for global energy, find the best and candis within the range
    "fast_version": 0, # 1 is fast, 0 is slow
}

class AGMatcherF(object):
    """ get area matches for each level of area graph from two images
    Funcs:
        0. load paths of two images
        1. get area graph from two images
        2. init an area matcher
        3. cycle match for two area graphs
        4. collect matches for each level of area graph
    """
    def __init__(self, configs={}) -> None:
        """
        Args:   
            level_areamatch_dict: dict, key: level, value: list of matches
                - list of matches: [[area0_idx, area1_idx], [area0_idx, area1_idx], ...]
        """
        # update dft_config by configs
        dft_config.update(**configs)

        self.level_areamatch_dict = defaultdict(list)
        self.matcher_name = dft_config["matcher_name"]
        self.mast3r_weight_path = dft_config["mast3r_weight_path"] # path to mast3r weights
        self.datasetName = dft_config["datasetName"]
        self.out_path = dft_config["out_path"]
        self.level_num = dft_config["level_num"]
        self.level_step = dft_config["level_step"]
        self.stop_match_level = dft_config["stop_match_level"]
        self.sigma_thd = dft_config["sigma_thd"]
        self.global_energy_weights = dft_config["global_energy_weights"]
        self.adj_weight = dft_config["adj_weight"]
        self.iou_fusion_thd = dft_config["iou_fusion_thd"]
        self.candi_energy_thd = dft_config["candi_energy_thd"]
        self.global_energy_candi_range = dft_config["global_energy_candi_range"]
        self.global_refine = dft_config["global_refine"] # 1 is refine, 0 is refine by self energy
        self.show_flag = dft_config["show_flag"] # int 1 is show, 0 is not show
        self.areagraph0 = None
        self.areagraph1 = None
        self.activity_map = None
        self.img_path0 = None
        self.img_path1 = None
        self.name0: Any = None
        self.name1 = None
        self.img0 = None
        self.img1 = None
        self.reverse_flag = False
        
        self.W, self.H = configs["W"], configs["H"]
        self.coarse_match_thd = configs["coarse_match_thd"]
        self.patch_size = configs["patch_size"]
        self.similar_area_dist_thd = configs["similar_area_dist_thd"]
        self.area_w, self.area_h = configs["area_w"], configs["area_h"]
        # self.show_flag = configs["show_flag"] # int 1 is show, 0 is not show
        self.fast_version = configs["fast_version"]
        
        # collect final matched areas during the match flow (idxs)
        self.final_area_match_list_src = []
        self.final_area_match_list_dst = []

        # results
        self.res_area_match_list_src = []
        self.res_area_match_list_dst = []

        pass

    def init_area_matcher(self):
        """
        """
        matcher_configs = {
            "matcher_name": self.matcher_name,
            'mast3r_weight_path': self.mast3r_weight_path if self.matcher_name=="mast3r" else "",
            "datasetName": self.datasetName,
            "out_path": self.out_path,
            "pair_name": self.name0 + "_" + self.name1,
            "area_w": self.area_w,
            "area_h": self.area_h,
            "patch_size": self.patch_size,
            "conf_thd": self.coarse_match_thd,
        }
        self.area_matcher = CoarseAreaMatcher(matcher_configs)
        self.area_matcher.init_matcher()
        
    def path_loader(self, img_path0, sam_res_path0, img_path1, sam_res_path1, name0, name1):
        """
        """
        logger.info("load paths of two images")
        self.name0 = name0
        self.name1 = name1
        self.img_path0 = img_path0
        self.img_path1 = img_path1
        self.sam_res_path0 = sam_res_path0
        self.sam_res_path1 = sam_res_path1
        self.out_path = self.out_path + "/" + self.name0 + "_" + self.name1
        # test_dir_if_not_create(self.out_path)
        self._set_ag_config()
    
    def ori_img_load(self):
        """
        """
        if self.img0 is not None and self.img1 is not None:
            return
        assert self.img_path0 is not None and self.img_path1 is not None, "img_path0 and img_path1 should not be None"
        self.img0 = cv2.imread(self.img_path0, cv2.IMREAD_COLOR)
        self.img1 = cv2.imread(self.img_path1, cv2.IMREAD_COLOR)

        # resize
        self.img0 = cv2.resize(self.img0, (self.W, self.H))
        self.img1 = cv2.resize(self.img1, (self.W, self.H))

        return self.img0, self.img1

    def _set_ag_config(self):
        """
        """
        self.ag_config0 = deepcopy(areagraph_configs)
        self.ag_config0["ori_img_path"] = self.img_path0
        self.ag_config0["sam_res_path"] = self.sam_res_path0
        self.ag_config0["level_num"] = self.level_num
        self.ag_config0["level_step"] = self.level_step
        save_path0 = os.path.join(self.out_path, "area_graph0")
        self.ag_config0["save_path"] = save_path0
        self.ag_config0["show_flag"] = self.show_flag
        if self.show_flag == 1:
            test_dir_if_not_create(save_path0)


        self.ag_config1 = deepcopy(areagraph_configs)
        self.ag_config1["ori_img_path"] = self.img_path1
        self.ag_config1["sam_res_path"] = self.sam_res_path1
        self.ag_config1["level_num"] = self.level_num
        self.ag_config1["level_step"] = self.level_step
        save_path1 = os.path.join(self.out_path, "area_graph1")
        self.ag_config1["save_path"] = save_path1
        self.ag_config1["show_flag"] = self.show_flag
        if self.show_flag == 1:
            test_dir_if_not_create(save_path1)
    
    def img_areagraph_construct(self, efficient=False):
        """
        """
        # logger.info("construct area graph from two images")
        self.areagraph0 = AreaGraph(self.ag_config0)
        self.areagraph1 = AreaGraph(self.ag_config1)
        # clear
        self.final_area_match_list_src = []
        self.final_area_match_list_dst = []

        return self.areagraph0, self.areagraph1

    def coarse_match(self, area0, area1, idx0, idx1):
        """
        Return:
            mkpts0, mkpts1: np.array (N, 2), pts in the area0 and area1
        """
        match_res: Any = self.area_matcher.match(area0, area1)
        mkpts0, mkpts1, mconf, conf_mat = match_res
        
        if self.show_flag:
            self.area_matcher.visulization(area0, area1, mkpts0, mkpts1, mconf, conf_mat, name=f"{idx0}_{idx1}")

        # filter by confidence
        mkpts0 = mkpts0[mconf > self.coarse_match_thd]
        mkpts1 = mkpts1[mconf > self.coarse_match_thd]

        return mkpts0, mkpts1

    def ret_area_only(self, area_idx_list, area_graph):
        """
        """
        areas = []
        for area_idx in area_idx_list:
            areas.append(area_graph.AGNodes[area_idx].area)
        return areas

    def _fuse_match_res(self, matched_src_idx, matched_dst_idx, matched_src_idx_, matched_dst_idx_):
        """
        """
        matched_src_idx_f = []
        matched_dst_idx_f = []
        match_pairs = []
        for idx, (src_idx, dst_idx) in enumerate(zip(matched_src_idx, matched_dst_idx)):
            if (src_idx, dst_idx) in match_pairs:
                continue
            match_pairs.append((src_idx, dst_idx))
        
        for idx, (src_idx, dst_idx) in enumerate(zip(matched_src_idx_, matched_dst_idx_)):
            if (src_idx, dst_idx) in match_pairs:
                continue
            match_pairs.append((src_idx, dst_idx))
        
        for src_idx, dst_idx in match_pairs:
            matched_src_idx_f.append(src_idx)
            matched_dst_idx_f.append(dst_idx)
        
        return matched_src_idx, matched_dst_idx

    """ Graphical Model based Match============================================================================================================"""
    """ Graphical Model based Match============================================================================================================"""
    """ Graphical Model based Match============================================================================================================"""

    def dual_graphical_match(self, draw_flag=False):
        """
        Returns:
            matched_area0s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...]
            matched_area1s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...]
        """
        # logger.info(f"start dual graphical match")
        matched_area0, matched_area1 = self.match_by_graphical_model(reverse=False, draw_flag=draw_flag)
        r_matched_area1, r_matched_area0 = self.match_by_graphical_model(reverse=True, draw_flag=draw_flag)

        # fusion
        # self.fusion_matched_pairs(matched_area0, r_matched_area0, matched_area1, r_matched_area1)
        matched_area0, matched_area1 = self.fuse_repeat_area_matches(matched_area0, matched_area1, r_matched_area0, r_matched_area1)

        if draw_flag:
            self.ori_img_load()
            out_img_folder = os.path.join(self.out_path, f"final_matches")
            test_dir_if_not_create(out_img_folder)
            draw_matched_area_list(self.img0, self.img1, matched_area0, matched_area1, out_img_folder, "final", "match")

        return matched_area0, matched_area1

    def fuse_repeat_area_matches(self, matched_area0s, matched_area1s, r_matched_area0s, r_matched_area1s):
        """ exaustive repeat elimination, an update version for fusion_matched_pairs
        Args:
            matched_area0s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...], forward matched area0s
            matched_area1s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...]
            r_matched_area0s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...], reverse matched area0s
            r_matched_area1s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...]
        Returns:
            res_matched_area0s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...], fused matched area0s
            res_matched_area1s: list, [[u_min, u_max, v_min, v_max], [u_min, u_max, v_min, v_max], ...]
        """  
        res_matched_area0s = []
        res_matched_area1s = []

        # add matched_area0s and matched_area1s to res_matched_area0s and res_matched_area1s, without repeativeness
        for area0, area1 in zip(matched_area0s, matched_area1s):
            self.add_area_matches_non_repeat(area0, area1, res_matched_area0s, res_matched_area1s)

        # add r_matched_area0s and r_matched_area1s to res_matched_area0s and res_matched_area1s, without repeativeness
        for area0, area1 in zip(r_matched_area0s, r_matched_area1s):
            self.add_area_matches_non_repeat(area0, area1, res_matched_area0s, res_matched_area1s)
        
        res_matched_area0s_ = []
        res_matched_area1s_ = []

        # add res_matched_area0s and res_matched_area1s to res_matched_area0s_ and res_matched_area1s_, without repeativeness
        for area0, area1 in zip(res_matched_area0s, res_matched_area1s):
            self.add_area_matches_non_repeat(area0, area1, res_matched_area0s_, res_matched_area1s_)
        

        return res_matched_area0s_, res_matched_area1s_

    def add_area_matches_non_repeat(self, area0, area1, matched_area0s, matched_area1s):
        """ add area0 and area1 to matched_area0s and matched_area1s, without repeativeness
        """
        if len(matched_area0s) == 0:
            matched_area0s.append(area0)
            matched_area1s.append(area1)
            return

        max_iou0, max_id0 = self.calc_max_iou(area0, matched_area0s)
        if max_iou0 > self.iou_fusion_thd:
            matched_area0s[max_id0], matched_area1s[max_id0] = self.fusion_matched_pair(matched_area0s[max_id0], matched_area1s[max_id0], area0, area1)
            return
        
        max_iou1, max_id1 = self.calc_max_iou(area1, matched_area1s)
        if max_iou1 > self.iou_fusion_thd:
            matched_area0s[max_id1], matched_area1s[max_id1] = self.fusion_matched_pair(matched_area0s[max_id1], matched_area1s[max_id1], area0, area1)
            return
        
        matched_area0s.append(area0)
        matched_area1s.append(area1)

    def fusion_matched_pairs(self, matched_area0s, r_matched_area0s, matched_area1s, r_matched_area1s):
        """ update matched_area0s and matched_area1s by fusion
        """
        for area0, area1 in zip(r_matched_area0s, r_matched_area1s):
            # calc max of area0 in matched_area0s
            max_iou0, max_id0 = self.calc_max_iou(area0, matched_area0s)
            if max_iou0 > self.iou_fusion_thd:
                matched_area0s[max_id0], matched_area1s[max_id0] = self.fusion_matched_pair(matched_area0s[max_id0], matched_area1s[max_id0], area0, area1)
                continue

            # calc max of area1 in matched_area1s
            max_iou1, max_id1 = self.calc_max_iou(area1, matched_area1s)
            if max_iou1 > self.iou_fusion_thd:
                matched_area0s[max_id1], matched_area1s[max_id1] = self.fusion_matched_pair(matched_area0s[max_id1], matched_area1s[max_id1], area0, area1)
                continue

            matched_area0s.append(area0)
            matched_area1s.append(area1)
            
    def fusion_matched_pair(self, area0, area1, r_area0, r_area1):
        """ fusion two matched pairs
        Args:
            area0: list, [u_min, u_max, v_min, v_max]
            r_area0: list, [u_min, u_max, v_min, v_max]
            ...
        """
        # fuse area0 and r_area0
        area0 = self.fuse_2areas(area0, r_area0)
        area1 = self.fuse_2areas(area1, r_area1)
        
        return area0, area1

    def fuse_2areas(self, area0, area1):
        """
        """
        u_min = min(area0[0], area1[0])
        u_max = max(area0[1], area1[1])
        v_min = min(area0[2], area1[2])
        v_max = max(area0[3], area1[3])
        return [u_min, u_max, v_min, v_max]
    
    def calc_max_iou(self, area0, areas):
        """
        """
        max_iou = 0
        max_id = -1
        for i, area in enumerate(areas):
            iou = calc_areas_iou(area0, area)
            max_iou = max(max_iou, iou)
            if max_iou != iou:
                max_id = i

        return max_iou, max_id
        
    def match_by_graphical_model(self, reverse=False, draw_flag=False):
        """
        Flow:
            1. calc the activity map
            2. choose the nodes with specific level
            3. find the match for each node
                a. construct the E graph
                    - calc the terminal edge weight = E_self + E_son + E_parent + E_neighbor
                    - calc the edge weight between two nodes = overlap
                b. minimize the E graph by GraphCut
        Returns:
            matched_areas_src
            matched_areas_dst
        """
        if reverse:
            ag_src = self.areagraph1
            ag_dst = self.areagraph0
        else:
            ag_src: Any = self.areagraph0
            ag_dst = self.areagraph1

        logger.success("start calc activity map")
        if self.activity_map is None:
            self.activity_map = self.calc_activity_map(reverse=reverse)
        else: # if already calc, transpose the activity map
            # transpose the activity map
            self.activity_map = self.activity_map.T

        matched_areas_src = []
        matched_areas_dst = [] # -1 means no match

        logger.success("start match")
        # choose the nodes with specific level
        src_node_list = self.choose_nodes(ag_src, self.stop_match_level)

        # find the match for each node
        for src_idx in tqdm(src_node_list, ncols=80):
            # logger.success(f"match src node {src_idx}")
            src_expand_area = ag_src.AGNodes[src_idx].expand_area
            temp_e_graph = self.construct_E_graph(src_idx, ag_src, ag_dst, draw_flag=draw_flag, reverse=reverse)
            matched_list = self.minimize_E_by_GraphCut(temp_e_graph)
            
            if self.global_refine == 1:
                final_matched_area = self.global_match_refine(matched_list, temp_e_graph, ag_src, src_idx, ag_dst)
            else:
                final_matched_area = self.self_match_refine(matched_list, temp_e_graph, ag_dst)
            if len(final_matched_area) == 0: 
                # logger.warning(f"no match for src node {src_idx}")
                continue
            matched_areas_src.append(src_expand_area)
            matched_areas_dst.append(final_matched_area)

        return matched_areas_src, matched_areas_dst

    def choose_nodes(self, ag_src, node_level):
        """ choose node with specific level
        """
        node_list = ag_src.get_nodes_with_level(node_level)
        return node_list

    def construct_E_graph(self, src_idx, ag_src, ag_dst, draw_flag=False, reverse=False):
        """ link to source means the node is matched to the source node
            link to sink means the node is not matched to the source node
        Args:
            src_idx: int, the src node idx
            ag_src: AreaGraph, the src area graph
            ag_dst: AreaGraph, the dst area graph
        Returns:
            E_graph: adjacency matrix of the E graph
            NOTE: a new graph should be formed accoding to the spcific label setting for each node, which may change the adjenct energy
                size = (dst_nodes_num+1, dst_nodes_num+1): last node represents the weights between source and sink terminal node
                E_graph[i, -1] = edge weight between src node i and source; NOTE the energy is only the self energy              
                E_graph[-1, i] = edge weight between src node i and sink
                E_graph[i, j] = edge weight between src node i and dst node j
                    -1: no link
                    other: edge weight
        """
        node_num = len(ag_dst.AGNodes)
        E_graph = np.zeros((node_num+1, node_num+1)) - 1 # [-1,i] is the weight from source to node i, [i, -1] is the weight from node i to sink
        E_calced = np.zeros((node_num+1, node_num+1)) # 1 means the edge is calculated
        # make the digonal of E_calced to 1
        for i in range(node_num+1):
            E_calced[i, i] = 1

        # initial label is matched to source
        # calc the energy of each node to match src node
        for dst_idx in range(node_num):
            E_graph[dst_idx, -1] = self.calc_node_energy(src_idx, dst_idx) # link to source
            E_graph[-1, dst_idx] = 1 - E_graph[dst_idx, -1] # link to sink
            E_calced[dst_idx, -1] = 1
            E_calced[-1, dst_idx] = 1
            self.calc_adjenct_energy(dst_idx, ag_dst, E_graph, E_calced)
        
        # visualize the E graph
        # draw the best match for this src node
        # if draw_flag:
        #     self.ori_img_load()
        #     out_img_folder = os.path.join(self.out_path, f"match_inE_{src_idx}")
        #     test_dir_if_not_create(out_img_folder)
        #     for dst_idx in range(node_num):
        #         area_src = ag_src.AGNodes[src_idx].expand_area
        #         area_dst = ag_dst.AGNodes[dst_idx].expand_area
        #         color = (0, 0, 255)
        #         if not reverse:
        #             draw_matched_area(self.img0, self.img1, area_src, area_dst, color, out_img_folder, f"{src_idx}", f"{dst_idx}_{E_graph[dst_idx, -1]:.2f}")
        #         else:
        #             draw_matched_area(self.img1, self.img0, area_src, area_dst, color, out_img_folder, f"{src_idx}", f"{dst_idx}_{E_graph[dst_idx, -1]:.2f}")

        return E_graph

    def calc_adjenct_energy(self, dst_idx, ag_dst, E_graph, E_calced, adj_weight=0.01):
        """ for each [dst_idx, i] in E_graph, calc the adj-energy
        """
        node_num = len(ag_dst.AGNodes)
        for i in range(node_num):
            if E_calced[dst_idx, i] == 1 and E_calced[i, dst_idx] == 1:
                continue
            iou = self.calc_IOU(dst_idx, i, ag_dst)
            E_graph[dst_idx, i] = iou * adj_weight
            E_graph[i, dst_idx] = iou * adj_weight
            E_calced[dst_idx, i] = 1
            E_calced[i, dst_idx] = 1

    def calc_IOU(self, idx0, idx1, ag):
        """ calc the IOU of two nodes
        Args:
            idx0: int, the idx of node0
            idx1: int, the idx of node1
            ag: AreaGraph, the area graph
        Returns:
            iou: float, the IOU of two nodes
        """
        iou = ag.calc_IoU(idx0, idx1)
        return iou

    def calc_node_energy(self, src_idx, dst_idx):
        """ calc the energy of dst node to match src node
            E = E_self: 1 - Similarity
            Energy is small when the two nodes are matched
        Args:
            ar_src: AreaGraph, the src area graph
            ag_dst: AreaGraph, the dst area graph
            src_idx: int, the src node idx
            dst_idx: int, the dst node idx
            param: float, the weight of self energy
        """
        E_self = self.calc_self_energy(src_idx, dst_idx, 1)
        return E_self

    def calc_node_energy_global(self, ar_src, ag_dst, src_idx, dst_idx, param_list):
        """ calc the energy of dst node to match src node
            E = E_self + E_son + E_parent + E_neighbor
            Energy is small when the two nodes are similar
        Args:
            ar_src: AreaGraph, the src area graph
            ag_dst: AreaGraph, the dst area graph
            src_idx: int, the src node idx
            dst_idx: int, the dst node idx
            param_list: list, [w_self, w_son, w_parent, w_neighbor]
        """
        E_self = self.calc_self_energy(src_idx, dst_idx, param_list[0])
        E_son = self.calc_sons_energy(ar_src, ag_dst, src_idx, dst_idx, param_list[1])
        E_parent = self.calc_parents_energy(ar_src, ag_dst, src_idx, dst_idx, param_list[2])
        E_neighbor = self.calc_neighbor_energy(ar_src, ag_dst, src_idx, dst_idx, param_list[3])
        E = E_self + E_son + E_parent + E_neighbor
        
        # normalize
        E = E / (param_list[0] + param_list[1] + param_list[2] + param_list[3])

        if E_self == param_list[0]:
            E = 1

        E = min(E, 1)

        return E
    
    def calc_global_energy_list(self, matched_list, e_graph, ag_src, src_idx, ag_dst):
        """
        """
        E_list = []
        for dst_idx in matched_list:
            param_list = self.global_energy_weights
            E_self = e_graph[dst_idx, -1]
            E_son = self.calc_sons_energy(ag_src, ag_dst, src_idx, dst_idx, param_list[1])
            E_parent = self.calc_parents_energy(ag_src, ag_dst, src_idx, dst_idx, param_list[2])
            E_neighbor = self.calc_neighbor_energy(ag_src, ag_dst, src_idx, dst_idx, param_list[3])
            E = E_self + E_son + E_parent + E_neighbor

            # normalize
            E = E / (param_list[0] + param_list[1] + param_list[2] + param_list[3])

            if E_self == param_list[0]:
                E = 1

            E = min(E, 1)
            E_list.append(E)

        return E_list

    def calc_self_energy(self, src_idx, dst_idx, weight):
        """ achieve the activity map[src_idx, dst_idx] as the self energy
        """
        E_self = (1 - self.activity_map[src_idx, dst_idx]) * weight
        return E_self
    
    def calc_parents_energy(self, ag_src, ag_dst, src_idx, dst_idx, weight):
        """ calc the minimal energy of the parents of src node to match dst node's parents
        Flow:
            1. get the parents of src node
            2. get the parents of dst node
            3. calc the energy of each pair
            4. find the minimal energy
        """
        E_parent = 1
        src_parents_idx_list = ag_src.rt_parents(src_idx)
        dst_parents_idx_list = ag_dst.rt_parents(dst_idx)
        for src_parent_idx in src_parents_idx_list:
            for dst_parent_idx in dst_parents_idx_list:
                E_parent = min(E_parent, 1 - self.activity_map[src_parent_idx, dst_parent_idx])
        return E_parent * weight
    
    def calc_sons_energy(self, ag_src, ag_dst, src_idx, dst_idx, weight):
        """
        """
        E_son = 1
        src_sons_idx_list = ag_src.rt_all_sons(src_idx)
        dst_sons_idx_list = ag_dst.rt_all_sons(dst_idx)
        for src_son_idx in src_sons_idx_list:
            for dst_son_idx in dst_sons_idx_list:
                E_son = min(E_son, 1 - self.activity_map[src_son_idx, dst_son_idx])
        return E_son * weight

    def calc_neighbor_energy(self, ag_src, ag_dst, src_idx, dst_idx, weight):
        """
        """
        E_neighbor = 1
        src_neighbor_idx_list = ag_src.rt_neighbours(src_idx)
        dst_neighbor_idx_list = ag_dst.rt_neighbours(dst_idx)
        for src_neighbor_idx in src_neighbor_idx_list:
            for dst_neighbor_idx in dst_neighbor_idx_list:
                E_neighbor = min(E_neighbor, 1 - self.activity_map[src_neighbor_idx, dst_neighbor_idx])
        return E_neighbor * weight

    def minimize_E_by_GraphCut(self, E_graph):
        """ perform multi-times GraphCut to achieve the minimal cut with less than 3 nodes
        Args:
            E_graph: np.array, the E graph, the adjcent enery is the raw energy
        Returns:
            the minimal cut [list of node idxs]
        """
        # logger.info(f"start minimize E by GraphCut")
        solver = GraphCutSolver()
        node_idxs = solver.solve(E_graph)

        # log the energy of the minimal cut
        # logger.info(f"log the energy, smaller is better")
        # for idx in node_idxs:
        #     logger.info(f"node {idx} energy is {E_graph[idx, -1]}")

        return node_idxs

    def self_match_refine(self, matched_list, e_graph, ag_dst):
        """if the matched_list contians more than one node, do the refine
            mainly based on self energy, smaller is better
        """
        final_matched_area = []
        if len(matched_list) == 1:
            final_matched_area = ag_dst.AGNodes[matched_list[0]].expand_area
        elif len(matched_list) > 1:
            # get the candis to fuse
            # calc all global energies in the list
            # find the best match with the minimal energy and candis around it to fuse
            final_matched_area = self.multi_areas_fusion_weighted(matched_list, ag_dst, e_graph)
            # logger.success(f"find {len(matched_list)} matches, fused")
        else: 
            # no match by GC, select the best match from all by [NOTE] self energy
            candi_idxs = self.garp_candis_by_energy(e_graph)
            final_matched_area = self.multi_areas_fusion_weighted(candi_idxs, ag_dst, e_graph)
            # if len(final_matched_area) > 0 : logger.success(f"no match by GC, but find match by candis")

        return final_matched_area
    
    def global_match_refine(self, matched_list, e_graph, ag_src, src_idx, ag_dst):
        """ if the matched_list contians more than one node, do the refine
            mainly based on global energy, smaller is better
            E_global = E_self + E_son + E_parent + E_neighbor
        Funs:
            - weighted fusion
        Args:
            matched_list: list, matched dst node idxs
            e_graph: np.array, the E graph
            src_expand_area: np.array, the expand area of src node
            src_idx: int, the src node idx
            ag_src: AreaGraph, the src area graph
            ag_dst: AreaGraph, the dst area graph
            self.global_energy_weights: list, [w_self, w_son, w_parent, w_neighbor]
        Returns:
            final_matched_area: [u_min, v_min, u_max, v_max] if [] means no match
        """
        final_matched_area = []
        if len(matched_list) == 1:
            final_matched_area = ag_dst.AGNodes[matched_list[0]].expand_area
        elif len(matched_list) > 1:
            # get the candis to fuse
            # calc all global energies in the list
            # find the best match with the minimal energy and candis around it to fuse
            global_energy_list = self.calc_global_energy_list(matched_list, e_graph, ag_src, src_idx, ag_dst)
            fused_idxs, fused_energies = self.select_matches_by_global_energy(matched_list, global_energy_list)
            final_matched_area = self.multi_areas_fusion_weighted_from_list(fused_idxs, ag_dst, fused_energies)
            # logger.success(f"find {len(matched_list)} matches, fused")
        else: 
            # no match by GC, select the best match from all by [NOTE] self energy
            candi_idxs = self.garp_candis_by_energy(e_graph)
            final_matched_area = self.multi_areas_fusion_weighted(candi_idxs, ag_dst, e_graph)
            # if len(final_matched_area) > 0 : logger.success(f"no match by GC, but find match by candis")

        return final_matched_area

    def select_matches_by_global_energy(self, matched_list, global_energy_list):
        """ find the best match with the minimal energy and candis around it to fuse
        Args:
            self.global_energy_candi_range: float, the range to find candis
        """
        # find the best match
        min_energy = 1
        min_idx = -1
        for idx, energy in zip(matched_list, global_energy_list):
            if energy < min_energy:
                min_energy = energy
                min_idx = idx
        # find the candis around the best match
        candi_idxs = []
        candi_energy_list = []
        candi_idxs.append(min_idx)
        candi_energy_list.append(min_energy)
        for idx, energy in zip(matched_list, global_energy_list):
            if abs(energy - min_energy) < self.global_energy_candi_range:
                candi_idxs.append(idx)
                candi_energy_list.append(energy)

        return candi_idxs, candi_energy_list
    
    def multi_areas_fusion_weighted_from_list(self, matched_list, ag_dst, fused_energies):
        """ only use weighted fusion for multi areas in the list
            difference is energy is from the list
        """
        if len(matched_list) == 0:
            return []
        if len(matched_list) == 1:
            return ag_dst.AGNodes[matched_list[0]].expand_area

        # get the energies as the weights, smaller energy, bigger weight
        area_weights_list = []
        weights = []
        for i, idx in enumerate(matched_list):
            area_weights_list.append(1 - fused_energies[i])
        area_weights_list = np.array(area_weights_list)
        # normalize weights to sum to 1
        weights = area_weights_list / np.sum(area_weights_list)
        # logger.info(f"weights: {weights}")

        # weighted fusion
        # calc the fusion center
        area_centers = []
        area_sizes = [] # [[w/2, h/2], ...]
        for idx in matched_list:
            expand_area = ag_dst.AGNodes[idx].expand_area
            area_centers.append([(expand_area[0] + expand_area[1]) / 2, (expand_area[2] + expand_area[3]) / 2])
            area_sizes.append([(expand_area[1] - expand_area[0]) / 2, (expand_area[3] - expand_area[2]) / 2])

        area_centers = np.array(area_centers)
        area_sizes = np.array(area_sizes)
        # logger.info(f"area centers: {area_centers}")
        # logger.info(f"area sizes: {area_sizes}")
        fusion_center = np.sum(area_centers * weights[:, np.newaxis], axis=0)
        fusion_size = np.sum(area_sizes * weights[:, np.newaxis], axis=0)
        # logger.info(f"fusion center: {fusion_center}")
        # logger.info(f"fusion size: {fusion_size}")

        # calc the fusion area [u_min, u_max, v_min, v_max]
        fusion_area = [fusion_center[0] - fusion_size[0], fusion_center[0] + fusion_size[0], fusion_center[1] - fusion_size[1], fusion_center[1] + fusion_size[1]]

        return fusion_area

    def multi_areas_fusion_weighted(self, matched_list, ag_dst, e_graph):
        """ only use weighted fusion for multi areas
        Returns:
            final_matched_area: [u_min, v_min, u_max, v_max] if [] means no match
        """
        if len(matched_list) == 0:
            return []
        if len(matched_list) == 1:
            return ag_dst.AGNodes[matched_list[0]].expand_area

        # get the energies as the weights, smaller energy, bigger weight
        area_weights_list = []
        weights = []
        for idx in matched_list:
            area_weights_list.append(1 - e_graph[idx, -1])
        area_weights_list = np.array(area_weights_list)
        # normalize weights to sum to 1
        weights = area_weights_list / np.sum(area_weights_list)
        # logger.info(f"weights: {weights}")

        # weighted fusion
        # calc the fusion center
        area_centers = []
        area_sizes = [] # [[w/2, h/2], ...]
        for idx in matched_list:
            expand_area = ag_dst.AGNodes[idx].expand_area
            area_centers.append([(expand_area[0] + expand_area[1]) / 2, (expand_area[2] + expand_area[3]) / 2])
            area_sizes.append([(expand_area[1] - expand_area[0]) / 2, (expand_area[3] - expand_area[2]) / 2])

        area_centers = np.array(area_centers)
        area_sizes = np.array(area_sizes)
        # logger.info(f"area centers: {area_centers}")
        # logger.info(f"area sizes: {area_sizes}")
        fusion_center = np.sum(area_centers * weights[:, np.newaxis], axis=0)
        fusion_size = np.sum(area_sizes * weights[:, np.newaxis], axis=0)
        # logger.info(f"fusion center: {fusion_center}")
        # logger.info(f"fusion size: {fusion_size}")

        # calc the fusion area [u_min, u_max, v_min, v_max]
        fusion_area = [fusion_center[0] - fusion_size[0], fusion_center[0] + fusion_size[0], fusion_center[1] - fusion_size[1], fusion_center[1] + fusion_size[1]]

        return fusion_area
        
    def multi_areas_fusion(self, matched_list, e_graph, src_expand_area, ag_dst):
        """ TODO fusion for multi areas, take the first as the initial area, fusion with others
            1. if IOU > self.iou_fusion_thd, weighted fusion
            2. else, graph-guided interpolation
                - generate new nodes -> calc the energy -> fusion without IOU thd
                - no new nodes generated -> choose the best match with the minimal energy
        Args:
            matched_list: list, matched dst node idxs
            e_graph: np.array, the E graph
            src_expand_area: np.array, the expand area of src node
            ag_dst: AreaGraph, the dst area graph
        Returns:
            final_matched_area: [u_min, v_min, u_max, v_max] if [] means no match
        """
        if len(matched_list) == 0:
            return []
        if len(matched_list) == 1:
            return ag_dst.AGNodes[matched_list[0]].expand_area

        final_matched_area = []
        

        return final_matched_area

    def fusion_two_areas(self, idx0, idx1, src_expand_area, ag_dst):
        """ TODO fusion two areas by different approaches accoding to iou
            1. if IOU > self.iou_fusion_thd, weighted fusion
            2. else, graph-guided interpolation
                - generate new nodes -> calc the energy -> fusion without IOU thd
                - no new nodes generated -> choose the best match with the minimal energy
        Funs:
            - calc iou
            - weighted fusion for two areas
            - graph-guided interpolation for two areas
            - calc matching energy
        Args:
            idx0: int, the idx of node0
            idx1: int, the idx of node1
            src_expand_area: np.array, the expand area of src node
            ag_dst: AreaGraph, the dst area graph
        Returns:
        """

    def garp_candis_by_energy(self, e_graph):
        """ get the candidate areas by energy e_graph[dst_idx, -1] <= self.candi_energy_thd
    
        """
        candi_area_idxs = []
        for idx in range(e_graph.shape[0]-1):
            if e_graph[idx, -1] <= self.candi_energy_thd:
                candi_area_idxs.append(idx) 
        
        return candi_area_idxs

    def calc_activity_map(self, reverse=False):
        """ No need for symmetric calculation
        Returns:
            activity_map: np.array, the activity map
                size = (src_nodes_num, dst_nodes_num) HxW
                activity_map[i, j] = similarity of ag_src[i] and ag_dst[j]
        """
        if not reverse:
            ag_src = self.areagraph0
            ag_dst = self.areagraph1
        else:
            ag_src = self.areagraph1
            ag_dst = self.areagraph0

        ag_src.expand_each_node()
        ag_dst.expand_each_node()
    
        src_nodes_num = len(ag_src.AGNodes) 
        dst_nodes_num = len(ag_dst.AGNodes) 
        # logger.info(f"start calc activity map with src shape {src_nodes_num} and dst shape {dst_nodes_num}")
        activity_map = np.zeros((src_nodes_num, dst_nodes_num))
        calced_map = np.zeros((src_nodes_num, dst_nodes_num))

        if self.fast_version == 0:
            # slow version: self.fast_version = 0
            for node_idx in tqdm(range(src_nodes_num), ncols=80):
                self.match_given_node(node_idx, ag_src, ag_dst, activity_map, calced_map, draw_flag=0)
                calced_map[node_idx, :] = 1
                self.calced_map_pooling(calced_map, activity_map, ag_src, node_idx)
        elif self.fast_version == 1:
            # fast version -- level-based move: self.fast_version = 1
            for level in tqdm(range(ag_src.level_num+1), ncols=80):
                node_list = ag_src.get_nodes_with_level(level)
                for node_idx in node_list:
                    self.match_given_node(node_idx, ag_src, ag_dst, activity_map, calced_map, draw_flag=0)
                    calced_map[node_idx, :] = 1
                    self.calced_map_pooling(calced_map, activity_map, ag_src, node_idx)
        elif self.fast_version == 2:
            # faster version -- level-based move: self.fast_version = 2 & parallel
            for level in tqdm(range(ag_src.level_num+1), ncols=80):
                node_list = self.choose_nodes(ag_src, level)
                self.match_given_node_parallel(node_list, ag_src, ag_dst, activity_map, calced_map, draw_flag=0)
                for node_idx in node_list:
                    calced_map[node_idx, :] = 1
                    self.calced_map_pooling(calced_map, activity_map, ag_src, node_idx)

        if self.matcher_name == "PATS":
            # normalize the activity map
            activity_map = (activity_map - np.min(activity_map)) / (np.max(activity_map) - np.min(activity_map))

        # logger.success(f"activity map calc done")
        # logger.success(f"activity map shape is {activity_map.shape}")
        # logger.success(f"activity map is\n {activity_map}")

        # normalize the activity map
        activity_map = (activity_map - np.min(activity_map)) / (np.max(activity_map) - np.min(activity_map))

        return activity_map

    def calced_map_pooling(self, calced_map, activity_map, ag_src, node_idx):
        """ pooling the activity map by ag_src
            if activity_map[i, j] ==0: activity_map[next_level_sons_of_i, j] = 0
        """
        for i in range(activity_map.shape[1]):
            if activity_map[node_idx, i] == 0:
                sons_idx_list = ag_src.rt_next_level_sons(node_idx)
                for son_idx in sons_idx_list:
                    activity_map[son_idx, i] = 0
                    calced_map[son_idx, i] = 1
    
    def match_given_node(self, node_idx, ag_src, ag_dst, activity_map, calced_map, draw_flag=0):
        """ match the given node
            ag_src[node_idx] -> ag_dst[?]
            *update the activity map*
        Args:
            activity_map: np.array, the activity map
                size = (src_nodes_num, dst_nodes_num) HxW
                activity_map[i, j] = similarity of ag_src[i] and ag_dst[j]
        """
        # logger.info(f"match given node {node_idx}")
        src_area_img = ag_src.get_node_area_img(node_idx)

        # if draw_flag == 1:
        #     out_img_path = os.path.join(self.out_path, f"match_given_node_{node_idx}.jpg")
        #     cv2.imwrite(out_img_path, src_area_img)
        
        # calc the activity of the src node to each dst node according to the condition probability
        # self.calc_activity_by_condition_prob(node_idx, src_area_img, ag_dst, activity_map, draw_flag=draw_flag) # get the updated activity map
        self.calc_activity_by_1st_order_condition_prob(node_idx, src_area_img, ag_dst, activity_map, calced_map, draw_flag=draw_flag) # get the updated activity map

    def match_given_node_parallel(self, node_list, ag_src, ag_dst, activity_map, calced_map, draw_flag=0):
        """ match the given node -> parallel
            ag_src[node_idx] -> ag_dst[?]
            *update the activity map*
        """    
        src_area_img_list = []
        for node_idx in node_list:
            src_area_img_list.append(ag_src.get_node_area_img(node_idx))
        src_area_img_list = np.array(src_area_img_list)
        src_area_img_list = src_area_img_list.astype(np.float32)
        src_area_img_list = torch.from_numpy(src_area_img_list).cuda()
        src_area_img_list = src_area_img_list.unsqueeze(1)
        src_area_img_list = src_area_img_list / 255.0

        # calc the activity of the src node to each dst node according to the condition probability
        self.calc_activity_by_condition_prob_parallel(node_list, src_area_img_list, ag_dst, activity_map, calced_map, draw_flag=draw_flag)

    def calc_activity_by_condition_prob_parallel(self, node_list, src_area_img_list, ag_dst, activity_map, calced_map, draw_flag=0):
        """ FIXME: parallel version in CUDA
        """
        level_num = ag_dst.level_num
        filterd_node_list = []

    def calc_activity_by_1st_order_condition_prob(self, src_node_idx, src_node_img, ag_dst, activity_map, calced_map, draw_flag=0):
        """ only filter the next level son, if node's activity = 0
        """
        level_num = ag_dst.level_num
        filterd_node_list = []

        for i in range(level_num+1):
            node_list = ag_dst.get_nodes_with_level(i)
            for node_id in node_list:
                if calced_map[src_node_idx, node_id] == 1:
                    # logger.info(f"node {node_id} is calced")
                    continue
                if node_id in filterd_node_list:
                    # logger.info(f"node {node_id} is filtered")
                    continue
                temp_node_img = ag_dst.get_node_area_img(node_id)
                sigma0, sigma1 = self.coarse_match_for_activity(src_node_img, temp_node_img, draw_flag=draw_flag, name=f"{src_node_idx}_{node_id}")
                activity_map[src_node_idx, node_id] = sigma0 * sigma1
                # logger.info(f"match between src node {src_node_idx} and dst node {node_id} is {sigma0 * sigma1}")

                # condition probability in dst ag
                if sigma0 * sigma1 == 0:
                    sons_idx_list = ag_dst.rt_next_level_sons(node_id)
                    filterd_node_list += sons_idx_list

    def calc_activity_by_condition_prob(self, src_node_idx, src_node_img, ag_dst, activity_map, draw_flag=0):
        """
        """
        root_node_idx = ag_dst.get_root_node_idx()
        next_nodes = []
        next_nodes.append(root_node_idx)

        while len(next_nodes) > 0:
            node_idx = next_nodes.pop()
            temp_node_img = ag_dst.get_node_area_img(node_idx)
            sigma0, sigma1 = self.coarse_match_for_activity(src_node_img, temp_node_img, draw_flag=draw_flag, name=f"{src_node_idx}_{node_idx}")
            activity_map[src_node_idx, node_idx] = sigma0 * sigma1
            logger.info(f"match between src node {src_node_idx} and dst node {node_idx} is {sigma0 * sigma1}")

            if sigma0 * sigma1 != 0:
                son_idx_list = ag_dst.rt_all_sons(node_idx)
                next_nodes += son_idx_list
                # remove the repeat nodes
                next_nodes = list(set(next_nodes))
            else:
                son_idx_list = ag_dst.rt_all_sons(node_idx) 
                for son_idx in son_idx_list:
                    activity_map[src_node_idx, son_idx] = 0
                # remove the node in both next_nodes and son_idx_list
                next_nodes = list(set(next_nodes) - set(son_idx_list))

    def coarse_match_for_activity(self, src_node_img, dst_node_img, draw_flag, name=""):
        """
        """
        sigma0, sigma1 = self.area_matcher.match_ret_activity(src_node_img, dst_node_img, sigma_thd=self.sigma_thd,  draw_match=draw_flag, name=name)
        return sigma0, sigma1
    
    
    """Common Utilites==============================================================================="""

    def draw_matched_area_pair(self, src_ag, dst_ag, src_idx, dst_idx, name=""):
        """
        """
        self.ori_img_load()
        src_area_draw = src_ag.AGNodes[src_idx].area
        dst_area_draw = dst_ag.AGNodes[dst_idx].area
        color = (0, 255, 0)
        if self.reverse_flag:
            draw_matched_area(self.img1, self.img0, src_area_draw, dst_area_draw, color, self.out_path, f"r_{name}{src_idx}", f"r_{name}{dst_idx}", save=True)
        else:
            draw_matched_area(self.img0, self.img1, src_area_draw, dst_area_draw, color, self.out_path, f"{name}{src_idx}", f"{name}{dst_idx}", save=True)

    def draw_matched_area_pair_with_mkpts(self, src_area, dst_area, mkpts0, mkpts1, name0, name1, src_offset, dst_offset):
        """ draw the matched area pair with the matched keypoints
        """
        self.ori_img_load()
        color = (0, 255, 0)

        # add the offset
        src_area = [src_area[0] + src_offset[0], src_area[1] + src_offset[0], src_area[2] + src_offset[1], src_area[3] + src_offset[1]]
        dst_area = [dst_area[0] + dst_offset[0], dst_area[1] + dst_offset[0], dst_area[2] + dst_offset[1], dst_area[3] + dst_offset[1]]

        # add the offset to the mkpts
        mkpts0 = [[mkpt[0] + src_offset[0], mkpt[1] + src_offset[1]] for mkpt in mkpts0]
        mkpts1 = [[mkpt[0] + dst_offset[0], mkpt[1] + dst_offset[1]] for mkpt in mkpts1]
        mkpts0 = np.array(mkpts0)
        mkpts1 = np.array(mkpts1)

        if self.reverse_flag:
            draw_matched_area_with_mkpts(self.img1, self.img0, src_area, dst_area, mkpts0, mkpts1, color, self.out_path, "r_"+name0, "r_"+name1, save=True)
        else:
            draw_matched_area_with_mkpts(self.img0, self.img1, src_area, dst_area, mkpts0, mkpts1, color, self.out_path, name0, name1, save=True)

    def collect_pts_within_area(self, area, mkpts0, mkpts1):
        """ collect the matched points within the area
        Args:
            area: [u_min, u_max, v_min, v_max]
            mkpts0: np.array, matched keypoints in src area
            mkpts1: np.array, matched keypoints in dst area
        Returns:
            pts0: np.array, matched keypoints in src area
            pts1: np.array, matched keypoints in dst area
        """
        pts0 = []
        pts1 = []
        for i in range(len(mkpts0)):
            if mkpts0[i][0] >= area[0] and mkpts0[i][0] <= area[1] and mkpts0[i][1] >= area[2] and mkpts0[i][1] <= area[3]:
                pts0.append(mkpts0[i])
                pts1.append(mkpts1[i])

        pts0 = np.array(pts0)
        pts1 = np.array(pts1)

        return pts0, pts1

    def _ret_area_to_ori_img(self, area, offset):
        """ return the area info to the original image
        """
        area = [area[0] + offset[0], area[1] + offset[0], area[2] + offset[1], area[3] + offset[1]]
        return area

    def cal_the_overlap_ratio(self, area0, area1):
        """ calc the overlap ratio of area0 and area1
        Args:
            area0: [u_min, u_max, v_min, v_max]
            area1: [u_min, u_max, v_min, v_max]
        Returns:
            overlap_src_ratio: overlap / area0 size
            overlap_dst_ratio: overlap / area1 size
            overlap_part
        """    
        # calc the size of overlap area
        overlap_area, overlap_part = self.calc_the_overlap_size(area0, area1)

        # calc the size of area0 and area1
        area0_size = (area0[1] - area0[0]) * (area0[3] - area0[2])
        area1_size = (area1[1] - area1[0]) * (area1[3] - area1[2])

        # calc the overlap ratio
        overlap_src_ratio = overlap_area / (area0_size + 1e-6)
        overlap_dst_ratio = overlap_area / (area1_size + 1e-6)

        return overlap_src_ratio, overlap_dst_ratio, overlap_part

    def calc_the_overlap_size(self, area0, area1):
        """ calc the size of overlap area
        Args:
            area0: [u_min, u_max, v_min, v_max]
            area1: [u_min, u_max, v_min, v_max]
        Returns:
            overlap_area: int, the size of overlap area
            overlap_part: [u_min, u_max, v_min, v_max]
            
        """
        u_min0, u_max0, v_min0, v_max0 = area0
        u_min1, u_max1, v_min1, v_max1 = area1
        overlap_part = []

        # calc the overlap area
        u_min = max(u_min0, u_min1)
        u_max = min(u_max0, u_max1)
        v_min = max(v_min0, v_min1)
        v_max = min(v_max0, v_max1)

        # calc the size of overlap area
        overlap_area = max((u_max - u_min), 0) * max((v_max - v_min), 0)
        if overlap_area > 0 :
            overlap_part = [u_min, u_max, v_min, v_max]

        return overlap_area, overlap_part
        
    def fuse_area_mind_size(self, v_area, area_list, beyong_size_ratio=0.3):
        """ collect valid fuse candi and then fuse them
            NOTE mind the beyond size
        Args:
            v_area: [u_min, u_max, v_min, v_max]
            area_list: [[u_min, u_max, v_min, v_max], ...]
        Returns:
            fused_area: [u_min, u_max, v_min, v_max]
        """

        # collect
        fuse_candis = []
        for area in area_list:
            beyond_size = self.calc_beyond_size(v_area, area)
            if beyond_size / (area[1] - area[0]) / (area[3] - area[2]) <= beyong_size_ratio:
                fuse_candis.append(area)

        # fuse 
        fused_area = self.area_fusion(v_area, fuse_candis)

        return fused_area

    def calc_beyond_size(self, v_area, fuse_candi):
        """ calc the beyond size of the fuse_candi
            first calc the overlap area, then calc the beyond size using the area - overlap
        """
        overlap_area, overlap_part = self.calc_the_overlap_size(v_area, fuse_candi)
        beyond_size = (fuse_candi[1] - fuse_candi[0]) * (fuse_candi[3] - fuse_candi[2]) - overlap_area
        return beyond_size

    def area_fusion(self, v_area, area_list):
        """ fuse the area list and the v_area into a single area
        Args:
            v_area: [u_min, u_max, v_min, v_max]
            area_list: [[u_min, u_max, v_min, v_max], ...]
        Returns:
            fused_area: [u_min, u_max, v_min, v_max]
        """
        W, H = self.W, self.H

        temp_area_list = area_list.copy()
        temp_area_list.append(v_area)
        fused_area = [W, 0, H, 0]
        
        for area in temp_area_list:
            fused_area[0] = min(fused_area[0], area[0])
            fused_area[1] = max(fused_area[1], area[1])
            fused_area[2] = min(fused_area[2], area[2])
            fused_area[3] = max(fused_area[3], area[3])
        
        return fused_area

    def insert_fused_area_to_ag(self, fused_area, ag, related_node_idx_list):
        """ Updated the exsiting area node or insert a new area node to the area graph
        Args:
            fused_area: [u_min, u_max, v_min, v_max]
            ag: area graph
            related_node_idx_list: the related node idx list
        Returns:
            fused_area_node_idx
        """
        fused_area_node_idx = 1e6
        # construct the fused area node
        fused_area_info = {}
        fused_area_info["area_bbox"] = fused_area
        fused_area_info["area_size"] = (fused_area[1] - fused_area[0]) * (fused_area[3] - fused_area[2])
        fused_area_info['mask'] = None
        fused_area_info['area_center'] = [(fused_area[0] + fused_area[1]) / 2, (fused_area[2] + fused_area[3]) / 2]

        # construct the fused area node
        fused_area_node = AGNode(fused_area_info)

        # determine the level of fused area node
        size_level_thd_list = ag.size_level_thd_list
        level = fused_area_node.asign_level(size_level_thd_list)

        # determine if exsit the similar enough area node
        similar_flag = False
        similar_idx = 1e6
        for node_idx in related_node_idx_list:
            similar_candi = ag.AGNodes[node_idx]
            candi_area = similar_candi.area
            dist_temp = self.calc_two_area_dist(fused_area, candi_area)
            if dist_temp <= self.similar_area_dist_thd:
                # logger.info(f"area dist = {dist_temp} < similar_area_dist_thd = {self.similar_area_dist_thd}")
                # logger.info(f"find area matched with idx {node_idx}")
                similar_flag = True
                similar_idx = node_idx
                break
        
        # if exsit the similar enough area node, update the area node
        if similar_flag:
            assert similar_idx != 1e6
            ag.update_node_by_fuse_area(similar_idx, fused_area)
            fused_area_node_idx = similar_idx
        else:
            # if not exsit the similar enough area node, insert the new area node
            # logger.info(f"not find area matched, add new area node")
            fused_area_node_idx = ag.append_new_node(fused_area_node)

        return fused_area_node_idx

    def calc_two_area_dist(self, area0, area1):
        """ calc the mean dist of the ul and rd pts between two nodes
        Args:
            area0: [u_min, u_max, v_min, v_max]
            area1: [u_min, u_max, v_min, v_max]
        Returns:
            dist: float
        """

        ul0 = [area0[0], area0[2]]
        ul1 = [area1[0], area1[2]]
        dist_ul = np.linalg.norm(np.array(ul0) - np.array(ul1))

        rd0 = [area0[1], area0[3]]
        rd1 = [area1[1], area1[3]]
        dist_rd = np.linalg.norm(np.array(rd0) - np.array(rd1))

        dist = (dist_ul + dist_rd) / 2

        return dist

    def remove_repeat_area_matches(self, matched_src_areas, matched_dst_areas):
        """ remove the repeat area matches
        Flow:
            
        Args:
            matched_src_areas: [areas]
                area: [u_min, u_max, v_min, v_max]
            matched_dst_areas: [areas]
        Returns:
            filtered_matched_src_areas: [areas]
            filtered_matched_dst_areas: [areas]
        """
        assert len(matched_src_areas) == len(matched_dst_areas)
        
        filtered_matched_src_areas = []
        filtered_matched_dst_areas = []

        for src_area, dst_area in zip(matched_src_areas, matched_dst_areas):
            similar_src_flag, similar_src_idx = self.decide_area_similarity_in_area_list(src_area, filtered_matched_src_areas)
            similar_dst_flag, similar_dst_idx = self.decide_area_similarity_in_area_list(dst_area, filtered_matched_dst_areas)

            if not similar_src_flag and not similar_dst_flag:
                filtered_matched_src_areas.append(src_area)
                filtered_matched_dst_areas.append(dst_area)
            else:
                # update the area in the filtered_matched_src_areas with idx = similar_src_idx
                if similar_src_flag:
                    filtered_matched_src_areas[similar_src_idx] = self.update_area(filtered_matched_src_areas[similar_src_idx], src_area)
                    filtered_matched_dst_areas[similar_src_idx] = self.update_area(filtered_matched_dst_areas[similar_src_idx], dst_area)
                
                # update the area in the filtered_matched_dst_areas with idx = similar_dst_idx
                if similar_dst_flag:
                    filtered_matched_src_areas[similar_dst_idx] = self.update_area(filtered_matched_src_areas[similar_dst_idx], src_area)
                    filtered_matched_dst_areas[similar_dst_idx] = self.update_area(filtered_matched_dst_areas[similar_dst_idx], dst_area)
            
        
        return filtered_matched_src_areas, filtered_matched_dst_areas

    def update_area(self, target_area, area_candi):
        """ update the target area by the area candi, fuse the two areas
        """
        fused_area = self.fuse_two_area(target_area, area_candi)

        return fused_area

    def fuse_two_area(self, area0, area1):
        """ fuse two areas
        Args:
            area0: [u_min, u_max, v_min, v_max]
            area1: [u_min, u_max, v_min, v_max]
        Returns:
            fused_area: [u_min, u_max, v_min, v_max]
        """
        fused_area = [0, 0, 0, 0]

        fused_area[0] = min(area0[0], area1[0])
        fused_area[1] = max(area0[1], area1[1])
        fused_area[2] = min(area0[2], area1[2])
        fused_area[3] = max(area0[3], area1[3])

        return fused_area
    
    def decide_area_similarity_in_area_list(self, area, area_list):
        """ decide if the area is similar enough with the area_list
        Args:
            area: [u_min, u_max, v_min, v_max]
            area_list: [areas]
        Returns:
            similar_flag: bool
            similar_idx: int
        """
        similar_flag = False
        similar_idx = 1e6

        for idx, area_temp in enumerate(area_list):
            dist_temp = self.calc_two_area_dist(area, area_temp)
            if dist_temp <= self.similar_area_dist_thd:
                similar_flag = True
                similar_idx = idx
                break
        
        return similar_flag, similar_idx




    