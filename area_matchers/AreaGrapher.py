'''
Author: EasonZhang
Date: 2023-05-15 14:18:31
LastEditors: EasonZhang
LastEditTime: 2024-06-20 23:32:52
FilePath: /SA2M/hydra-mesa/area_matchers/AreaGrapher.py
Descriptin: Construct an area graph of an image
Copyright (c) 2023 by EasonZhang, All Rights Reserved. 
'''

import sys
sys.path.append("..")

import os.path as osp
import os
import numpy as np 
from loguru import logger
import cv2
import copy

from .AGBasic import AGNode, area_info_temp, AdjMat
from .AreaPreprocessor import AreaPreprocesser
from .AGUtils import KMCluster, AGViewer, MaskViewer
from .AGConfig import areagraph_configs

from utils.common import expand_mat_by1, clean_mat_idx



class AreaGraph(object):
    """ Main Class
    Args:
        AdjMat
        AGNodes: a list of AGNode
    """
    dft_configs = areagraph_configs

    def __init__(self, configs, efficient=False, build_init=True) -> None:
        """ initialization the area graph from sam_res
            get the adjacent matrix and the AGNodes -> area graph
        Args:
            configs: dict = {
                'preprocesser_config': {},
                'sam_res_path' = "",
                'sem_seg_path' = "",
                'W': 640,
                'H': 480,
                'save_path': '',
                'ori_img_path': '',
                'fs_overlap_thd': 0.8, # the overlap threshold of father and son
                'level_num': 4, # the number of level
                'level_step': [640, 480, 256, 100, 0], # the step of level
                # 'size_level_thd_list': [640*640, 480*480, 256*256, 100*100, 0], # the area size level threshold
                    - level i: size_level_thd_list[i] < area_size < size_level_thd_list[i-1]
            }
        """
        configs = {**self.dft_configs, **configs}
        logger.info(f"configs: {configs}")
        
        self.adj_mat = None
        self.ori_img = None
        self.root_idx = None
        self.AGNodes = []
        self.adj_mat = None
        self.W = configs['W'] # the width of the image
        self.H = configs['H'] # the height of the image
        self.ori_img_path = configs['ori_img_path']
        self.save_path = configs['save_path']
        self.graph_viewer = AGViewer(self.W, self.H, self.save_path)
        self.draw_flag = configs['show_flag']
        self.IOU_mat = None
        self.overlap_info = None # overlap info, described latter

        # param
        self.level_num = configs['level_num']
        self.level_step = configs['level_step']
        assert len(self.level_step) == self.level_num + 1, 'the length of level_step should be level_num + 1'
        self.size_level_thd_list = [level_step**2 for level_step in self.level_step]

        self.fs_overlap_thd = configs['fs_overlap_thd']

        # load the area preprocesser
        self.AreaPreprocesser = AreaPreprocesser(configs['preprocesser_config'])

        # load the cluster
        self.KMCluster = KMCluster(save_path=self.save_path)

        self.seg_source = configs['preprocesser_config']['seg_source']

        if build_init:
            if self.seg_source == 'SAM':
                # load the sam res
                sam_res_path = configs['sam_res_path']
                assert os.path.exists(sam_res_path), f"the {sam_res_path} is not exist"
                seg_res = np.load(sam_res_path, allow_pickle=True)
            elif self.seg_source == 'Sem':
                sem_res_path = configs['sem_res_path']
                assert os.path.exists(sem_res_path), f"the {sem_res_path} is not exist"
                seg_res = cv2.imread(sem_res_path, cv2.IMREAD_UNCHANGED)
            
            self.area_info_list = self._area_preprocess(res=seg_res, draw_flag=self.draw_flag)

            # load img
            self.load_ori_img()

            # build the initial area graph
            if efficient:
                self._build_efficient(area_info_list=self.area_info_list)
            else:
                self._build(area_info_list=self.area_info_list) # get adjMat and AGNodes

            # show
            if configs['show_flag'] == 1:
                # logger.info(f"save_path: {self.save_path}")
                self.show_graph_with_img(True)
    
    def build_area_graph(self, img_path, seg_path, efficient=True, show_flag=False):
        """
        """
        # load the sam res
        assert os.path.exists(seg_path), f"the {seg_path} is not exist"
        if self.seg_source == 'SAM':
            seg_res = np.load(seg_path, allow_pickle=True)
        elif self.seg_source == 'Sem':
            seg_res = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        self.area_info_list = self._area_preprocess(res=seg_res, draw_flag=self.draw_flag)

        # load img
        self.ori_img_path = img_path
        self.load_ori_img()

        if efficient:
            self._build_efficient(area_info_list=self.area_info_list)
        else:
            self._build(area_info_list=self.area_info_list) # get adjMat and AGNodes

        self.expand_each_node()

        # show
        if show_flag:
            # logger.info(f"save_path: {self.save_path}")
            self.show_graph_with_img(True)

    # init graph construction ================================================================================================================
    def _area_preprocess(self, res, draw_flag):
        """ use the AreaPreprocesser to preprocess the sam_res
        """
        self.AreaPreprocesser.refined_areas = None
        self.AreaPreprocesser.load(res=res)
        self.AreaPreprocesser.refine_bbox(draw_flag=draw_flag)
        self.AreaPreprocesser.filter_abnormal_areas(draw_flag=draw_flag)

        return self.AreaPreprocesser.refined_areas

    def _build_efficient(self, area_info_list):
        """ efficient AG build
        Flow:
            1. for each area_info, initial an AGNode
            2. each AGNode gets in, calc the overlap with the existing nodes
            3. fill the overlap_info matrix and construct the adjacent matrix
                - overlap_info: a matrix of shape (node_num, node_num)
                    overlap_info[i, j] = size_{i intersect j} / size_{i}
                    overlap_info[j, i] = size_{i intersect j} / size_{j}
        """
        # initial the adjacent matrix
        self.adj_mat = AdjMat(init_N=1)
        self.AGNodes = []

        # initial the AGNodes: add node one by one
        for idx, area_info in enumerate(area_info_list):
            self.append_new_node_efficient(area_node=AGNode(area_info=area_info, idx=idx))
        
        # clean the adjacent matrix
        self.adj_mat.clean_mat_value(AGNodes=self.AGNodes)

        self._fix_same_level_fathers(efficient=True)

        # complete the init graph
        self._complete_init_graph(efficient=True)

        # log the graph info
        # self.log_graph_info()
    
    def log_graph_info(self):
        """ log the graph info
        """
        logger.success(f"the number of nodes: {len(self.AGNodes)}")
        logger.success(f"the shape of adj_mat: {self.adj_mat.mat.shape}")
        logger.success(f"the shape of overlap_info: {self.overlap_info.shape}")
        logger.success(f"the idx of root node: {self.root_idx}")
        logger.success(f"the level of root node: {self.AGNodes[self.root_idx].level}")
    
    def append_new_node_efficient(self, area_node, root=False):
        """
        """
        node_idx = len(self.AGNodes)
        self.AGNodes.append(area_node)
        if root:
            self.AGNodes[node_idx].level = 0
        else:
            self.AGNodes[node_idx].asign_level(size_level_thd_list=self.size_level_thd_list)
        area_node.idx = node_idx
        if node_idx > 0:
            self.adj_mat.append_node()
        self.overlap_info = expand_mat_by1(self.overlap_info)
        flag = self.update_for_a_node_efficient(node_idx=node_idx, root=root)
        if not flag:
            self.AGNodes.pop()
            self.adj_mat.pop_node()
            self.overlap_info = clean_mat_idx(self.overlap_info, node_idx)
            return False
        return True
        
    def update_for_a_node_efficient(self, node_idx, root=False):
        """ add a new node, update the overlap_info and adjacent matrix
        """
        node_num = len(self.AGNodes)

        for idx in range(node_num):
            if idx == node_idx:
                continue
            overlap_flag, overlap_ratio_s, overlap_ratio_d = self.AGNodes[node_idx].overlap_check_complete(dst_node=self.AGNodes[idx], fs_overlap_thd=self.fs_overlap_thd)
            
            if root:
                overlap_flag = 1
            
            if overlap_flag == -2:
                # repeat node
                return False
            elif overlap_flag == 0:
                # no overlap
                continue
            elif overlap_flag == 1:
                # node_idx is the father of idx
                self.adj_mat.mat[node_idx, idx] = 2
                self.overlap_info[node_idx, idx] = overlap_ratio_s
                self.overlap_info[idx, node_idx] = overlap_ratio_d
            elif overlap_flag == -1:
                # node_idx is the son of idx
                self.adj_mat.mat[idx, node_idx] = 2
                self.overlap_info[node_idx, idx] = overlap_ratio_s
                self.overlap_info[idx, node_idx] = overlap_ratio_d
            elif overlap_flag == 2:
                # node_idx and idx are neighbors
                self.adj_mat.mat[node_idx, idx] = 1
                self.adj_mat.mat[idx, node_idx] = 1
                self.overlap_info[node_idx, idx] = overlap_ratio_s
                self.overlap_info[idx, node_idx] = overlap_ratio_d
            
        return True

    def _build(self, area_info_list):
        """
        Flow:
            1. for each area_info, initial an AGNode
            2. for each AGNode, acroding to the bbox overlap, determine the adjacency matrix
        Args:
            area_info_list: a list of area_info
        """
        # initial the adjacent matrix
        self.adj_mat = AdjMat(init_N=len(area_info_list))

        # initial the AGNodes
        for idx, area_info in enumerate(area_info_list):
            self.AGNodes.append(AGNode(area_info=area_info, idx=idx))
            # asign level
            self.AGNodes[idx].asign_level(size_level_thd_list=self.size_level_thd_list)

        # build the adjacent matrix
        for idx, node in enumerate(self.AGNodes):
            for idx_dst, node_dst in enumerate(self.AGNodes):
                if idx == idx_dst:
                    continue
                overlap_flag, overlap_percentage = node.overlap_check(dst_node=node_dst)

                # neighbor
                if overlap_flag and overlap_percentage < self.fs_overlap_thd:
                    self.adj_mat.mat[idx, idx_dst] = 1       

                # son
                if overlap_flag and overlap_percentage >= self.fs_overlap_thd:
                    self.adj_mat.mat[idx_dst, idx] = 2

        # clean the adjacent matrix
        self.adj_mat.clean_mat_value(AGNodes=self.AGNodes)

        # fix the same level fathers --> level up 
        self._fix_same_level_fathers()

        # complete the init graph
        self._complete_init_graph()

    def _fix_same_level_fathers(self, efficient=False):
        """ fix the same level fathers
        Flow:
            1. fro each level nodes, get their fathers
            2. record the fathers at the same level
            3. level up these fathers
                - expand the AGNode.bbox
                - update the AGNode.level
                - update the AGNode.info
                - update the AdjMat
        Args:
            self.AGNodes
            self.adj_mat 
        """
        for node in self.AGNodes:
            same_level_fathers = []
            if node.level == 1:
                # change father-son to neighbor
                father_idxs = self.adj_mat.get_fathers(self.adj_mat.mat, node.idx)
                if len(father_idxs) == 0:
                    continue
                for father_idx in father_idxs:
                    self.adj_mat.mat[node.idx, father_idx] = 1
                    self.adj_mat.mat[father_idx, node.idx] = 1
            else:
                father_idxs = self.adj_mat.get_fathers(self.adj_mat.mat, node.idx)
                if len(father_idxs) == 0:
                    continue
                # record the same level fathers
                for father_idx in father_idxs:
                    if self.AGNodes[father_idx].level == node.level:
                        same_level_fathers.append(father_idx)
                # level up the same level fathers
                for father_idx in same_level_fathers:
                    # logger.info(f"level up the node: {father_idx}")
                    if efficient:
                        self.level_up_a_node_remain_old_efficient(target_idx=father_idx)
                    else:
                        self.level_up_a_node(target_idx=father_idx)
    
    def level_up_a_node(self, target_idx):
        """ level up the node: self.AGNodes[target_idx]
            first up the node level
            second update the node info
            - update the bbox
            - update the area
            - update the center
            Third update the adjacent matrix
        Args:
            self.AGNodes
            self.adj_mat
        """
        # up the node level
        if self.AGNodes[target_idx].level_up():
            # update the node info
            self._update_node_by_bbox(AGNode=self.AGNodes[target_idx])
            # update the adjacent matrix
            self.update_adjMat_for_a_node(node_idx=target_idx)
        else:
            logger.warning('ERROR')

    def level_up_a_node_remain_old_efficient(self, target_idx):
        """
        """
        # create a new node, copy the old node info
        old_node = self.AGNodes[target_idx]
        new_node = old_node.copy_to(idx=len(self.AGNodes))

        if new_node.level_up():
            # update the node info
            self._update_node_by_bbox(AGNode=new_node)
            # add the new node
            flag = self.append_new_node_efficient(area_node=new_node)
            if not flag:
                logger.info('ERROR: update the node failed')

    def level_up_a_node_remain_old(self, target_idx):
        """ level up the node: self.AGNodes[target_idx], preserve the old node
            first up the node level
            second update the node info
            - update the bbox
            - update the area
            - update the center
            Third update the adjacent matrix"""
        # create a new node, copy the old node info
        old_node = self.AGNodes[target_idx]
        new_node = old_node.copy_to(idx=len(self.AGNodes))
        # add the new node
        self.AGNodes.append(new_node)
        # update the adjacent matrix
        self.adj_mat.append_node()
        # update the adjacent matrix
        self.update_adjMat_for_a_node(node_idx=new_node.idx)

        # level up the node
        if new_node.level_up():
            # update the node info
            self._update_node_by_bbox(AGNode=new_node)
            # update the adjacent matrix
            self.update_adjMat_for_a_node(node_idx=new_node.idx)
        else:
            logger.warning('ERROR: No need to update the node')

    def _expand_node_bbox(self, AGNode, target_size):
        """ expand with the same WHRatio
        """
        # info output
        # logger.info(f"expand to size {target_size}")
        # logger.info(f"before expand: {AGNode.W} * {AGNode.H} = {AGNode.size}")

        # get the expand ratio
        expand_ratio = np.sqrt(target_size**2 / AGNode.size)
        # logger.info(f"expand ratio: {expand_ratio}")

        # fix the overbig ratio
        max_u_ratio = self.W / AGNode.W
        max_v_ratio = self.H / AGNode.H
        u_exp_ratio = expand_ratio
        v_exp_ratio = expand_ratio

        if expand_ratio > max_u_ratio and expand_ratio > max_v_ratio:
            u_exp_ratio = max_u_ratio
            v_exp_ratio = max_v_ratio
        elif expand_ratio > max_u_ratio:
            u_exp_ratio = max_u_ratio
            v_exp_ratio = expand_ratio**2 / max_u_ratio
        elif expand_ratio > max_v_ratio:
            v_exp_ratio = max_v_ratio
            u_exp_ratio = expand_ratio**2 / max_v_ratio

        # expand the bbox
        # logger.info(f"expand ratio: {u_exp_ratio} * {v_exp_ratio}")
        AGNode.W *= u_exp_ratio
        AGNode.H *= v_exp_ratio

        try:
            assert AGNode.W <= self.W and AGNode.H <= self.H, 'the expanded bbox is out of the image'
        except AssertionError:
            # logger.error('the expanded bbox is out of the image')
            pass

        # tune the center to ensure the center is in the image
        AGNode.tune_center_and_more(W=self.W, H=self.H)

    def _update_node_by_bbox(self, AGNode):
        """ update the node info by the bbox
        Flow:
            self.level_step[ori_level] <= ori_size < self.level_step[ori_level-1]
            µ = self.level_step[ori_level-1] <= dst_size < self.level_step[ori_level-2]
            - determine the expand ratio
                * expand_ratio**2 = µ*µ / ori_area
            - expand the bbox
                W *= expand_ratio
                H *= expand_ratio
                tune the center, area & bbox
        """
        # already level up
        src_level = AGNode.level + 1
        dst_level = AGNode.level

        # get the dst_size
        dst_size = self.level_step[dst_level]
        self._expand_node_bbox(AGNode=AGNode, target_size=dst_size)
    
    def update_adjMat_for_a_node(self, node_idx):
        """ recalc the target node's relationships with other nodes
            used after node is changed (level up, expand by fusion, etc.)
        """
        # get the target node
        node = self.AGNodes[node_idx]

        # update the adjacent matrix
        for node_dst in (self.AGNodes):
            idx = node_dst.idx
            if idx == node_idx:
                continue
            overlap_flag, overlap_percentage = node.overlap_check(dst_node=node_dst)

            # neighbor
            if overlap_flag and overlap_percentage < self.fs_overlap_thd:
                self.adj_mat.mat[node_idx, idx] = 1
                self.adj_mat.mat[idx, node_idx] = 1

            # father
            if overlap_flag and overlap_percentage >= self.fs_overlap_thd:
                self.adj_mat.mat[idx, node_idx] = 2
            
            # son
            overlap_flag_dst, overlap_percentage_dst = node_dst.overlap_check(dst_node=node)
            if overlap_flag_dst and overlap_percentage_dst >= self.fs_overlap_thd:
                self.adj_mat.mat[node_idx, idx] = 2

        # clean the adjacent matrix
        self.adj_mat.clean_mat_value(AGNodes=self.AGNodes)

    def _complete_init_graph(self, fusion_thd=100, efficient=False):
        """ 
        Args:
            self.adj_mat
            self.AGNodes
        Flow:
            0. from the bottom to top collect nodes without the upper level father
            1. cluster nodes without the upper level father
            2. same cluster -> two nodes fusion or single node level up
                - fusion mind size
        """
        smallest_level = self.level_num
        # from the bottom to top collect nodes without the upper level father
        for level in range(smallest_level, 1, -1):
            # logger.info(f"graph complete - level: {level}")
            # get the nodes without the upper level father
            nodes_without_father = []
            within_level_nodes = [node for node in self.AGNodes if node.level == level]
            for node in within_level_nodes:
                temp_idx = node.idx
                temp_fathers = self.adj_mat.get_fathers(self.adj_mat.mat_simp, temp_idx)
                if len(temp_fathers) == 0:
                    nodes_without_father.append(node)
                else:
                    for father_idx in temp_fathers:
                        temp_flag = False
                        father = self.AGNodes[father_idx]
                        if father.level == level - 1:
                            temp_flag = True
                            break
                    if not temp_flag:
                        nodes_without_father.append(node)
            # cluster nodes without the upper level father
            # logger.info(f"nodes without father: {[node.idx for node in nodes_without_father]}")
            node_num = len(nodes_without_father)
            if node_num == 0:
                continue
            elif node_num == 1:
                node_level_up_idx = nodes_without_father[0].idx
                # self.level_up_a_node(target_idx=node_level_up_idx)
                if efficient:
                    self.level_up_a_node_remain_old_efficient(target_idx=node_level_up_idx)
                else:
                    self.level_up_a_node_remain_old(target_idx=node_level_up_idx)
                # logger.info(f"level up the node: {node_level_up_idx}")
            elif node_num == 2:
                # only two nodes, fusion them if overlap
                temp_node0 = nodes_without_father[0]
                temp_node1 = nodes_without_father[1]
                overlap_flag, overlap_percentage = temp_node0.overlap_check(dst_node=temp_node1)
                if overlap_flag:
                    # fusion
                    self.fusion_two_nodes_get_father(node0=temp_node0, node1=temp_node1, efficient=efficient)
                else:
                    # level up
                    # self.level_up_a_node(target_idx=temp_node0.idx)
                    # self.level_up_a_node(target_idx=temp_node1.idx)
                    if efficient:
                        self.level_up_a_node_remain_old_efficient(target_idx=temp_node0.idx)
                        self.level_up_a_node_remain_old_efficient(target_idx=temp_node1.idx)
                    else:
                        self.level_up_a_node_remain_old(target_idx=temp_node0.idx)
                        self.level_up_a_node_remain_old(target_idx=temp_node1.idx)
                    # logger.info(f"level up the node: {temp_node0.idx}, {temp_node1.idx}")
            else:
                # more than two nodes, cluster them
                
                # get center list
                center_list = [x for x in range(1, node_num+1)]
                self.KMCluster.load_center_list(center_list)

                # get nodes' centers, form as np.ndarray[[u,v],...]
                centers = np.zeros((node_num, 2))
                for idx, node in enumerate(nodes_without_father):
                    centers[idx] = node.center
                
                # cluster
                cluster_num, labels = self.KMCluster.cluste_2d_points(centers, show=False, name=f"level_{level}_cluster")

                label_nodes_dict = {}
                for idx, label in enumerate(labels):
                    if label not in label_nodes_dict:
                        label_nodes_dict[label] = [nodes_without_father[idx]]
                    else:
                        label_nodes_dict[label].append(nodes_without_father[idx])
                
                # fusion or level up
                for label, nodes in label_nodes_dict.items():
                    # under the same level
                    # logger.info(f"cluster label: {label}, nodes: {[node.idx for node in nodes]}")
                    node_num = len(nodes)
                    if node_num == 1:
                        # level up
                        # self.level_up_a_node(target_idx=nodes[0].idx)
                        if efficient:
                            self.level_up_a_node_remain_old_efficient(target_idx=nodes[0].idx)
                        else:
                            self.level_up_a_node_remain_old(target_idx=nodes[0].idx)
                        # logger.info(f"level up the node: {nodes[0].idx}")
                    elif node_num == 2:
                        node0 = nodes[0]
                        node1 = nodes[1]
                        overlap_flag, overlap_percentage = node0.overlap_check(dst_node=node1)
                        if overlap_flag:
                            # fusion
                            self.fusion_two_nodes_get_father(node0=node0, node1=node1, efficient=efficient)
                            # logger.info(f"fusion the node: {node0.idx}, {node1.idx}")
                        else:
                            # level up
                            # self.level_up_a_node(target_idx=node0.idx)
                            # self.level_up_a_node(target_idx=node1.idx)
                            if efficient:
                                self.level_up_a_node_remain_old_efficient(target_idx=node0.idx)
                                self.level_up_a_node_remain_old_efficient(target_idx=node1.idx)
                            else:
                                self.level_up_a_node_remain_old(target_idx=node0.idx)
                                self.level_up_a_node_remain_old(target_idx=node1.idx)
                            # logger.info(f"level up the node: {node0.idx}, {node1.idx}")
                    else:
                        # for each node, find the nearest node and fusion
                        # if nearest dist is larger than the fusion thd, level up
                        fused_node_idx = []
                        for idx, node in enumerate(nodes):
                            if idx in fused_node_idx:
                                continue
                            nearest_dist = 10000000
                            nearest_outside_dist = 10000000
                            nearest_idx = -1
                            for idx_dst, node_dst in enumerate(nodes):
                                if idx == idx_dst:
                                    continue
                                temp_dist, temp_outside_dist = node.get_dist(dst_node=node_dst)
                                if temp_dist < nearest_dist: # aim at center nearest pair
                                    nearest_dist = temp_dist
                                    nearest_idx = idx_dst
                                    nearest_outside_dist = temp_outside_dist
                            if nearest_outside_dist < fusion_thd: # ensure the fusion is reasonable
                                # fusion
                                self.fusion_two_nodes_get_father(node0=node, node1=nodes[nearest_idx], efficient=efficient)
                                # logger.info(f"fusion the node: {node.idx}, {nodes[nearest_idx].idx}")
                                fused_node_idx.append(nearest_idx)

                            else:
                                # level up
                                if efficient:
                                    self.level_up_a_node_remain_old_efficient(target_idx=node.idx)
                                else:
                                    self.level_up_a_node_remain_old(target_idx=node.idx)
                                # logger.info(f"level up the node: {node.idx}")

            # fix same level father 
            self._fix_same_level_fathers()

        # add the root node
        self._add_root_node(efficient=efficient)

    def rt_node_number(self):
        """ return the number of nodes
        """
        assert len(self.AGNodes) == self.adj_mat.mat.shape[0], f"the number of nodes {len(self.AGNodes)} is not equal to the number of rows of the adjacent matrix {self.adj_mat.mat.shape[0]}"
        return len(self.AGNodes)    

    def _add_root_node(self, efficient=False):
        """ add the entire image as the root node
        """
        # if no area
        if len(self.AGNodes) == 0:
            root_area = [0, self.W, 0, self.H]
            root_size = (root_area[1] - root_area[0]) * (root_area[3] - root_area[2])
            root_center = [(root_area[0] + root_area[1]) / 2, (root_area[2] + root_area[3]) / 2]
            root_info = {
                "area_bbox": root_area,
                "area_size": root_size,
                "area_center": root_center,
                "mask": None,
            }
            root_node = AGNode(area_info=root_info, idx=0)
            self.AGNodes.append(root_node)
            self.adj_mat.add_root_node(self.AGNodes)
            self.root_idx = 0
            return

        node_num = self.rt_node_number()
        root_area = [0, self.W, 0, self.H]
        root_size = (root_area[1] - root_area[0]) * (root_area[3] - root_area[2])
        root_center = [(root_area[0] + root_area[1]) / 2, (root_area[2] + root_area[3]) / 2]
        root_info = {
            "area_bbox": root_area,
            "area_size": root_size,
            "area_center": root_center,
            "mask": None,
        }
        root_node = AGNode(area_info=root_info, idx=node_num)
        if efficient:
            flag = self.append_new_node_efficient(area_node=root_node, root=True)
            if not flag:
                logger.error('ERROR: add the new node failed')
        else:
            self.AGNodes.append(root_node)
            self.adj_mat.add_root_node(self.AGNodes)
        self.root_idx = node_num

    def fusion_two_nodes_get_father(self, node0, node1, efficient=False):
        """ generate the common father for two nodes
        Args:
            node0, node1: AGNode
        Flow:
            1. get the common father
                - fusion the areas
            2. add to AGNodes
            3. update the adjacent matrix
        """
        u_min0, u_max0, v_min0, v_max0 = node0.area
        u_min1, u_max1, v_min1, v_max1 = node1.area
        father_area = [min(u_min0, u_min1), max(u_max0, u_max1), min(v_min0, v_min1), max(v_max0, v_max1)]
        father_size = (father_area[1] - father_area[0]) * (father_area[3] - father_area[2])
        father_center = [(father_area[0] + father_area[1]) / 2, (father_area[2] + father_area[3]) / 2]
        father_info = {
            'area_bbox': father_area,
            'area_size': father_size,
            'area_center': father_center,
            'mask': None,
        }
        father_idx = len(self.AGNodes)
        father_node = AGNode(area_info=father_info, idx=father_idx)
        father_node.level = node0.level - 1

        if not efficient:
            # add to AGNodes
            self.AGNodes.append(father_node)

            # update the adjacent matrix
            self.adj_mat.append_node()
            self.update_adjMat_for_a_node(node_idx=father_idx)
        else:
            # add to AGNodes
            flag = self.append_new_node_efficient(area_node=father_node)
            if not flag:
                logger.info('ERROR: add the new node failed')

    def fix_graph_relation(self):
        """ NOTE NOT WORK: done in AdjMat.clean_mat_value
        fix the relationship between nodes 
            1. multiple father -> determine grandfather
            2. neighbor -> assert symmetric
        Args:
            self.adj_mat

        """

    def load_ori_img(self):
        """
        """
        self.ori_img = cv2.imread(self.ori_img_path, cv2.IMREAD_COLOR)
        # resize 
        self.ori_img = cv2.resize(self.ori_img, (self.W, self.H))
        # # turn to RGB, if it is gray
        # if len(self.ori_img.shape) == 2:
        #     self.ori_img = cv2.cvtColor(self.ori_img, cv2.COLOR_GRAY2RGB)

    def show_graph_with_img(self, save=False):
        """draw the graph with the ori_img at each level
        Funcs:
            1. get each level 
            2. draw the graph with node at each level highlighed
            3. draw the node's bbox on the ori_img at each level
        """
        if self.ori_img is None:
            self.load_ori_img()
        if len(self.ori_img.shape) == 2:
            self.ori_img = cv2.cvtColor(self.ori_img, cv2.COLOR_GRAY2RGB)

            
        for level in range(0, self.level_num+1):
            # get the node idx at this level
            level_node_idx_list = [node.idx for node in self.AGNodes if node.level == level]
            # draw the graph
            self.graph_viewer.draw_from_adjMat(adjMat=self.adj_mat.mat, AGNodes=self.AGNodes, level_num=self.level_num, node_dist=2, highlighted_idx_list=level_node_idx_list, name=f"{level}_graph", save=save)
            self.graph_viewer.draw_from_adjMat(adjMat=self.adj_mat.mat_simp, AGNodes=self.AGNodes, level_num=self.level_num, node_dist=2, highlighted_idx_list=level_node_idx_list, name=f"{level}_simp_graph", save=save)
            
            # get the level AGNodes list
            level_AGNodes = [node for node in self.AGNodes if node.level == level]

            # draw the bbox on the ori_img
            self.graph_viewer.draw_multi_nodes_areas_in_img(img=self.ori_img, area_nodes=level_AGNodes, name=f"{level}_area_img", save=save)

        # # @DEBUG
        # # draw the 6 node area
        # self.graph_viewer.draw_single_node_area_in_img(img_=self.ori_img, area_node=self.AGNodes[6], name=f"6_area_img", save=save)
        # self.graph_viewer.draw_single_node_area_in_img(img_=self.ori_img, area_node=self.AGNodes[7], name=f"7_area_img", save=save)

    def fuse_two_area(self, area0, area1):
        """
        """
        u_min0, u_max0, v_min0, v_max0 = area0
        u_min1, u_max1, v_min1, v_max1 = area1
        fused_area = [min(u_min0, u_min1), max(u_max0, u_max1), min(v_min0, v_min1), max(v_max0, v_max1)]
        return fused_area
    
    def append_new_node(self, area_node):
        """
        Args:
            area_node: AGNode
        Returns:
            node_idx: int
        """
        node_idx = len(self.AGNodes)
        self.AGNodes.append(area_node)
        area_node.idx = node_idx
        self.adj_mat.append_node()
        self.update_adjMat_for_a_node(node_idx=node_idx)

        return node_idx

    # graph matching utils =============================================================================================================
    def calc_IoU_efficient(self, node_idx0, node_idx1):
        """ calc IOU by self.overlap_info
        """
        overlap_r_s = self.overlap_info[node_idx0, node_idx1]
        overlap_r_d = self.overlap_info[node_idx1, node_idx0]

        if overlap_r_s == 0 and overlap_r_d == 0:
            return 0
        elif overlap_r_s == 0 or overlap_r_d == 0:
            logger.error(f'ERROR: one of the overlap_r is 0, s: {overlap_r_s}, d: {overlap_r_d}')
            return 0
        else:
            IoU = 1/(1/overlap_r_s + 1/overlap_r_d - 1 + 1e-6)

        return IoU

    def calc_IoU(self, node_idx0, node_idx1):
        """
        """
        if self.adj_mat.mat[node_idx0, node_idx1] == 0 and self.adj_mat.mat[node_idx1, node_idx0] == 0:
            return 0

        if self.IOU_mat is None:
            node_num = self.rt_node_number()
            self.IOU_mat = np.zeros((node_num, node_num)) - 1

        if self.IOU_mat[node_idx0, node_idx1] != -1:
            return self.IOU_mat[node_idx0, node_idx1]

        area0 = self.AGNodes[node_idx0].area
        area1 = self.AGNodes[node_idx1].area
        # calc the intersection area
        u_min = max(area0[0], area1[0])
        u_max = min(area0[1], area1[1])
        v_min = max(area0[2], area1[2])
        v_max = min(area0[3], area1[3])
        intersection_area = max(u_max - u_min, 0) * max(v_max - v_min, 0)
        # calc the union area
        union_area = self.AGNodes[node_idx0].size + self.AGNodes[node_idx1].size - intersection_area
        # calc the IOU
        IOU = intersection_area / union_area

        self.IOU_mat[node_idx0, node_idx1] = IOU
        self.IOU_mat[node_idx1, node_idx0] = IOU

        return IOU
        
    def get_nodes_with_level(self, level):
        """ get node idxs whose level is level
        """
        node_idx_list = [node.idx for node in self.AGNodes if node.level == level]
        return node_idx_list
    
    def get_root_node_idx(self):
        """ return the root node idx
        """
        return self.root_idx

    def get_node_area_img(self, node_idx):
        """ return the area sub-img of a node: a np.ndarray of shape (area_H, area_W, 3)
            Note the area size is spreaded and refined to w_h_ratio
        """
        if node_idx == self.root_idx:
            return self.ori_img
        
        node = self.AGNodes[node_idx]
        [u_min, u_max, v_min, v_max] = node.expand_area 
        node_area = self.ori_img[v_min:v_max, u_min:u_max]
        return node_area

    def get_node_area_expand(self, node_idx):
        """ return [u_min, u_max, v_min, v_max]
        """
        if node_idx == self.root_idx:
            return [0, self.W, 0, self.H]
        else:
            node = self.AGNodes[node_idx]
            return node.expand_area

    def get_node_area(self, node_idx):
        """
        """
        if node_idx == self.root_idx:
            return [0, self.W, 0, self.H]
        else:
            node = self.AGNodes[node_idx]
            return node.area

    def expand_each_node(self, w_h_ratio=1.0, spread_ratio=1.2):
        """ expand each node to a sub-img
        """
        for node in self.AGNodes:

            H, W = self.ori_img.shape[:2]
            u_min, u_max, v_min, v_max = node.area
            # spread the area for specific w_h_ratio
            area_W = u_max - u_min
            area_H = v_max - v_min
            area_center = node.center
            
            if area_W > area_H:
                area_H = area_W / w_h_ratio
            else:
                area_W = area_H * w_h_ratio

            # spread the area
            area_W = area_W * spread_ratio
            area_H = area_H * spread_ratio

            # refine the center, if area is out of the image, move the center
            if area_center[0] - area_W / 2 < 0:
                area_center[0] = area_W / 2
            if area_center[0] + area_W / 2 > W:
                area_center[0] = W - area_W / 2
            if area_center[1] - area_H / 2 < 0:
                area_center[1] = area_H / 2
            if area_center[1] + area_H / 2 > H:
                area_center[1] = H - area_H / 2

            # get the area
            u_min = int(area_center[0] - area_W / 2)
            u_min = max(0, u_min)
            u_max = int(area_center[0] + area_W / 2)
            u_max = min(W, u_max)

            v_min = int(area_center[1] - area_H / 2)
            v_min = max(0, v_min)
            v_max = int(area_center[1] + area_H / 2)
            v_max = min(H, v_max)

            node.expand_area = [u_min, u_max, v_min, v_max]

    def rt_son_node_idxs(self, node_idx, level):
        """ return the idxs of the son nodes of a node
        """
        son_node_idxs = self.adj_mat.get_sons("dense", node_idx)
        son_node_idxs = [idx for idx in son_node_idxs if self.AGNodes[idx].level == level]

        return son_node_idxs

    def rt_next_level_sons(self, node_idx):
        """ return the idxs of the next level son nodes of a node
        """
        son_node_idxs = self.adj_mat.get_sons("dense", node_idx)
        son_node_idxs = [idx for idx in son_node_idxs if self.AGNodes[idx].level == self.AGNodes[node_idx].level + 1]

        return son_node_idxs    

    def rt_all_sons(self, node_idx):
        """ return all the sons of a node
        """
        son_node_idxs = self.adj_mat.get_sons("dense", node_idx)
        return son_node_idxs
  
    def rt_son_node_idxs_with_no_father(self, node_idx):
        """
        """
        son_node_idxs = self.adj_mat.get_sons("dense", node_idx)
        for idx in son_node_idxs:
            father_idxs = self.get_father_node_idxs(idx)
            for father_idx in father_idxs:
                if father_idx in son_node_idxs:
                    son_node_idxs.remove(idx)
                    break

        return son_node_idxs

    def rt_son_node_idxs_mind_size(self, node_idx):
        """ return sons without overlap
        """
        rt_son_list = []
        area_cache = np.zeros((self.H, self.W))
        src_area = self.AGNodes[node_idx].area
        src_area = [int(x) for x in src_area]

        son_node_idxs = self.adj_mat.get_sons("dense", node_idx)
        
        # sort the son nodes by area size from small to large
        son_node_idxs = sorted(son_node_idxs, key=lambda x: self.AGNodes[x].size)

        for son_node_idx in son_node_idxs:
            son_area = self.AGNodes[son_node_idx].area
            son_area = [int(x) for x in son_area]
            new_part_ratio = self.get_new_part_ratio_in_cache(area_cache, son_area, src_area)
            if new_part_ratio > 0.8:
                area_cache[son_area[2]:son_area[3], son_area[0]:son_area[1]] = 1
                rt_son_list.append(son_node_idx)
            else:
                continue
        
        return rt_son_list
    
    def get_inside_area_part(self, candi_area, src_area):
        """ return the common area of two areas
        """
        u_min, u_max, v_min, v_max = candi_area
        src_u_min, src_u_max, src_v_min, src_v_max = src_area

        u_min = max(u_min, src_u_min)
        u_max = min(u_max, src_u_max)
        v_min = max(v_min, src_v_min)
        v_max = min(v_max, src_v_max)

        common_part_size = max(0, u_max - u_min) * max(0, v_max - v_min)

        return [u_min, u_max, v_min, v_max], common_part_size

    def get_new_part_ratio_in_cache(self, area_cache, candi_area, src_area):
        """
        Args:
            area_cache: np.array of shape (H, W), value 1 means the area is occupied
        Return:
            new_part_ratio: float
        """
        new_part_ratio = 0
        common_area, common_part_size = self.get_inside_area_part(candi_area, src_area)
        if common_part_size == 0:
            return new_part_ratio

        u_min, u_max, v_min, v_max = common_area
        # count the non-zero value of common_area in area_cache
        common_area_cache = area_cache[v_min:v_max, u_min:u_max]
        common_area_cache = common_area_cache.flatten()
        # count the non-zero value number
        common_area_cache = [1 if x > 0 else 0 for x in common_area_cache]
        non_zero_value_num = sum(common_area_cache)
        # count the ratio
        new_part_ratio = 1 - non_zero_value_num / common_part_size

        return new_part_ratio

    def rt_parents(self, node_idx, mode="dense"):
        """
        """
        if mode == "dense":
            father_node_idxs = self.adj_mat.get_fathers(self.adj_mat.mat, node_idx)
        elif mode == "simp":
            father_node_idxs = self.adj_mat.get_fathers(self.adj_mat.mat_simp, node_idx)

        return father_node_idxs

    def rt_neighbours(self, node_idx, mode="dense"):
        """
        """
        if mode == "dense":
            neighbour_node_idxs = self.adj_mat.get_neighbours(self.adj_mat.mat, node_idx)
        elif mode == "simp":
            neighbour_node_idxs = self.adj_mat.get_neighbours(self.adj_mat.mat_simp, node_idx)

        return neighbour_node_idxs

    def get_node_area_offset(self, node_idx):
        """
        """
        node = self.AGNodes[node_idx]
        u_min, u_max, v_min, v_max = node.area
        offset = [u_min, v_min]
        return offset
    
    def update_node_by_fuse_area(self, node_idx, fuse_area):
        """
        Args:
            node_idx: int
            fuse_area: [u_min, u_max, v_min, v_max]
        Returns:
            None
        """
        node = self.AGNodes[node_idx]
        fused_area = self.fuse_two_area(node.area, fuse_area)
        # update the node
        node.area = fused_area
        node.size = (fused_area[1] - fused_area[0]) * (fused_area[3] - fused_area[2])
        node.center = [(fused_area[0] + fused_area[1]) / 2, (fused_area[2] + fused_area[3]) / 2]
        node.W = fused_area[1] - fused_area[0]
        node.H = fused_area[3] - fused_area[2]

        node.asign_level(self.size_level_thd_list)

        # update adjmat
        self.update_adjMat_for_a_node(node_idx=node_idx)