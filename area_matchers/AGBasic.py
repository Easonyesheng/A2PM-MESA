

import os.path as osp
import os
import numpy as np 
from loguru import logger
import cv2
import copy

from .AGUtils import KMCluster, AGViewer, MaskViewer


area_info_temp = {
    "area_bbox": [0, 0, 0, 0],
    "area_center": [0, 0],
    "area_size": 0,
    "mask": None,
}

class AGNode(object):
    """ the area graph node class
    Args:
        area_info: dict, area information
        ----> split into 4 parts:
                self.area: [u_min, u_max, v_min, v_max]
                self.center: [u_center, v_center]
                self.size: int, the area size
                self.mask: np.array, the mask of the area

        idx: int, the index of the node
        level: int, the level of the node, the smaller the larger
    """
    def __init__(self, area_info, idx=None) -> None:
        """
        """
        self.area = area_info['area_bbox']
        self.bbox = self.area # area is ambiguous, bbox is more clear
        self.center = area_info['area_center']
        self.size = area_info['area_size']
        self.mask = area_info['mask']
        self.W = self.area[1] - self.area[0]
        self.H = self.area[3] - self.area[2]
        if idx is not None:
            self.idx = idx
        self.level = 0
        self.expand_area = None # the area after expand

    def copy_to(self, idx):
        """ copy the node
        """
        new_node = AGNode({
            "area_bbox": self.area,
            "area_center": self.center,
            "area_size": self.size,
            "mask": self.mask,
        })
        new_node.idx = idx
        new_node.level = self.level
        new_node.W = self.W
        new_node.H = self.H
        new_node.expand_area = self.expand_area

        return new_node

    def __expand_area(self, bbox, nergibor_thd):
        # expand the area by nergibor_thd
        bbox[0] = bbox[0] - nergibor_thd
        bbox[0] = max(bbox[0], 0)
        bbox[1] = bbox[1] + nergibor_thd
        bbox[2] = bbox[2] - nergibor_thd
        bbox[2] = max(bbox[2], 0)
        bbox[3] = bbox[3] + nergibor_thd
        return bbox

    def overlap_check_complete(self, dst_node, fs_overlap_thd, nergibor_thd = 10):
        """ check the overlap between self and dst_node
        Args:
            fs_overlpa_thd: float, the overlap threshold for father and son
        Returns:
            overlap_flag:
                - -2: dst and self is repeated area
                - -1: dst is the father of self
                - 0: no overlap
                - 1: self is the father of dst
                - 2: dst and self is neighbour
            overlap_ratio_s: float, {overlap} / {size of self}
            overlap_ratio_d_s: float, {overlap} / {size of dst}
        """
        bbox_src = copy.copy(self.area)
        size_src = self.size
        bbox_dst = copy.copy(dst_node.area)
        size_dst = dst_node.size

        bbox_src = self.__expand_area(bbox_src, nergibor_thd)
        bbox_dst = self.__expand_area(bbox_dst, nergibor_thd)

        # check the overlap and the overlap percentage
        overlap_flag = 0
        overlap_ratio_s = 0.0
        overlap_ratio_d = 0.0
        
        # calc the overlap part size
        overlap_part = [max(bbox_src[0], bbox_dst[0]), min(bbox_src[1], bbox_dst[1]), max(bbox_src[2], bbox_dst[2]), min(bbox_src[3], bbox_dst[3])]
        overlap_part_size = (overlap_part[1] - overlap_part[0]) * (overlap_part[3] - overlap_part[2])

        overlap_ratio_s = overlap_part_size / size_src
        overlap_ratio_d = overlap_part_size / size_dst

        # check the overlap
        if overlap_ratio_s > fs_overlap_thd and overlap_ratio_d > fs_overlap_thd:
            overlap_flag = -2
        elif overlap_ratio_s > fs_overlap_thd:
            overlap_flag = -1
        elif overlap_ratio_d > fs_overlap_thd:
            overlap_flag = 1
        else:
            overlap_flag = 0

        return overlap_flag, overlap_ratio_s, overlap_ratio_d

    def overlap_check(self, dst_node, nergibor_thd = 10):
        """ check the overlap between self and dst_node
        Args:
            bbox = [u_min, u_max, v_min, v_max]
        """
        bbox_src = self.area
        bbox_dst = dst_node.area
        # check the overlap and the overlap percentage
        overlap_flag = False
        overlap_percentage = 0.0
        if bbox_src[0] < bbox_dst[1] + nergibor_thd and bbox_src[1] > bbox_dst[0] - nergibor_thd and \
            bbox_src[2] < bbox_dst[3] + nergibor_thd and bbox_src[3] > bbox_dst[2] - nergibor_thd:
            overlap_flag = True
            overlap_percentage = max(min(bbox_src[1], bbox_dst[1]) - max(bbox_src[0], bbox_dst[0]), 0) * \
                max(min(bbox_src[3], bbox_dst[3]) - max(bbox_src[2], bbox_dst[2]), 0) / \
                    (self.size)

        return overlap_flag, overlap_percentage

    def asign_level(self, size_level_thd_list):
        """ asign the level of the node by area size
        Args:
            size_level_thd_list: list, the area size level threshold
                [640*480, 480*480, 256*256, 100*100, 0]
                - level i: size_level_thd_list[i] <= area_size < size_level_thd_list[i-1]
                - the smaller the larger
        """
        for idx, thd in enumerate(size_level_thd_list):
            if self.size >= thd:
                self.level = idx
                break
        
        return self.level

    def level_up(self):
        """
        """
        if self.level > 1:
            self.level -= 1
            return True
        else:
            # logger.warning("The node level is already the largest, cannot level up!")
            return False

    def tune_center_and_more(self, W, H):
        """ tune the center of the node to ensure the center is in the image
        Args:
            W: int, the width of the img
            H: int, the height of the img
            self.center: [u_center, v_center]
        Returns:
            self.center: [u_center, v_center]
            self.area: [u_min, u_max, v_min, v_max]
            self.size: int, the area size
        """
        u_center, v_center = self.center

        if u_center - self.W / 2 < 0:
            u_center = self.W / 2
        elif u_center + self.W / 2 > W:
            u_center = W - self.W / 2

        if v_center - self.H / 2 < 0:
            v_center = self.H / 2
        elif v_center + self.H / 2 > H:
            v_center = H - self.H / 2
        
        self.center = [u_center, v_center]
        self.area = [int(u_center - self.W / 2), int(u_center + self.W / 2), int(v_center - self.H / 2), int(v_center + self.H / 2)]
        self.size = self.W * self.H

    def get_dist(self, dst_node):
        """ get the distance between self and dst_node
        Returns:
            dist: float, the distance between self and dst_node
            outside_dist: float, the distance between the outside boundary of self and dst_node
        """
        dst_center = dst_node.center
        dst_W, dst_H = dst_node.W, dst_node.H
        dist = ((self.center[0] - dst_center[0]) ** 2 + (self.center[1] - dst_center[1]) ** 2) ** 0.5
        outside_dist = max(max(abs(self.center[0] - dst_center[0]) - (self.W + dst_W) / 2, 0), max(abs(self.center[1] - dst_center[1]) - (self.H + dst_H) / 2, 0))

        return dist, outside_dist

    def area_expanding(self, img_W, img_H, w_h_ratio=1.0, spread_ratio=1.2):
        """
        """
        u_min, u_max, v_min, v_max = self.area
        # spread the area for specific w_h_ratio
        area_W = u_max - u_min
        area_H = v_max - v_min
        area_center = self.center
        
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
        if area_center[0] + area_W / 2 > img_W:
            area_center[0] = img_W - area_W / 2
        if area_center[1] - area_H / 2 < 0:
            area_center[1] = area_H / 2
        if area_center[1] + area_H / 2 > img_H:
            area_center[1] = img_H - area_H / 2

        # get the area
        u_min = int(area_center[0] - area_W / 2)
        u_min = max(0, u_min)
        u_max = int(area_center[0] + area_W / 2)
        u_max = min(img_W, u_max)

        v_min = int(area_center[1] - area_H / 2)
        v_min = max(0, v_min)
        v_max = int(area_center[1] + area_H / 2)
        v_max = min(img_H, v_max)

        self.expand_area = [u_min, u_max, v_min, v_max]

class AdjMat(object):
    """ the area graph adjacent matrix class
            1 - neighbor; 
            Mat[src, dst] = 2 -> src is the father of dst; 
            0 - no connection
    Rules:
        1. only father node direct to son with weight 2
        2. neibour node direct to each other with weight 1, symmetric
    Funcs:
        0. Update
        1. Insert Graph Node
        2. Search
            Son
            Father
            Brother
    Args:
        init_N: int, the initial number of nodes
        self.mat: np.array, the dense adjacent matrix: father's father is also father
        self.mat_simp: np.array, the simple adjacent matrix: father's father is not father
            - father only connect to next level son, or only son
            - same level only neighbor, never father
    """
    def __init__(self, init_N) -> None:
        """ adjacent matrix initialization
        """
        if init_N < 1:
            raise ValueError("init_N must be larger than 1")
        self.mat = np.zeros((init_N, init_N), dtype=np.int32) # dense father-son adjacent matrix
        self.mat_simp = np.zeros((init_N, init_N), dtype=np.int32) # simple father-son adjacent matrix
        self.N = init_N
    
    def clean_mat_value(self, AGNodes, simp_gen=True):
        """ clean the mat
            - set the neibour value to 0, if the neighbor is son
            - if both edge are 2, set the area with larger area as father and the other as son
            - generate the simple adjacent matrix - through node level
        """
        if len(AGNodes) == 0: # no nodes
            self.mat = None
            self.mat_simp = None
            self.N = 0
            return

        for src_idx in range(self.N):
            for dst_idx in range(self.N):
                if self.mat[src_idx, dst_idx] == 1:
                    # check if the neighbor is father
                    if self.mat[dst_idx, src_idx] == 2:
                        self.mat[src_idx, dst_idx] = 0
                # check if both edge are 2
                elif self.mat[src_idx, dst_idx] == 2:
                    if self.mat[dst_idx, src_idx] == 2:
                        if AGNodes[src_idx].size > AGNodes[dst_idx].size:
                            self.mat[dst_idx, src_idx] = 0
                        else:
                            self.mat[src_idx, dst_idx] = 0

        if simp_gen:
            # generate the simple adjacent matrix
            self.mat_simp = copy.deepcopy(self.mat)
            for father_idx in range(self.N):
                for son_idx in range(self.N):
                    if self.mat[father_idx, son_idx] == 2:
                        son_fathers_list = self.get_fathers(self.mat, son_idx)
                        if len(son_fathers_list) == 1:
                            continue
                        if AGNodes[father_idx].level == AGNodes[son_idx].level:
                            self.mat_simp[father_idx, son_idx] = 1
                        elif AGNodes[father_idx].level == AGNodes[son_idx].level - 1:
                            self.mat_simp[father_idx, son_idx] = 2
                        else:
                            self.mat_simp[father_idx, son_idx] = 0
                    # only save the same level neighbor or only neighbor
                    if self.mat[father_idx, son_idx] == 1:
                        temp_neighbor_list = self.get_neighbors(self.mat, son_idx)
                        if len(temp_neighbor_list) == 1:
                            continue
                        if AGNodes[father_idx].level != AGNodes[son_idx].level:
                            self.mat_simp[father_idx, son_idx] = 0

    def append_node(self):
        """
        """
        self.N += 1
        self.mat = np.pad(self.mat, ((0, 1), (0, 1)), 'constant', constant_values=0)
        self.mat_simp = np.pad(self.mat_simp, ((0, 1), (0, 1)), 'constant', constant_values=0)
        return self.N

    def pop_node(self):
        """
        """
        self.N -= 1
        if self.N < 1:
            self.mat = None
            self.mat_simp = None
            return self.N
        
        self.mat = np.delete(self.mat, self.N, axis=0)
        self.mat = np.delete(self.mat, self.N, axis=1)
        self.mat_simp = np.delete(self.mat_simp, self.N, axis=0)
        self.mat_simp = np.delete(self.mat_simp, self.N, axis=1)
        return self.N

    def add_root_node(self, AGNodes):
        """ root node for show
        Args:
            AGNodes: list, the area graph nodes
        """
        if self.N == 0:
            self.N += 1
            self.mat = np.zeros((1, 1), dtype=np.int32)
            self.mat_simp = np.zeros((1, 1), dtype=np.int32)
            return self.N

        self.N += 1
        self.mat = np.pad(self.mat, ((0, 1), (0, 1)), 'constant', constant_values=0)
        self.mat_simp = np.pad(self.mat_simp, ((0, 1), (0, 1)), 'constant', constant_values=0)
        
        # get the level 1 nodes idx
        level_1_nodes_idx = []
        for idx in range(len(AGNodes)):
            if AGNodes[idx].level == 1:
                level_1_nodes_idx.append(idx)
        
        # connect the root node to level 1 nodes as father
        for idx in level_1_nodes_idx:
            self.mat[self.N - 1, idx] = 2
            self.mat_simp[self.N - 1, idx] = 2

        return self.N

    def get_neighbours(self, mat, Nidx):
        """ get the neighbor list for Nidx
        """
        neighbor_list = []
        for idx in range(self.N):
            if mat[Nidx, idx] == 1 or mat[idx, Nidx] == 1:
                neighbor_list.append(idx)
        return neighbor_list

    def get_neighbors(self, mat, Nidx):
        """ get the neighbor list for Nidx
        """
        neighbor_list = []
        for idx in range(self.N):
            if mat[Nidx, idx] == 1 or mat[idx, Nidx] == 1:
                neighbor_list.append(idx)
        return neighbor_list

    def get_fathers(self, mat, Nidx):
        """ get the fathers list for Nidx
        """
        fathers_list = []
        for idx in range(self.N):
            if mat[idx, Nidx] == 2:
                fathers_list.append(idx)
        return fathers_list

    def get_sons(self, mat_mode, father_idx):
        """ get the sons idxs list for father_idx
        """

        if mat_mode == "simp":
            mat = self.mat_simp
        elif mat_mode == "dense":
            mat = self.mat
        else:
            raise ValueError("mat_mode must be simp or dense")

        sons_list = []
        for idx in range(self.N):
            if mat[father_idx, idx] == 2:
                sons_list.append(idx)
        return sons_list

class OverlapInfoMat(object):
    """ TODO:
    """

