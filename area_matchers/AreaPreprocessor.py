
import sys
sys.path.append("..")

import os.path as osp
import os
import numpy as np 
from loguru import logger
import cv2
import copy

from .AGUtils import KMCluster, AGViewer, MaskViewer
from .AGConfig import preprocess_configs
from utils.transformer import SEEM2SAM

""" after the area preprocess, we get the area_info_list"""
class AreaPreprocesser(object):
    """
    Preprocess the areas for a single SAM result
    Funcs:
        0. check seg source: SAM or Mask
        if SAM:
            1. load SAM results *.np
                - get 'segmentation', 'bbox'
                - transform to our format
                    area_info = {
                        "area_bbox": [u_min, u_max, v_min, v_max],
                        "area_center": [u_center, v_center],
                        "area_size": area_size,
                        "mask": None,
                        }
                - calc area_size, area_center, segmentation->mask
        if Mask:
            1. load Mask results *.png
                - get all segs based on labels in the mask
                - transform to our format

        2. refine bbox
            split multi-connected components
        3. filter abnormal areas
            - too small areas -> fusion with nearest or drop
            - too slim areas -> filter <W >= 0.9*H || H >= 0.9*W>
            # - too large areas (mostly background) -> filter <W >= 0.9*W_ori || H >= 0.9*H_ori>
            - repeated areas -> extreme close center & similar size
        4. achieve 
    Flow:
        self.load() -> self.refine_bbox() -> self.filter_abnormal_areas() -> self.refined_areas
    Returns:
        filtered areas info : list[area_info, ...]
    """
    default_configs = preprocess_configs

    def __init__(self, configs) -> None:
        """
        Args:
            self.min_area_size
            self.W
            self.H
            self.WH_ratio_thd
            # self.max_size_ratio_thd
            self.tiny_area_size: get area size < tiny_area_size, drop
            self.save_path
            self.MaskViewer

        """
        configs = {**self.default_configs, **configs}

        self.min_area_size = configs["min_area_size"]
        self.W = configs["W"]
        self.H = configs["H"]
        self.max_wh_ratio= configs["max_wh_ratio"] # slim thd
        # self.max_size_ratio_thd = configs["max_size_ratio_thd"]
        self.tiny_area_size = configs["tiny_area_size"]
        self.save_path = configs["save_path"]
        self.MaskViewer = MaskViewer(self.save_path)
        self.topK = configs["topK"]
        self.min_dist_thd = configs["min_dist_thd"]
        self.seg_source = configs["seg_source"]

        self.refined_areas = None

    def load(self, res_path="", res=None):
        """
        """
        if self.seg_source == "SAM":
            self._load_sam(sam_results_path=res_path, sam_res=res)
        elif self.seg_source == "Sem":
            self.transform = SEEM2SAM()
            self._load_sem(mask_results_path=res_path, mask_res=res)

    def _load_sem(self, mask_results_path="", mask_res=None):
        """ load segmentations from semantic masks
        """
        sam_res = []
        if mask_results_path != "":
            mask_res =  cv2.imread(mask_results_path, cv2.IMREAD_UNCHANGED)
        
        # resize the mask to W*H
        mask_res = cv2.resize(mask_res, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        sam_res = self.transform.trans_png2npy(mask_res, save=False)

        self.sam_results = self._load_sam(sam_res=sam_res)

        return self.sam_results

    def _load_sam(self, sam_results_path="", sam_res=None):
        """ load & transform
        Args:
            sam_results_path
        Returns:
            self.sam_results: list of sam results
        """
        self.sam_results = []    
        if sam_results_path != "":
            # logger.info(f"Loading sam results from {sam_results_path}...")
            sam_res = np.load(sam_results_path, allow_pickle=True)
        elif sam_res is None:
            logger.error("No sam results path or sam results!")
            return

        for sam_res_i in sam_res:
            area_info = {}
            # print(sam_res_i["bbox"])
            # logger.info(f"before loading bbox is {sam_res_i['bbox']}")
            area_info["area_bbox"] = self._bbox2area(sam_res_i["bbox"])
            # logger.info(f"after loading bbox is {area_info['area_bbox']}")
            area_info["area_center"] = [int((sam_res_i["bbox"][0] + sam_res_i["bbox"][2]/2)), int((sam_res_i["bbox"][1] + sam_res_i["bbox"][3]/2))]
            area_info["area_size"] = sam_res_i["bbox"][2]*sam_res_i["bbox"][3]
            area_info["mask"] = sam_res_i["segmentation"]
            self.sam_results.append(area_info)

        logger.info(f"Loaded sam results with size: {len(self.sam_results)}")
        if False:
            self.MaskViewer.draw_multi_masks_in_one(self.sam_results, self.W, self.H, "after_load")
            self.MaskViewer.draw_multi_masks_in_one(sam_res, self.W, self.H, "input", key="segmentation")

        return self.sam_results

    def _bbox2area(self, bbox):
        """
        Args:
            bbox: [uvwh]
        Returns:
            area: [umin,umax,vmin,vmax]
        """
        area = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]

        return area
    
    def refine_bbox(self, draw_flag=False):
        """
        Args:
            sam_results
        Returns:
            sam_results
        """
        refined_areas = []
        if draw_flag:
            self.MaskViewer.draw_multi_masks_in_one(self.sam_results, self.W, self.H, "before_refine_bbox")

        # logger.info(f"Refining bbox with size: {len(self.sam_results)}...")
        for i, area_info in enumerate(self.sam_results):
            refined_areas += self._split_multi_connected_components(area_info, i)
        # logger.info(f"Refined bbox with size: {len(refined_areas)}")

        self.refined_areas = refined_areas

        # # filter repeat areas
        # self.refined_areas = self.filter_repeat_areas(self.refined_areas, draw_flag=draw_flag)

        if draw_flag:
            self.MaskViewer.draw_multi_masks_in_one(self.refined_areas, self.W, self.H, "after_refine_bbox")

        return self.refined_areas

    def filter_repeat_areas(self, area_list, dist_thd=50, draw_flag=False):
        """ filter areas with similar bbox located point
        """
        area_list_filtered = copy.deepcopy(area_list)
        pop_list = []
        # logger.info(f"Filtering repeat areas with size: {len(area_list_filtered)}...")
        for i, area_info in enumerate(area_list):
            area_bbox = area_info["area_bbox"]
            pop_flag = False
            for j, area_info_j in enumerate(area_list):
                if i == j:
                    continue
                area_bbox_j = area_info_j["area_bbox"]
                # calc the mean distance between the points of bbox
                dist_ul = np.sqrt(np.sum(np.square(np.array([area_bbox[0], area_bbox[2]]) - np.array([area_bbox_j[0], area_bbox_j[2]]))))
                dist_rd = np.sqrt(np.sum(np.square(np.array([area_bbox[1], area_bbox[3]]) - np.array([area_bbox_j[1], area_bbox_j[3]]))))
                dist_mean = (dist_ul + dist_rd) / 2
                if dist_mean < dist_thd:
                    if area_info["area_size"] < area_info_j["area_size"]:
                        pop_flag = True
                    elif area_info["area_size"] == area_info_j["area_size"]:
                        if i > j:
                            pop_flag = True
            if pop_flag:
                pop_list.append(i)
        
        for i in pop_list[::-1]:
            area_list_filtered.pop(i)
        
        # logger.info(f"Filtered repeat areas with size: {len(area_list_filtered)}")

        return area_list_filtered

    def _split_multi_connected_components(self, area_info, idx, draw_flag=False):
        """ if no multiple areas, return [area_info]
            else return [area_info_1, area_info_2, ...]
        Args:
            area_info
        Returns:
            list of area_info
        """
        mask = area_info["mask"].astype(np.uint8)
        # calc the number of connected components in mask
        # donnot count the 0 background
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if draw_flag:
            self.MaskViewer.draw_single_mask(mask, area_info['area_bbox'], f"before_refine_{idx}_num_cc_{num_labels}")

        # logger.info(f"For area with center {area_info['area_center']}, there are {num_labels} connected components.")
        if num_labels == 2:
            area_info_refined = {}
            area_info_refined["area_bbox"] = [stats[1][0], stats[1][0]+stats[1][2], stats[1][1], stats[1][1]+stats[1][3]]
            area_info_refined["area_center"] = [int((stats[1][0] + stats[1][0]+stats[1][2])/2), int((stats[1][1] + stats[1][1]+stats[1][3])/2)]
            area_info_refined["area_size"] = stats[1][2] * stats[1][3]
            if area_info_refined["area_size"] < self.tiny_area_size:
                return []
            area_info_refined["mask"] = (labels == 1).astype(np.uint8)
            if draw_flag:
                self.MaskViewer.draw_single_mask(area_info_refined["mask"], area_info_refined["area_bbox"], f"after_refine_{idx}")

            return [area_info_refined]
        elif num_labels > 2:
            areas_info = []
            for i in range(1, num_labels):
                area_info_i = {}
                area_info_i["area_bbox"] = [stats[i][0], stats[i][0]+stats[i][2], stats[i][1], stats[i][1]+stats[i][3]]
                area_info_i["area_center"] = [int((stats[i][0] + stats[i][0]+stats[i][2])/2), int((stats[i][1] + stats[i][1]+stats[i][3])/2)]
                area_info_i["area_size"] = stats[i][2] * stats[i][3]
                if area_info_i["area_size"] < self.tiny_area_size:
                    continue
                area_info_i["mask"] = (labels == i).astype(np.uint8)
                areas_info.append(area_info_i)
                if draw_flag:
                    self.MaskViewer.draw_single_mask(area_info_i["mask"], area_info_i["area_bbox"], f"after_refine_{idx}_idx_cc_{i}")
            return areas_info

    def filter_abnormal_areas(self, draw_flag=False):
        """ filter the abnormal areas
        Args:
            self.refined_areas
        Returns:
            self.filtered_areas
        """
        topK = self.topK
        min_dist_thd = self.min_dist_thd
        # logger.info(f"Filtering abnormal areas with size: {len(self.refined_areas)}...")
        if draw_flag:
            self.MaskViewer.draw_multi_masks_in_one(self.refined_areas, self.W, self.H, "before_filter_abnormal_areas")
        small_flag = True
        slim_flag = True
        iter_num = 0
        while small_flag or slim_flag:
            iter_num += 1
            if small_flag:
                
                small_idx_list = self._check_small_areas(self.refined_areas)
                # logger.info(f"Filtering {len(small_idx_list)} small areas ...")

                # draw the small areas
                # if draw_flag:
                #     self.MaskViewer.draw_multi_masks_in_one([self.refined_areas[idx] for idx in small_idx_list], self.W, self.H, f"small_areas_{iter_num}")

                self.refined_areas = self.fusion_areas(self.refined_areas, small_idx_list, topK, min_dist_thd)

                # draw the filtered areas
                # if draw_flag:
                #     self.MaskViewer.draw_multi_masks_in_one(self.refined_areas, self.W, self.H, f"after_filter_small_areas_{iter_num}")
            
            if slim_flag:
                slim_idx_list = self._check_slim_areas(self.refined_areas)
                # logger.info(f"Filtering {len(slim_idx_list)} slim areas ...")

                # draw the slim areas
                # if draw_flag:
                #     self.MaskViewer.draw_multi_masks_in_one([self.refined_areas[idx] for idx in slim_idx_list], self.W, self.H, f"slim_areas_{iter_num}")

                self.refined_areas = self.fusion_areas(self.refined_areas, slim_idx_list, topK, min_dist_thd)

                # # draw the filtered areas
                # if draw_flag:
                #     self.MaskViewer.draw_multi_masks_in_one(self.refined_areas, self.W, self.H, f"after_filter_slim_areas_{iter_num}")

            if len(self._check_slim_areas(self.refined_areas)) == 0:
                slim_flag = False
            if len(self._check_small_areas(self.refined_areas)) == 0:
                small_flag = False

            # break
        
        # filter repeat areas
        self.refined_areas = self.filter_repeat_areas(self.refined_areas, draw_flag=draw_flag)

        # logger.info(f"Filtered abnormal areas with size: {len(self.refined_areas)}")
        if draw_flag:
            self.MaskViewer.draw_multi_masks_in_one(self.refined_areas, self.W, self.H, "after_filter_abnormal_areas")
            # draw every mask in the filtered areas
            # for i, area_info in enumerate(self.refined_areas):
            #     self.MaskViewer.draw_single_mask(area_info["mask"], area_info["area_bbox"], f"after_filter_abnormal_areas_{i}")

        self.filtered_areas = self.refined_areas

        return self.refined_areas
    
    def _check_slim_areas(self, area_infos):
        """
        Args:
            area_info
        Returns:
            idx_list    
        """
        idx_list = []
        for i, area_info in enumerate(area_infos):
            area_w = area_info["area_bbox"][1] - area_info["area_bbox"][0]
            area_h = area_info["area_bbox"][3] - area_info["area_bbox"][2]
            wh_ratio = max(area_w, area_h) / (min(area_w, area_h)+1e-6)
            if wh_ratio > self.max_wh_ratio:
                idx_list.append(i)
                break
        
        return idx_list

    def _check_small_areas(self, area_infos):
        """ check the small areas, return small area idx list
        Args:
            area_infos
        Returns:
            idx_list
        """
        idx_list = []
        for i, area_info in enumerate(area_infos):
            if area_info["area_size"] < self.min_area_size:
                idx_list.append(i)
                break

        return idx_list

    def fusion_areas(self, src_area_info_list, await_area_idx_list, topK=3, min_dist_thd=100):
        """ fusion the areas, take the areas from list accroding to idx list
            fuse to area in left areas 
        """
        for idx in await_area_idx_list:
            await_fuse_area = copy.deepcopy(src_area_info_list[idx])
            # remove the await_fuse_area from src_area_info_list
            src_area_info_list.pop(idx)
            src_area_info_list = self._fuse_single_area_to_list(await_fuse_area, src_area_info_list)

        return src_area_info_list

    def _fuse_single_area_to_list(self, await_fuse_area, area_info_list, topK=3, min_dist_thd=100):
        """ find the target and fuse
        Args:

        Returns:
            modified area_info_list
        """
        fused_area_info_list = copy.deepcopy(area_info_list)
        await_fuse_center = await_fuse_area["area_center"]
        # calc the distance matrix between await_fuse_area and area_info_list
        dist_matrix = []
        for i, area_info in enumerate(area_info_list):
            dist = np.sqrt((await_fuse_center[0] - area_info["area_center"][0])**2 + (await_fuse_center[1] - area_info["area_center"][1])**2)
            dist_matrix.append({i: dist})

        if len(dist_matrix) == 0:
            logger.info("No fuse target")
            return fused_area_info_list

        max_sim_idx = -1
        max_sim = 0
        
        # sort the dist_matrix by dist
        dist_matrix.sort(key=lambda x: list(x.values())[0])
        if list(dist_matrix[0].values())[0] >= min_dist_thd: 
            logger.info("No fuse target")
            return fused_area_info_list

        # get the topK closest area size similarity
        topK = min(topK, len(dist_matrix))
        for i in range(topK):
            try:
                area_sim = (self._calc_area_similarity(await_fuse_area, area_info_list[list(dist_matrix[i].keys())[0]]))
            except Exception as e:
                logger.exception(e)

            if area_sim > max_sim and list(dist_matrix[i].values())[0] < 0.7*min_dist_thd:
                max_sim_idx = list(dist_matrix[i].keys())[0]
                max_sim = area_sim

        # fuse area, including bbox, size, center, mask
        if max_sim_idx == -1:
            logger.info("No fuse target")
            return fused_area_info_list
        fused_area_info = {}
        
        # fuse mask
        fused_mask = await_fuse_area["mask"] + area_info_list[max_sim_idx]["mask"]
        fused_mask[fused_mask > 1] = 1
        fused_area_info["mask"] = fused_mask

        # fuse bbox
        fused_bbox = [min(await_fuse_area["area_bbox"][0], area_info_list[max_sim_idx]["area_bbox"][0]), max(await_fuse_area["area_bbox"][1], area_info_list[max_sim_idx]["area_bbox"][1]), min(await_fuse_area["area_bbox"][2], area_info_list[max_sim_idx]["area_bbox"][2]), max(await_fuse_area["area_bbox"][3], area_info_list[max_sim_idx]["area_bbox"][3])]
        fused_area_info["area_bbox"] = fused_bbox

        # fuse center
        fused_center = [(fused_bbox[0] + fused_bbox[1])//2, (fused_bbox[2] + fused_bbox[3])//2]
        fused_area_info["area_center"] = fused_center

        # fuse size
        fused_size = (fused_bbox[1] - fused_bbox[0]) * (fused_bbox[3] - fused_bbox[2])
        fused_area_info["area_size"] = fused_size
        
        fused_area_info_list[max_sim_idx] = fused_area_info

        return fused_area_info_list

    def _calc_area_similarity(self, area_info_1, area_info_2):
        """ calc the similarity of area size between two areas
        Returns:
            area_size_similarity: bigger is better
        """
        area_size_1 = area_info_1["area_size"]
        area_size_2 = area_info_2["area_size"]
        area_size_similarity = min(area_size_1, area_size_2) / (max(area_size_1, area_size_2) + 1e-6)
        return area_size_similarity