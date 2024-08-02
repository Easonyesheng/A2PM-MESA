'''
Author: EasonZhang
Date: 2023-06-29 10:32:53
LastEditors: EasonZhang
LastEditTime: 2023-12-27 11:28:36
FilePath: /A2PM/configs/AGConfig.py
Description: config for area graph

Copyright (c) 2023 by EasonZhang, All Rights Reserved. 
'''



preprocess_configs = {
    "min_area_size": 6000,
    "W": 640,
    "H": 480,
    "save_path": "/data0/zys/A2PM/testAG/graphResComplete",
    "max_wh_ratio": 4.0,
    "max_size_ratio_thd": 0.5,
    "tiny_area_size": 900,
    'topK': 3,
    'min_dist_thd': 100,
    'seg_source': "SAM", # "SAM" or "Sem"
}

areagraph_configs = {
    'preprocesser_config': preprocess_configs,
    'sam_res_path': "/data0/zys/A2PM/testAG/res/SAMRes.npy",
    'sem_res_path': "/data0/zys/A2PM/testAG/res/semRes.png",
    'W': 640,
    'H': 480,
    'save_path': "/data0/zys/A2PM/testAG/graphResComplete",
    'ori_img_path': "/data0/zys/A2PM/data/ScanData/scene0000_00/color/12.jpg",
    'fs_overlap_thd': 0.8,
    'level_num': 4,
    'level_step': [560, 480, 256, 130, 0],
    'show_flag': 0,
}
