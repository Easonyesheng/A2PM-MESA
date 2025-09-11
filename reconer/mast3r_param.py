'''
Author: EasonZhang
Date: 2024-09-03 15:10:09
LastEditors: EasonZhang
LastEditTime: 2024-09-03 15:10:34
FilePath: /mast3r-main/3d_reconers/mast3r_param.py
Description: params for mast3r

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

class SceneParams:
    """data class for get_reconstructed_scene"""

    def __init__(self,
        outdir,
        filelist, # image list
        model,
        divice = 'cuda',
        gradio_delete_cache = 0,
        silent = False,
        image_size = 512,
        current_scene_state = None, # for two views
        optim_level = 'refine', # 'refine' or 'coarse' or 'refine+depth'(refine depth)
        lr1 = 0.07, # coarse refine learning rate
        niter1 = 500, # coarse refine iterations
        lr2 = 0.014, # fine refine learning rate
        niter2 = 200, # fine refine iterations
        min_conf_thr = 1.5, # minimum confidence threshold
        matching_conf_thr = 5,
        as_pointcloud = True,
        mask_sky = False,
        clean_depth = True,
        transparent_cams = False,
        cam_size = 0.2,
        scenegraph_type = 'complete', # multi-view need others
        winsize = 1,
        win_cyclic = False,
        refid = 0,
        TSDF_thresh = 0.0,
        shared_intrinsics = False,
    ):
        self.params = {
            'outdir': outdir,
            'filelist': filelist,
            'model': model,
            'device': divice,
            'gradio_delete_cache': gradio_delete_cache,
            'silent': silent,
            'image_size': image_size,
            'current_scene_state': current_scene_state,
            'optim_level': optim_level,
            'lr1': lr1,
            'niter1': niter1,
            'lr2': lr2,
            'niter2': niter2,
            'min_conf_thr': min_conf_thr,
            'matching_conf_thr': matching_conf_thr,
            'as_pointcloud': as_pointcloud,
            'mask_sky': mask_sky,
            'clean_depth': clean_depth,
            'transparent_cams': transparent_cams,
            'cam_size': cam_size,
            'scenegraph_type': scenegraph_type,
            'winsize': winsize,
            'win_cyclic': win_cyclic,
            'refid': refid,
            'TSDF_thresh': TSDF_thresh,
            'shared_intrinsics': shared_intrinsics,
            'rt_scene': True,
            'rt_matches': True,
        }

