'''
Author: EasonZhang
Date: 2024-06-28 19:30:20
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-11 15:18:42
FilePath: /SA2M/hydra-mesa/metric/eval_ratios.py
Description: scripts for evaluation of ratios

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import sys
sys.path.append('..')

import os
import numpy as np
from loguru import logger

from metric.Evaluation import AMEval, PoseAUCEval, TimeEval, MMAEval

root_path = '/opt/data/private/A2PM-git/A2PM-MESA/res'

# specific
root_folder = 'dmesa-dkm-md-eval-res'
root_folder = 'dmesa-dkm-scannet-res'
root_folder = 'mesa-f-dkm-md-eval-res'
# root_folder = 'mesa-f-dkm-sn-eval-res'
root_folder = "mesa-f-mast3r-md-eval-res"
root_folder = "mesa-f-mast3r-sn-eval-res"




baseline_name = 'pm'
challenger_name = 'a2pm'
folder_name = 'ratios'
phis = ['0.5', '2.0', '3.0', '3.5', '5.0']



log_folder = f"{root_path}/{root_folder}"
logger.add(f"{log_folder}/AMEval/res.log", rotation="500 MB", level="INFO", retention="10 days")

# Pose Error Eval
# specific
output_path = os.path.join(root_path, root_folder, folder_name)

pose_eval_cfg = {
    'root_path': os.path.join(root_path, root_folder),
    'folder_name': folder_name,
    'baseline_name': baseline_name,
    'challenger_name': challenger_name,
    'phi_list': phis,
    'output_path': output_path,
}

pose_eval = PoseAUCEval(pose_eval_cfg)
pose_eval.run()


# AMEval
try:
    am_name = 'am'
    gam_name = 'gam'
    AMP_Thd = [0.6, 0.7, 0.8]
    res_folder = 'ratios'

    am_eval_cfg = {
        'root_path': os.path.join(root_path, root_folder),
        'name': am_name,
        'AMP_Thd': AMP_Thd,
    }

    am_eval = AMEval(am_eval_cfg)
    am_eval.run_AMEval()

    for phi in phis:
        am_eval_cfg['name'] = f'{am_name}+{gam_name}-{phi}'
        am_eval = AMEval(am_eval_cfg)
        am_eval.run_AMEval()
except Exception as e:
    logger.error(f"AMEval failed: {e}")
    