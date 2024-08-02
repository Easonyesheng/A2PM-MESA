'''
Author: EasonZhang
Date: 2023-09-05 14:28:22
LastEditors: EasonZhang
LastEditTime: 2024-07-09 17:03:53
FilePath: /SA2M/hydra-mesa/metric/Evaluation.py
Description: Models to evaluate the performance of the method from Files
    batch-level evaluation

Copyright (c) 2023 by EasonZhang, All Rights Reserved. 
'''

import numpy as np
import os
import sys
sys.path.append("../")
from loguru import logger
import matplotlib.pyplot as plt

from utils.common import test_dir_if_not_create
from utils.geo import relative_pose_error

"""Matching Accuracy Evaluation ===================================================="""
class MMAEval(object):
    """
    """
    cfg_dft = {
        "root_path": "",
        "folder_name":"",
        "baseline_name": "",
        "challenger_name": "",
        "dataset_name": "",
        "phi_list": [0.25, 0.5, 1.0, 2.0, 5.0],
        "px_list": [1,2,3], # order is px-1
        "ratio_postfix": "ratios.txt",
        "name_postfix": "ratios_name.txt",
        "base_ratio_postfix": "ratios.txt",
        "base_name_postfix": "ratios_name.txt",
        "output_path": "",
    }

    def __init__(self, cfg={}) -> None:
        """
        """
        self.config = {**self.cfg_dft, **cfg}
        self.root_path = self.config["root_path"]
        self.folder_name = self.config["folder_name"]
        self.baseline_name = self.config["baseline_name"]
        self.challenger_name = self.config["challenger_name"]
        self.dataset_name = self.config["dataset_name"]
        self.phi_list = self.config["phi_list"]
        self.px_list = self.config["px_list"]
        self.ratio_postfix = self.config["ratio_postfix"]
        self.name_postfix = self.config["name_postfix"]
        self.output_path = self.config["output_path"]
        test_dir_if_not_create(self.output_path)

    def assemble_paths(self):
        """ assemble paths for each phi
        Returns:
            path_dict = {
                phi: { 
                    baseline_path: baseline_path, 
                    baseline_name_path: baseline_name_path,
                    challenger_path: challenger_path,
                    challenger_name_path: challenger_name_path,
                    }
            }
        """
        self.path_dict = {}

        for phi in self.phi_list:
            baseline_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+f"_"+self.ratio_postfix)
            baseline_name_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+f"_"+self.name_postfix)
            challenger_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+f"_phi_{phi}_"+self.ratio_postfix)
            challenger_name_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+f"_phi_{phi}_"+self.name_postfix)
            self.path_dict[phi] = {
                "baseline_path": baseline_path,
                "baseline_name_path": baseline_name_path,
                "challenger_path": challenger_path,
                "challenger_name_path": challenger_name_path,
            }
        
        return self.path_dict

    def eval_single_phi(self, phi):
        """
        """
        baseline_path = self.path_dict[phi]["baseline_path"]
        baseline_name_path = self.path_dict[phi]["baseline_name_path"]
        challenger_path = self.path_dict[phi]["challenger_path"]
        challenger_name_path = self.path_dict[phi]["challenger_name_path"]

        baseline_auc, challenger_auc = self.get_MMA(baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi)

        return baseline_auc, challenger_auc

    def get_MMA(self, baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi=""):
        """ mma file format:
                each line: ratios@px-1 ratios@px-2 ... ratios@px-10

        """
        baseline_ratios = np.loadtxt(baseline_path)
        baseline_name = self.load_name_txt(baseline_name_path)
        logger.info(f'load baseline {self.baseline_name} ratios shape = {baseline_ratios.shape}')

        challenger_ratios = np.loadtxt(challenger_path)
        challenger_name = self.load_name_txt(challenger_name_path)
        logger.info(f'load challenger {self.challenger_name} ratios shape = {challenger_ratios.shape}')

        # get the values according to the px_list
        baseline_ratios_valid = []
        challenger_ratios_valid = []
        for px in self.px_list:
            ratio_id = px - 1
            assert ratio_id < 10, f"{ratio_id} should be less than 10"
            assert ratio_id >= 0, f"{ratio_id} should be greater than 0"

            baseline_ratio = baseline_ratios[:, ratio_id]
            mean_baseline_ratio = np.mean(baseline_ratio)
            baseline_ratios_valid.append(mean_baseline_ratio)

            challenger_ratio = challenger_ratios[:, ratio_id]
            mean_challenger_ratio = np.mean(challenger_ratio)
            challenger_ratios_valid.append(mean_challenger_ratio)

        logger.success(f"get MMA ratios @ px: {self.px_list}")
        logger.success(f"baseline {self.baseline_name} mean ratios: {baseline_ratios_valid}")
        logger.success(f"challenger {self.challenger_name}_{phi} mean ratios: {challenger_ratios_valid}")

        return baseline_ratios_valid, challenger_ratios_valid
    
    def run(self):
        """
        """
        self.assemble_paths()
        for phi in self.phi_list:
            self.eval_single_phi(phi)

    def load_name_txt(self, name_file): 
        """
        """
        rt_names = []
        with open(name_file, "r") as f:
            names = f.readlines()
            for name in names:
                if name=="":continue
                name = name.replace('\n','')
                rt_names.append(name)
        return rt_names


"""Pose Estimation Evaluation ======================================================"""

class PoseAUCEval(object):
    """
        TODO: add the sample function
    """

    cfg_dft = {
        "root_path": "",
        "folder_name":"",
        "baseline_name": "",
        "challenger_name": "",
        "dataset_name": "",
        "phi_list": [0.25, 0.5, 1.0, 2.0, 5.0],
        "err_postfix": "pose_errs.txt",
        "name_postfix": "pose_err_names.txt",
        "base_err_postfix": "pose_errs.txt",
        "base_name_postfix": "pose_err_names.txt",
        "output_path": "",
        "fix_len": None,
    }

    def __init__(self, cfg={}) -> None:
        """
        """
        self.config = {**self.cfg_dft, **cfg}
        self.root_path = self.config["root_path"]
        self.folder_name = self.config["folder_name"]
        self.baseline_name = self.config["baseline_name"]
        self.challenger_name = self.config["challenger_name"]
        self.dataset_name = self.config["dataset_name"]
        self.phi_list = self.config["phi_list"]
        self.err_postfix = self.config["err_postfix"]
        self.name_postfix = self.config["name_postfix"]
        self.base_err_postfix = self.config["base_err_postfix"]
        self.base_name_postfix = self.config["base_name_postfix"]
        self.output_path = self.config["output_path"]
        if self.config["fix_len"] is not None:
            self.fix_len = self.config["fix_len"]
        else:
            self.fix_len = None
        test_dir_if_not_create(self.output_path)        

    def assemble_paths(self):
        """ assemble paths for each phi
        Returns:
            path_dict = {
                phi: { 
                    baseline_path: baseline_path, 
                    baseline_name_path: baseline_name_path,
                    challenger_path: challenger_path,
                    challenger_name_path: challenger_name_path,
                    }
            }
        """
        self.path_dict = {}

        for phi in self.phi_list:
            baseline_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.err_postfix)
            baseline_name_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.name_postfix)
            challenger_path = os.path.join(self.root_path, self.folder_name, f'{self.challenger_name}-{phi}_{self.err_postfix}')
            challenger_name_path = os.path.join(self.root_path, self.folder_name, f'{self.challenger_name}-{phi}_{self.name_postfix}')
            self.path_dict[phi] = {
                "baseline_path": baseline_path,
                "baseline_name_path": baseline_name_path,
                "challenger_path": challenger_path,
                "challenger_name_path": challenger_name_path,
            }
        
        return self.path_dict
    
    def eval_single_phi(self, phi):
        """
        """
        baseline_path = self.path_dict[phi]["baseline_path"]
        baseline_name_path = self.path_dict[phi]["baseline_name_path"]
        challenger_path = self.path_dict[phi]["challenger_path"]
        challenger_name_path = self.path_dict[phi]["challenger_name_path"]

        baseline_auc, challenger_auc = self.get_PoseAUC(baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi)

        return baseline_auc, challenger_auc

    def get_PoseAUC(self, baseline_path, baseline_name_path, challenger_path, challenger_name_path, phi=""):
        """
        """

        logger.success(f"load baseline {self.baseline_name} pose errors from {baseline_path}")
        logger.success(f"load challenger {self.challenger_name}_{phi} pose errors from {challenger_path}")

        try:
            baseline_errs = np.loadtxt(baseline_path)
        except ValueError as e:
            logger.critical(f"{baseline_path} load error")
            
        baseline_name = self.load_name_txt(baseline_name_path)
        # get the idx of nans
        idx_nans_baseline = np.argwhere(np.isnan(baseline_errs))
        baseline_errs[idx_nans_baseline] = [180, 180]

        # get the repeated idx of baseline_name
        idx_repeated = []
        for i in range(len(baseline_name)):
            if baseline_name[i] in baseline_name[:i]:
                idx_repeated.append(i)
        
        # delete the repeated idx
        baseline_errs = np.delete(baseline_errs, idx_repeated, axis=0)
        baseline_name = np.delete(baseline_name, idx_repeated, axis=0)
        
        logger.info(f'load baseline {self.baseline_name} pose errors shape = {baseline_errs.shape}')

        try:
            challenger_errs = np.loadtxt(challenger_path)
        except Exception as e:
            logger.critical(f"{challenger_path} load error")
        challenger_name = self.load_name_txt(challenger_name_path)
        # get the idx of nans
        idx_nans_challenger = np.argwhere(np.isnan(challenger_errs))
        challenger_errs[idx_nans_challenger] = [180, 180]

        # get the repeated idx of challenger_name
        idx_repeated = []
        for i in range(len(challenger_name)):
            if challenger_name[i] in challenger_name[:i]:
                idx_repeated.append(i)
        
        # delete the repeated idx
        challenger_errs = np.delete(challenger_errs, idx_repeated, axis=0)
        challenger_name = np.delete(challenger_name, idx_repeated, axis=0)

        logger.info(f'load challenger {self.challenger_name} pose errors shape = {challenger_errs.shape}')

        # get the common idx of baseline_name and challenger_name
        common_name = np.intersect1d(baseline_name, challenger_name)
        logger.info(f"common name shape {common_name.shape}")

        # get the common idx of baseline_name and challenger_name
        common_idx_baseline = []
        common_idx_challenger = []
        for i in range(len(common_name)):
            common_idx_baseline.append(np.argwhere(baseline_name == common_name[i]))
            common_idx_challenger.append(np.argwhere(challenger_name == common_name[i]))

        common_idx_baseline = np.squeeze(np.array(common_idx_baseline))
        common_idx_challenger = np.squeeze(np.array(common_idx_challenger))

        logger.info(f"common idx baseline shape {common_idx_baseline.shape}")

        # get the common pose errs
        baseline_errs = baseline_errs[common_idx_baseline]
        challenger_errs = challenger_errs[common_idx_challenger]

        # get the fix len
        if self.fix_len is not None:
            baseline_errs = baseline_errs[:self.fix_len]
            challenger_errs = challenger_errs[:self.fix_len]

        logger.info(f"common baseline pose errs shape {baseline_errs.shape}")
        logger.info(f"common challenger pose errs shape {challenger_errs.shape}")

        from utils.geo import aggregate_pose_auc_simp

        # baseline 
        baseline_err_mean = np.mean(baseline_errs, axis=0)
        baseline_auc = aggregate_pose_auc_simp(baseline_errs, self.dataset_name)
        logger.success(f"baseline {self.baseline_name} pose auc: {baseline_auc}")

        # challenger
        challenger_err_mean = np.mean(challenger_errs, axis=0)
        challenger_auc = aggregate_pose_auc_simp(challenger_errs, self.dataset_name)
        logger.success(f"challenger {self.challenger_name}_{phi} pose auc: {challenger_auc}")

        logger.success(f"baseline error mean: {baseline_err_mean}")
        logger.success(f"challenger error mean: {challenger_err_mean}")

        return baseline_auc, challenger_auc

    def run(self):
        """
        """
        best_phi = -1
        best_auc = -1
        
        self.assemble_paths()
        for phi in self.phi_list:
            baseline_auc, challenge_auc = self.eval_single_phi(phi)

            total_diff = 0
            for k in baseline_auc.keys():
                temp_diff = - baseline_auc[k] + challenge_auc[k]
                total_diff += temp_diff
            
            total_diff = total_diff / len(baseline_auc.keys())
            
            if total_diff > best_auc:
                best_auc = total_diff
                best_phi = phi
        
        logger.critical(f"best phi: {best_phi}, best auc diff: {best_auc}")
            

    def run_without_phi(self):
        """
        """
        baseline_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.base_err_postfix)
        baseline_name_path = os.path.join(self.root_path, self.folder_name, self.baseline_name+"_"+self.base_name_postfix)
        challenger_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+"_"+self.err_postfix)
        challenger_name_path = os.path.join(self.root_path, self.folder_name, self.challenger_name+"_"+self.name_postfix)

        self.get_PoseAUC(baseline_path, baseline_name_path, challenger_path, challenger_name_path)
    
    
    def load_name_txt(self, name_file): 
        """
        """
        rt_names = []
        with open(name_file, "r") as f:
            names = f.readlines()
            for name in names:
                if name=="":continue
                name = name.replace('\n','')
                rt_names.append(name)
        return rt_names

        
"""Area Matching Evaluation ======================================================"""

class AMEval(object):
    """ load single area matching result
        - ratio file
        - time file
    """

    AM_default_configs = {
        "root_path": "",
        "name": "",
        "AMP_Thd": 0.8,
        "res_folder": "ratios",
        "AOR_post": "_aor.txt",
        "ACR_post": "_acr.txt",
        "file_name_post": "_ameval_names.txt",
        "time_post": "_area_match_time.txt",
    }

    def __init__(self, configs = {}) -> None:
        """
        """
        self.configs = self.AM_default_configs
        self.configs = {**self.configs, **configs}
        logger.info(f"Initialize the AM Evaluation with configs: {self.configs}")
        self.root_path = self.configs["root_path"]
        self.name = self.configs["name"]
        self.res_folder = self.configs["res_folder"]
        self.AOR_post = self.configs["AOR_post"]
        self.ACR_post = self.configs["ACR_post"]
        self.file_name_post = self.configs["file_name_post"]
        self.time_post = self.configs["time_post"]
        self.AMP_Thd = self.configs["AMP_Thd"]
        self.output_path = os.path.join(self.root_path, "AMEval")
        test_dir_if_not_create(self.output_path)

        self.ratios = None
        self.times = None
        self.names = None
    
    def load_ratio_file(self):
        """ load the ratio file
        """
        ratio_file_name = os.path.join(self.root_path, self.res_folder, self.name + self.AOR_post)
        try:
            ratios = np.loadtxt(ratio_file_name)
        except ValueError as e:
            logger.exception(e)
            logger.critical(f"{ratio_file_name} load error")

        logger.info(f"load ratios with shape: {ratios.shape}")
        self.ratios = ratios
        return ratios
    
    def load_acr_file(self):
        """
        """
        acr_file_name = os.path.join(self.root_path, self.res_folder, self.name + self.ACR_post)
        acrs = np.loadtxt(acr_file_name)
        logger.info(f"load acrs with shape: {acrs.shape}")
        self.acrs = acrs
        return acrs

    def load_time_file(self):
        """ load the time file
        """
        time_file_name = os.path.join(self.root_path, self.res_folder, self.name + self.time_post)
        times = np.loadtxt(time_file_name)
        logger.info(f"load times with shape: {times.shape}")
        self.times = times
        self.img_num = times.shape[0]
        return times
    
    def load_file_name(self):
        """ load the file name
        """
        file_name = os.path.join(self.root_path, self.res_folder, self.name + self.file_name_post)
        names = np.loadtxt(file_name, dtype=str)
        logger.info(f"load names with shape: {names.shape}")
        self.names = names
        return names
    
    def calc_mean_ratio(self):
        """ calculate the mean ratio of the area matching result
        """
        if self.ratios is None:
            self.load_ratio_file()
        mean_ratio = np.mean(self.ratios)
        logger.success(f"mean ratio is {mean_ratio}")
        return mean_ratio

    def calc_mean_time(self):
        """ calculate the mean time of the area matching result
        """
        if self.times is None:
            self.load_time_file()
        mean_time = np.mean(self.times)
        logger.success(f"mean time is {mean_time}")
        return mean_time

    def calc_AMP(self):
        """
        """
        if self.ratios is None:
            self.load_ratio_file()
        area_num = self.ratios.shape[0]
        valid_num = 0
        if type(self.AMP_Thd) is list:
            amp_list = []
            for thd in self.AMP_Thd:
                for ratio in self.ratios:
                    if ratio > thd:
                        valid_num += 1
                AMP = valid_num / area_num
                logger.success(f"AMP@{thd} is {AMP}")
                amp_list.append(AMP)
                valid_num = 0
        else:
            assert self.AMP_Thd is float, f"{self.AMP_Thd} is not float"
            for ratio in self.ratios:
                if ratio > self.AMP_Thd:
                    valid_num += 1
            AMP = valid_num / area_num
            logger.success(f"AMP@{self.AMP_Thd} is {AMP}")
        return AMP
    
    def calc_ACR(self):
        """
        """
        if self.acrs is None:
            self.load_acr_file()
        acr = np.mean(self.acrs)
        logger.success(f"ACR is {acr}")
        return acr

    def run(self):
        """
        """
        self.load_ratio_file()
        self.load_acr_file()
        self.load_time_file()
        self.load_file_name()
        self.calc_mean_ratio()
        self.calc_mean_time()
        self.calc_AMP()
        self.calc_ACR()

    def run_AMEval(self):
        """
        """
        self.load_ratio_file()
        self.load_acr_file()
        self.calc_mean_ratio()
        self.calc_AMP()
        self.calc_ACR()


"""Time Evaluation ======================================================"""

class TimeEval(object):
    """ save as time_name.time_post -> read time_name and report
    """
    time_eval_default_configs = {
        "root_path": "",
        "res_folder": "MMARatios",
        "time_post": "_time.txt",
    }

    def __init__(self, configs={}) -> None:
        """
        """
        self.configs = self.time_eval_default_configs
        self.configs = {**self.configs, **configs}
        logger.info(f"Initialize the Time Evaluation with configs: {self.configs}")
        self.root_path = self.configs["root_path"]
        self.res_folder = self.configs["res_folder"]
        self.time_post = self.configs["time_post"]

        self.time_dict = None

    def load_time_file(self):
        """ load all file with time_post in root_path/res_folder
            1. get all files whose name post is self.time_post
            2. load the time file (record name) and calc the mean time
            3. save the time in self.time_dict
        """
        import glob
        time_files = glob.glob(os.path.join(self.root_path, self.res_folder, "*" + self.time_post))
        logger.info(f"Get {len(time_files)} time files")
        time_dict = {}
        for time_file in time_files:
            name = os.path.basename(time_file).split(".")[0]
            time = np.loadtxt(time_file)
            # calc the mean time
            time = np.mean(time)
            time_dict[name] = time
            logger.info(f"Get {name} mean time: {time}")
        self.time_dict = time_dict
        return time_dict

    def run(self):
        """
        """
        self.load_time_file()