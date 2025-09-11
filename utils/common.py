'''
Author: EasonZhang
Date: 2024-06-12 22:42:41
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-09-09 17:03:18
FilePath: /SA2M/hydra-mesa/utils/common.py
Description: TBD

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

from typing import Any, Type
from loguru import logger
import numpy as np

# from nuplan
def validate_type(instantiated_class: Any, desired_type: Type[Any]) -> None:
    """
    Validate that constructed type is indeed the desired one
    :param instantiated_class: class that was created
    :param desired_type: type that the created class should have
    """
    assert isinstance(
        instantiated_class, desired_type
    ), f"Class to be of type {desired_type}, but is {type(instantiated_class)}!"

def test_dir_if_not_create(path):
    """ create folder

    Args:

    Returns:
    """
    import os
    if os.path.isdir(path):
        return True
    else:
        logger.info(f'Create New Folder: {path}')
        os.makedirs(path)
        return True


def norm_conf(conf):
    """ normalize the conf to [0, 1]
    """
    if conf is None:
        return conf
    conf = (conf - np.min(conf)) / (np.max(conf) - np.min(conf)+1e-6)
    return conf

def clean_mat_idx(mat, idx):
    """ delete the mat value in mat[idx, :] and mat[:, idx]
        shrink the mat shape by 1
    """
    assert mat.shape[0] > idx, f"mat.shape: {mat.shape} < idx: {idx}"
    mat = np.delete(mat, idx, axis=0)
    mat = np.delete(mat, idx, axis=1)
    if mat.shape[0] == 0:
        mat = None
    return mat

def expand_mat_by1(mat):
    """ expand the mat by 1 (add a row and a column)
    """
    if mat is None:
        mat = np.zeros((1, 1))
        return mat

    mat = np.pad(mat, ((0, 1), (0, 1)), 'constant', constant_values=0)
    return mat