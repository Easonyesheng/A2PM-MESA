'''
Author: EasonZhang
Date: 2024-06-19 23:15:51
LastEditors: EasonZhang
LastEditTime: 2024-07-18 12:10:20
FilePath: /SA2M/hydra-mesa/utils/vis.py
Description: TBD

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''
import numpy as np
import cv2
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import copy

import os
from loguru import logger
import collections

from .img_process import img_to_color

def draw_matched_area(img0, img1, area0, area1, color, out_path, name0, name1, save=True):
    """
    """
    img0 = copy.deepcopy(img0)
    img1 = copy.deepcopy(img1)
    
    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:
        img1 = img_to_color(img1)
    
    W, H = img0.shape[1], img0.shape[0]
    
    out = stack_img(img0, img1)

    draw_matched_area_in_img(out, area0, area1, color)

    if save:
        cv2.imwrite(os.path.join(out_path, f"{name0}_{name1}_matched_area.png"), out)
        logger.info(f"save matched area to {os.path.join(out_path, f'{name0}_{name1}_matched_area.png')}")
    
    return out

def draw_matched_area_list(img0, img1, area0_list, area1_list, out_path, name0, name1, save=True):
    """
    """
    n = len(area0_list)
    assert n == len(area1_list)

    color_map = get_n_colors(n)

    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:  
        img1 = img_to_color(img1)

    W, H = img0.shape[1], img0.shape[0]
    out = stack_img(img0, img1)

    flag = True
    for i in range(n):
        color = color_map[i]
        flag = draw_matched_area_in_img(out, area0_list[i], area1_list[i], color)
    
    if save:
        cv2.imwrite(os.path.join(out_path, f"{name0}_{name1}_matched_areas.png"), out)
    
    return flag

def draw_matched_area_with_mkpts(img0, img1, area0, area1, mkpts0, mkpts1, color, out_path, name0, name1, save=True):
    """
    """
    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:
        img1 = img_to_color(img1)
    
    W, H = img0.shape[1], img0.shape[0]
    
    out = stack_img(img0, img1)

    out = draw_matched_area_in_img(out, area0, area1, color)

    out = draw_mkpts_in_img(out, mkpts0, mkpts1, color)

    if save:
        cv2.imwrite(os.path.join(out_path, f"{name0}_{name1}_matched_area_kpts.png"), out)
        logger.info(f"save matched area to {os.path.join(out_path, f'{name0}_{name1}_matched_area_kpts.png')}")
    
    return out

def paint_semantic(ins0, ins1, out_path="", name0="", name1="", save=True):
    """ fill color by sematic label
    """
    assert len(ins0.shape) == len(ins1.shape) == 2

    H, W = ins0.shape

    label_list = []
    label_color_dict = {}

    for i in range(H):
        for j in range(W):
            temp0 = ins0[i, j]
            temp1 = ins1[i, j]
            if temp0 not in label_list:
                label_list.append(temp0)
            if temp1 not in label_list:
                label_list.append(temp1)

    label_list = sorted(label_list)

    # print(label_list)

    N = len(label_list)
    cmaps_='gist_ncar'
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, N)))

    for i in range(N):
        # black is background
        if label_list[i] == 0:
            label_color_dict[label_list[i]] = [0, 0, 0]
            continue
        c = cmap(i)
        label_color_dict[label_list[i]] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]

    

    outImg0 = np.zeros((H,W,3))
    # print(outImg0.shape)

    for i in range(H):
        for j in range(W):
            outImg0[i, j, :] = label_color_dict[ins0[i,j]]

    outImg1 = np.zeros((H,W,3))
    # print(outImg0.shape)

    for i in range(H):
        for j in range(W):
            outImg1[i, j, :] = label_color_dict[ins1[i,j]]

    if save:
        cv2.imwrite(os.path.join(out_path, "{0}_color.jpg".format(name0)), outImg0)
        cv2.imwrite(os.path.join(out_path, "{0}_color.jpg".format(name1)), outImg1)

    return outImg0, outImg1

def get_n_colors(n):
    """
    """
    label_color_dict = {}

    cmaps_='gist_ncar'
    cmap = matplotlib.colors.ListedColormap(plt.get_cmap(cmaps_)(np.linspace(0, 1, n)))

    for i in range(n):
        c = cmap(i)
        label_color_dict[i] = [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
    
    return label_color_dict

def stack_img(img0, img1):
    """ stack two image in horizontal
    Args:
        img0: numpy array 3 channel
    """
    # assert img0.shape == img1.shape

    if len(img0.shape) == 2:
        img0 = img_to_color(img0)
    if len(img1.shape) == 2:    
        img1 = img_to_color(img1)

    assert len(img0.shape) == 3

    W0, H0 = img0.shape[1], img0.shape[0]
    W1, H1 = img1.shape[1], img1.shape[0]

    H_s = max(H0, H1)
    W_s = W0 + W1

    out = 255 * np.ones((H_s, W_s, 3), np.uint8)

    try:
        out[:H0, :W0, :] = img0.copy()
        out[:H1, W0:, :] = img1.copy()
    except ValueError as e:
        logger.exception(e)
        logger.info(f"img0 shape is {img0.shape}, img1 shape is {img1.shape}")
        logger.info(f"out shape is {out.shape}")
        raise e

    return out

def draw_matched_area_in_img(out, patch0, patch1, color):
    """
    """
    W = out.shape[1] // 2
    W = int(W)
    patch0 = [int(i) for i in patch0]
    patch1_s = [patch1[0]+W, patch1[1]+W, patch1[2], patch1[3]]
    try:
        patch1_s = [int(i) for i in patch1_s]
    except ValueError as e:
        logger.exception(e)
        return False


    # logger.info(f"patch0 are {patch0[0]}, {patch0[1]}, {patch0[2]}, {patch0[3]}")
    # logger.info(f"patch1 are {patch1_s[0]}, {patch1_s[1]}, {patch1_s[2]}, {patch1_s[3]}")

    cv2.rectangle(out, (patch0[0], patch0[2]), (patch0[1], patch0[3]), tuple(color), 3)
    try:
        cv2.rectangle(out, (patch1_s[0], patch1_s[2]), (patch1_s[1], patch1_s[3]), color, 3)
    except cv2.error:
        logger.exception("what?")
        return False

    line_s = [(patch0[0]+patch0[1])//2, (patch0[2]+patch0[3])//2]
    line_e = [(patch1_s[0]+patch1_s[1])//2, (patch1_s[2]+patch1_s[3])//2]

    cv2.line(out, (line_s[0], line_s[1]), (line_e[0], line_e[1]), color=color, thickness=3, lineType=cv2.LINE_AA)

    return True

def plot_matches_lists_lr(image0, image1, matches, outPath, name):
    """
    Args:
        matches: [u0, v0, u1,v1]s
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0:, :] = image1

    color = np.zeros((len(matches), 3), dtype=int)
    color[:, 1] = 255

    for match, c in zip(matches, color):
        c = c.tolist()
        u0, v0, u1, v1 = match
        # print(u0)
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) + W0
        v1 = int(v1)
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)

    path = os.path.join(outPath, name+".jpg")
    # logger.critical(f"save match list img to {path}")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

def plot_matches_lists_ud(image0, image1, matches, outPath, name):
    """
    Args:
        matches: [u0, v0, u1,v1]s
    """
    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[H0:, :W1, :] = image1

    color = np.zeros((len(matches), 3), dtype=int)
    color[:, 1] = 255

    for match, c in zip(matches, color):
        c = c.tolist()
        u0, v0, u1, v1 = match
        # print(u0)
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) 
        v1 = int(v1) + H0
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)

    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

def plot_matches_with_mask_ud(image0, image1, mask, matches, outPath, name, sample_num=500):
    """
    Args:
        mask: 0 -> false match
    """
    # random sample
    if len(matches) > sample_num:
        matches = random.sample(matches, sample_num)

    image0 = img_to_color(image0)
    image1 = img_to_color(image1)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[H0:, :W1, :] = image1

    for i, match in enumerate(matches):
        if mask[i] == 0: c = [0, 0, 255]
        if mask[i] == 1: c = [0, 255, 0]
        if mask[i] == -1: continue

        u0, v0, u1, v1 = match
        u0 = int(u0)
        v0 = int(v0)
        u1 = int(u1) 
        v1 = int(v1) + H0
        cv2.line(out, (u0, v0), (u1, v1), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u0, v0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u1, v1), 2, c, -1, lineType=cv2.LINE_AA)


    path = os.path.join(outPath, name+".jpg")
    logger.info(f"save match list img to {path}")
    cv2.imwrite(path, out)

    return out
