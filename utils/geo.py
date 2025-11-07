'''
Author: EasonZhang
Date: 2024-06-19 21:30:03
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2025-11-07 15:11:44
FilePath: /SA2M/hydra-mesa/utils/geo.py
Description: geo utils

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''

import numpy as np
from loguru import logger
import cv2
import copy
import math

def achieve_depth(pt, depth_map):
    """
    """
    [u, v] = pt
    # u_ = int(math.floor(u))
    # v_ = int(math.floor(v))
    u_ = int(round(u))
    v_ = int(round(v))
    W, H = depth_map.shape[1], depth_map.shape[0]
    if u_ < 0 or u_ >= W or v_ < 0 or v_ >= H:
        return -1

    return depth_map[v_, u_] if depth_map[v_, u_] != 0 else -1

def inv_proj(pt, depth, K, pose, Z_neg=0):
    """img to world
    """
    # print(f"depth is {depth}")
    Pt_cam = img2cam(pt, K, depth, Z_neg)
    Pt_cam = Homo_3d(Pt_cam)

    Pt_w = pose @ Pt_cam

    return Pt_w[:3]

def Homo_3d(Pt):
    """ make coordinate homogeneous
    Args:
        Pt: 3x1
    """
    return np.row_stack((Pt, np.array([[1]])))

def img2cam(pt, K, depth, Z_neg=0):
    """
    Args:
        pt [2,] np.array
    Returns:
        3 x 1 mat
    """
    # logger.info(f"K is \n{K}\nd is {depth}, uv is {pt}")
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    [u, v] = pt
    X = (u - cx)*depth / fx
    Y = (v - cy)*depth / fy
    if Z_neg == 0:
        Z = depth
    elif Z_neg == 1:
        Z = -depth
    else:
        raise KeyError
    Pt = np.matrix([[X, Y, Z]]).T
    return Pt

def cam2img(Pt, K, Z_neg=0):
    """
    Args:
        Pt [3x1] XYZ
        Z_neg is for MatterPort3D
    Returns:
        pt [u, v] np.array[2,]
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    X = Pt[0,0]
    Y = Pt[1,0]
    if Z_neg == 0:
        Z = Pt[2,0]
    elif Z_neg == 1:
        Z = -Pt[2,0]
    else:
        raise KeyError

    u = X/Z*fx + cx
    v = Y/Z*fy + cy

    return np.array([u,v])

def calc_euc_dist_2d(pt0, pt1):
    """
    Args:
        pt [u, v]
    """
    # logger.info(f"calc 2d euc dist between {pt0} and {pt1}")
    (u0, v0) = pt0
    (u1, v1) = pt1
    return math.sqrt((u0-u1)**2 + (v0 -v1)**2)

def assert_match_reproj(matches, depth0, depth1, depth_factor, K0, K1, pose0, pose1, thred=0.2, Z_neg=0):
    """ NOTE: For ScanNet whose pose is cam to world & depth factor is 1000
    Args:
        matches: [u0, v0, u1, v1]s
    Returns:
        mask: [0/1/-1]:
            0 - false match
            1 - true match
           -1 - invalid depth
    """
    mask = []
    gt_pts = []
    bad_count = 0
    l = len(matches)

    if len(matches) == 0:
        return [], 100, []
    
    for match in matches:
        try:
            u0, v0, u1, v1 = match
        except ValueError as e:
            logger.exception(e)
            logger.critical(f"invalid match {match}")
            continue
        
        d0 = achieve_depth([u0, v0], depth0)
        d1 = achieve_depth([u1, v1], depth1)

        if d0 == -1 or d1 == -1:
            mask.append(-1)
            gt_pts.append([0,0])
            l-=1
            continue

        d0 /= depth_factor
        d1 /= depth_factor
        
        P_w = inv_proj([u0, v0], d0, K0, pose0, Z_neg)
        P_w = Homo_3d(P_w)
        P0_C1 = np.matmul(pose1.I, P_w)
        p1 = cam2img(P0_C1, K1, Z_neg)
        gt_pts.append(p1)

        dist = calc_euc_dist_2d(p1, np.array([u1, v1]))

        if dist > thred:
            # logger.info(f"bad match with dist = {dist}")
            mask.append(0)
            bad_count += 1
        else:
            mask.append(1)
    
    bad_ratio = bad_count / (l + 1e-5) * 100
    # logger.info(f"assert match bad ratio is {bad_ratio}")
    if l == 0:
        bad_ratio = 100

    return mask, bad_ratio, gt_pts

def calc_pFq(corr, F):
    """ calc q^T F p
    Args:
        corr: [u0, v0, u1, v1]
    """
    assert F.shape == (3,3)

    kpts0 = np.array([corr[0], corr[1]])
    kpts1 = np.array([corr[2], corr[3]])

    uv0H = np.array([kpts0[0], kpts0[1], 1])
    uv1H = np.array([kpts1[0], kpts1[1], 1]) # 1x3

    try:
        up = np.matmul(uv1H, np.matmul(F, uv0H.reshape(3,1)))
    except Exception as e:
        logger.exception(e)

    up = up.flatten()
    up = up**2

    return up

"""NOTE dR, dt here comes from world2cam pose 0to1"""
def np_skew_symmetric(v):
    """
    Args:
        v: Nx3
    """

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def pose2F(pose0, pose1, K0, K1):
    """ calc F from pose
    Args:
        pose0/1: cam2world pose, np.matrix
    """
    K0 = np.matrix(K0)
    K1 = np.matrix(K1)

    # get world2cam pose
    pose0_w2c = pose0.I
    pose1_w2c = pose1.I
    R0 = pose0_w2c[:3, :3]
    R1 = pose1_w2c[:3, :3]
    t0 = pose0_w2c[:3, 3]
    t1 = pose1_w2c[:3, 3]

    # get dR, dt of T_0to1
    dR_0to1 = np.matmul(R1, R0.I)
    dt_0to1 = t1 - np.dot(dR_0to1, t0)
    dt_0to1 = np.array(dt_0to1).reshape(1,3)

    # normalize dt_0to1
    try:
        dt_norm = np.sqrt(np.sum(dt_0to1**2))
    except Exception as e:
        logger.exception(e)
        logger.critical(f"dt_0to1 is {dt_0to1}")
        raise ValueError

    dt_0to1 /= dt_norm

    # get F
    try:
        t_xR = np.matmul(
            np.reshape(np_skew_symmetric(dt_0to1), (3, 3)),
            dR_0to1
        ) # 3x3
    except Exception as e:
        logger.exception(e)
        logger.critical(f"dt_0to1 is {dt_0to1}, dR_0to1 is {dR_0to1}")
        raise ValueError

    F = np.matmul(K1.I.T, np.matmul(t_xR, K0.I))

    return F

def assert_match_qFp(corrs, K0, K1, pose0, pose1, thd):
    """ assert the corrs with Fundamental Matrix
    Args:
        corrs: corrs list
        F: Fundamental Matrix
        thd: threshold
    Returns:
        mask: mask of corrs
        bad_ratio: ratio of bad corrs
    """
    mask = []
    bad_ratio = 0

    F = pose2F(pose0, pose1, K0, K1)

    for corr in corrs:
        pFq = calc_pFq(corr, F)
        if pFq < thd:
            # logger.critical(f"good pFq {pFq} < {thd}")
            mask.append(1)
        else:
            mask.append(0)
            bad_ratio += 1
    bad_ratio /= (len(corrs)+1e-5)

    return mask, bad_ratio

def tune_corrs_size_diff(corrs, src_W0, src_W1, src_H0, src_H1, dst_W0, dst_W1, dst_H0, dst_H1):
    """
    """
    rt_corrs = []
    W0_ratio = dst_W0 / src_W0
    H0_ratio = dst_H0 / src_H0
    W1_ratio = dst_W1 / src_W1
    H1_ratio = dst_H1 / src_H1

    for corr in corrs:
        u0, v0, u1, v1 = corr
        u0_ = u0 * W0_ratio
        v0_ = v0 * H0_ratio
        u1_ = u1 * W1_ratio
        v1_ = v1 * H1_ratio

        rt_corrs.append([u0_, v0_, u1_, v1_])
    
    return rt_corrs

def tune_corrs_size(corrs, src_W, src_H, dst_W, dst_H):
    """
    Args:
        corrs: [[u0, v0, u1, v1], ...]
    """
    rt_corrs = []
    W_ratio = dst_W / src_W
    H_ratio = dst_H / src_H

    for corr in corrs:
        u0, v0, u1, v1 = corr
        u0_ = u0 * W_ratio
        v0_ = v0 * H_ratio
        u1_ = u1 * W_ratio
        v1_ = v1 * H_ratio

        rt_corrs.append([u0_, v0_, u1_, v1_])
    
    return rt_corrs

def nms_for_corrs(corrs, r=3):
    """
    """
    corrs_np = np.array(corrs)
    u0s = corrs_np[:, 0]
    v0s = corrs_np[:, 1]
    u0_min = u0s.min()
    u0_max = u0s.max()
    v0_min = v0s.min()
    v0_max = v0s.max()

    u0_range = np.arange(u0_min, u0_max, r)
    v0_range = np.arange(v0_min, v0_max, r)
    u0_grid, v0_grid = np.meshgrid(u0_range, v0_range)

    mask = np.zeros(len(corrs), dtype=bool)
    for u0, v0 in zip(u0_grid.flatten(), v0_grid.flatten()):
        mask |= (np.abs(u0s - u0) < r/2) & (np.abs(v0s - v0) < r/2)

    corrs_after = corrs_np[mask].tolist()
    return corrs_after

def compute_pose_error_simp(corrs, K0, K1, gt_pose, pix_thd=0.5, conf=0.9999, sac_mode="RANSAC"):
    """
    Args:
        corrs: [corrs]
    Returns:
        error: [R_err, t_err]
    """

    rt_errs = [180, 180]
    corrs = np.array(corrs) # N x 4
    pts0, pts1 = corrs[:, :2], corrs[:, 2:]
    pose_ret = estimate_pose(pts0, pts1, K0, K1, pix_thd, conf, sac_mode=sac_mode)
    logger.success(f'pose est success')

    if pose_ret is None:
        logger.success(f"use len={corrs.shape[0]} corrs to eval pose err failed")
        return rt_errs
    else:
        R, t, inlier = pose_ret
        # logger.info(f"achieve R:\n{R}\n t:{t} \n gt pose is \n{gt_pose}")
        t_err, R_err = relative_pose_error(gt_pose, R, t, 0.0)
        logger.success(f"use len={corrs.shape[0]} corrs to eval pose err = t-{t_err:.4f}, R-{R_err:.4f}")
        rt_errs[0] = R_err
        rt_errs[1] = t_err
    
    return rt_errs

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, sac_mode="RANSAC"):
    """
    Returns:
        ret = (R, t, mask)
    """
    if len(kpts0) < 5:
        return None

    K0 = np.array(K0)
    K1 = np.array(K1)
    
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None] # normalize by minus principal point and divide by focal length
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]]) 

    # compute pose with cv2
    if sac_mode == "RANSAC":
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC, maxIters=10000)
    elif sac_mode == "MAGSAC":
        # logger.critical(f"Use MAGSAC to estimate pose with keypoint number {len(kpts0)} and threshold {ransac_thr}")
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.USAC_MAGSAC, maxIters=10000)
            # FIXME: sometimes stuck in MAGSAC for SPSG (linux 20.04)
    else:
        raise ValueError(f"Invalid sac_mode: {sac_mode}")

    # logger.critical(f'get E is {E}')

    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    try:
        t_err = np.minimum(t_err, 180 - t_err)[0][0]  # handle E ambiguity
    except IndexError:
        t_err = min(t_err, 180 - t_err)

        
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return float(t_err), float(R_err)

def calc_areas_iou(area0, area1):
    """ calc areas iou
    Args:
        area0: [u_min, u_max, v_min, v_max]
        area1: [u_min, u_max, v_min, v_max]
    """
    u_min0, u_max0, v_min0, v_max0 = area0
    u_min1, u_max1, v_min1, v_max1 = area1

    u_min = max(u_min0, u_min1)
    u_max = min(u_max0, u_max1)
    v_min = max(v_min0, v_min1)
    v_max = min(v_max0, v_max1)

    if u_min >= u_max or v_min >= v_max:
        return 0

    area0 = (u_max0 - u_min0) * (v_max0 - v_min0)
    area1 = (u_max1 - u_min1) * (v_max1 - v_min1)
    area = (u_max - u_min) * (v_max - v_min)

    return area / (area0 + area1 - area + 1e-5)

def adopt_K(K, scale):
    """
    Args:
        scale: [scale_x, scale_y]
    """
    fu = K[0,0] * scale[0]
    fv = K[1,1] * scale[1]
    cu = K[0,2] * scale[0]
    cv = K[1,2] * scale[1]
    K_ = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    return K_

def calc_euc_dist_2d(pt0, pt1):
    """
    Args:
        pt [u, v]
    """
    # logger.info(f"calc 2d euc dist between {pt0} and {pt1}")
    (u0, v0) = pt0
    (u1, v1) = pt1
    return math.sqrt((u0-u1)**2 + (v0 -v1)**2)

def tune_mkps_size(mkps, src_W, src_H, dst_W, dst_H):
    """ tune mkps size to dst size
    Args:
        mkps: np.array [[u, v], ...]
    Returns:
        rt_mkps: np.array [[u, v], ...]
    """
    rt_mkps = []
    W_ratio = dst_W / src_W
    H_ratio = dst_H / src_H

    for mkp in mkps:
        u, v = mkp
        u_ = u * W_ratio
        v_ = v * H_ratio

        rt_mkps.append([u_, v_])
    
    return np.array(rt_mkps)

def recover_corrs_offset_scales(corrs, offsets, scales):
    """
        Args:
            corrs: [[u0, v0, u1 v1]s]: NOTE np.ndarray Nx4
            offsets: [u0_offset, v0_offset, u1_offset, v1_offset]
            scales: [u0_scale, v0_scale, u1_scale, v1_scale]
        Returns:
            ori_matches: [[u0, v0, u1, v1]s] NOTE list
    """
    uv0, uv1 = corrs[:, :2], corrs[:, 2:]
    logger.info(f"fuse with offset: {offsets} and scales: {scales}")
    # print("uv0 size:", uv0.shape)
    # print("uv1 size:", uv1.shape)

    u0_offset, v0_offset, u1_offset, v1_offset = offsets
    u0_scale, v0_scale, u1_scale, v1_scale = scales
    # print("offsets: ", u0_offset, v0_offset, u1_offset, v1_offset)

    ori_matches = []
    N = uv0.shape[0]
    assert N == uv1.shape[0]
    for i in range(N):
        u0, v0 = uv0[i]
        u1, v1 = uv1[i]
        u0_ = u0 * u0_scale
        v0_ = v0 * v0_scale
        u1_ = u1 * u1_scale
        v1_ = v1 * v1_scale
        ori_matches.append([u0_+u0_offset, v0_+v0_offset, u1_+u1_offset, v1_+v1_offset])
    
    logger.info(f"fuse {len(ori_matches)} matches")
    # print("ori match\n", ori_matches)
    return ori_matches

def cal_corr_F_and_mean_sd_rt_sd(corrs_raw):
    """
    Args:
        corrs: [[u0, v0, u1, v1], ...]
    """
    corrs = copy.deepcopy(corrs_raw)
    if len(corrs) < 8:
        logger.error(f"too few corrs {len(corrs)}")
        return np.zeros((3,3)), 10000, []
    corrs = np.array(corrs)
    corrs_F0 = corrs[:, :2]
    corrs_F1 = corrs[:, 2:]
    logger.info(f"achieve corrs with shape {corrs_F0.shape} == {corrs_F1.shape} to calc F")
    F, mask = cv2.findFundamentalMat(corrs_F0, corrs_F1, method=cv2.FM_RANSAC,ransacReprojThreshold=1, confidence=0.99)
    # get corrs inliers from mask
    corrs_inliers = corrs[mask.ravel()==1]

    # calc mean sampson dist
    samp_dist, samp_dist_list = calc_sampson_dist_rt_dist(F, corrs_inliers)

    return F, samp_dist, samp_dist_list

def calc_sampson_dist_rt_dist(F, corrs_):
    """
    Args:
        F: 3x3
        corrs: np.ndarray [[u0, v0, u1, v1], ...]
    """
    if type(corrs_) == list:
        corrs = np.array(corrs_)
    else:
        corrs = corrs_

    samp_dist_list = []

    assert len(corrs.shape) == 2 and corrs.shape[1] == 4, f"invalid shape {corrs.shape}"
    uv0, uv1 = corrs[:, :2], corrs[:, 2:]
    uv0_norm = norm_pts_NT(uv0)
    uv1_norm = norm_pts_NT(uv1)
    uv0_h, uv1_h = Homo_2d_pts(uv0_norm), Homo_2d_pts(uv1_norm) # N x 3
    samp_dist = 0

    for i in range(corrs.shape[0]):
        temp_samp_dist = calc_sampson_1_pt(F, uv0_h[i,:], uv1_h[i,:])
        samp_dist_list.append(temp_samp_dist)
        samp_dist += temp_samp_dist
    
    samp_dist /= corrs.shape[0]

    return samp_dist, samp_dist_list

def cal_corr_F_and_mean_sd(corrs_raw):
    """
    Args:
        corrs: [[u0, v0, u1, v1], ...]
    """
    corrs = copy.deepcopy(corrs_raw)
    if len(corrs) < 8:
        logger.error(f"too few corrs {len(corrs)}")
        return np.zeros((3,3)), 10000
    corrs = np.array(corrs)
    corrs_F0 = corrs[:, :2]
    corrs_F1 = corrs[:, 2:]
    logger.info(f"achieve corrs with shape {corrs_F0.shape} == {corrs_F1.shape} to calc F")
    F, mask = cv2.findFundamentalMat(corrs_F0, corrs_F1, method=cv2.FM_RANSAC,ransacReprojThreshold=1, confidence=0.99)
    # get corrs inliers from mask
    corrs_inliers = corrs[mask.ravel()==1]

    # calc mean sampson dist
    samp_dist = calc_sampson_dist(F, corrs_inliers)

    return F, samp_dist

def calc_sampson_dist(F, corrs_):
    """
    Args:
        F: 3x3
        corrs: np.ndarray [[u0, v0, u1, v1], ...]
    """
    if type(corrs_) == list:
        corrs = np.array(corrs_)
    else:
        corrs = corrs_

    assert len(corrs.shape) == 2 and corrs.shape[1] == 4, f"invalid shape {corrs.shape}"
    uv0, uv1 = corrs[:, :2], corrs[:, 2:]
    uv0_norm = norm_pts_NT(uv0)
    uv1_norm = norm_pts_NT(uv1)
    uv0_h, uv1_h = Homo_2d_pts(uv0_norm), Homo_2d_pts(uv1_norm) # N x 3
    samp_dist = 0

    for i in range(corrs.shape[0]):
        samp_dist += calc_sampson_1_pt(F, uv0_h[i,:], uv1_h[i,:])
    
    samp_dist /= corrs.shape[0]

    return samp_dist

def calc_E_from_corrs(corrs, K0, K1, sac_mode="RANSAC", thresh=0.5, conf=0.9999):
    """
    Returns:
        E: [3,3]
        corrs: [[u0, v0, u1, v1]s], inliers
    """
    corrs_np = np.array(corrs)
    kpts0 = corrs_np[:, :2]
    kpts1 = corrs_np[:, 2:]

    if len(kpts0) < 5:
        return None

    K0 = np.array(K0)
    K1 = np.array(K1)
    
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    if sac_mode == "RANSAC":
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    elif sac_mode == "MAGSAC":
        # logger.critical(f"Use MAGSAC to estimate pose")
        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.USAC_MAGSAC)
    else:
        raise ValueError(f"Invalid sac_mode: {sac_mode}")


    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # mask corrs
    mask = mask.ravel() > 0
    corrs = corrs_np[mask]
    # logger.critical(f"corrs shape = {corrs.shape}")

    return E, corrs.tolist()

def recover_F_from_E_K(E, K0, K1):
    """
    """
    K0 = np.mat(K0).I
    K1 = np.mat(K1).I

    E = E / np.linalg.norm(E)
    F = np.matmul(np.matmul(K1.T, E), K0)

    return F    

def list_of_corrs2corr_list(list_of_corrs):
    """
        [[corrs],...] -> [[corr], ...]
    Args:
        list_of_corrs: list of corrs
    Returns:
        rt_corrs: list of corr
    """
    rt_corrs = []
    total_num = 0
    for corrs in list_of_corrs:
        rt_corrs += corrs
        total_num += len(corrs)
    logger.info(f"fuse {total_num} corrs into {len(rt_corrs)} corrs")
    return rt_corrs
    
def is_corr_inside_area(corr, area0, area1):
    """
    """
    u0, v0, u1, v1 = corr
    u_min, u_max, v_min, v_max = area0
    uv0_area0 = False
    if u_min <= u0 <= u_max and v_min <= v0 <= v_max:
        uv0_area0 = True
    else:
        uv0_area0 = False
    
    u_min, u_max, v_min, v_max = area1
    uv1_area1 = False
    if u_min <= u1 <= u_max and v_min <= v1 <= v_max:
        uv1_area1 = True
    else:
        uv1_area1 = False
    
    if uv0_area0 and uv1_area1:
        return True
    else:
        return False

def calc_merge_size_of_areas(areas, ori_W, ori_H):
    """
    Args:
        areas: list of [u_min, u_max, v_min, v_max]
    Returns:
        merge_size: size of merged area
    """
    blank = np.zeros((ori_H, ori_W))
    for area in areas:
        logger.debug(f"area = {area} in {ori_W}x{ori_H}")
        u_min, u_max, v_min, v_max = area
        # turn to int
        u_min, u_max, v_min, v_max = int(u_min), int(u_max), int(v_min), int(v_max)
        blank[v_min:v_max, u_min:u_max] = 1
    
    merge_size = np.sum(blank)
    return merge_size

def filter_corrs_by_F(corrs, F, sd_thd, thd=2.0):
    """
    """
    filtered_corrs = []

    for corr in corrs:
        samp_dist_temp = calc_sampson_dist(F, np.array([corr]))
        if samp_dist_temp <= thd*sd_thd:
            filtered_corrs.append(corr)
    
    return filtered_corrs

def Homo_2d_pts(pts):
    """ 
    Args:
        pts: Nx2 
    """
    N = pts.shape[0]
    homo = np.ones((N, 1))

    pts_homo = np.hstack((pts, homo))
    # logger.info(f"homo pts to {pts_homo.shape}")
    return pts_homo

def norm_pts_NT(pts):
    """ use 8-pts algorithm normalization
    """
    norm_pts = pts.copy()
    N = pts.shape[0]
    
    mean_pts = np.mean(pts, axis=0)
    
    mins_mean_pts = pts - mean_pts

    pts_temp = np.mean(np.abs(mins_mean_pts), axis=0)
    pts_temp+=1e-5

    norm_pts = mins_mean_pts / pts_temp

    # logger.info(f"after norm pts shape = {norm_pts.shape}")

    return norm_pts

def calc_sampson_1_pt(F, uv0H, uv1H):
    """
    Args:
        uviH: 1 x 3
        F: 3 x 3
    Returns:
        (uv1H^T * F * uv0H)^2 / [(F*uv0H)_0^2 + (F*uv0H)_1^2 + (F^T*uv1H)_0^2 + (F^T*uv1H)_1^2]
    """
    uv0H = uv0H.reshape((1,3))
    uv1H = uv1H.reshape((1,3))

    assert uv0H.shape[0] == 1 and uv0H.shape[1] == 3, f"invalid shape {uv0H.shape}"
    assert uv1H.shape[0] == 1 and uv1H.shape[1] == 3, f"invalid shape {uv1H.shape}"

    # logger.debug(f"calc sampson dist use:\n{uv0H}\n{uv1H}")

    up = np.matmul(uv1H, np.matmul(F, uv0H.reshape(3,1)))[0][0]
    up = up**2
    Fx0 = np.matmul(F, uv0H.T)
    FTx1 = np.matmul(F.T, uv1H.T)
    # logger.info(f"Fx1 = {Fx1}\nFTx0 = {FTx0}")
    down = Fx0[0,0]**2 + Fx0[1,0]**2 + FTx1[0,0]**2 + FTx1[1, 0]**2

    logger.debug(f"calc sampson dist use {up} / {down}")
    
    dist = up / (down + 1e-5)


    return dist

def warp_area_by_MC(area0, depth0_, depth1_, K0, K1, pose0, pose1, sample_step=2, depth_factor=1):
    """
    Args:
        area0: [u_min, u_max, v_min, v_max]
        pose0/1: cam2world pose
    Returns:
        area1: [u_min, u_max, v_min, v_max]
    """
    try:
        depth0 = depth0_.copy().astype(np.float32)
        depth1 = depth1_.copy().astype(np.float32)
        depth0 /= depth_factor
        depth1 /= depth_factor
    except Exception as e:
        logger.exception(e)
        raise ValueError

    # sample points in area0
    u_min, u_max, v_min, v_max = area0
    u_min, u_max = int(u_min), int(u_max)
    v_min, v_max = int(v_min), int(v_max)
    u_range = np.arange(u_min, u_max, sample_step)
    v_range = np.arange(v_min, v_max, sample_step)
    u, v = np.meshgrid(u_range, v_range)
    # logger.success(f"sample {u.shape} points")

    # achieve depth0
    depth0 = depth0[v, u]
    # depth0 /= depth_factor
    depth0_mask = depth0 > 0
    depth0 = depth0[depth0_mask]
    u = u[depth0_mask]
    v = v[depth0_mask]
    # logger.success(f"after deptht0 mask get {u.shape} points")

    # achieve 3D points in cam0
    x = (u - K0[0, 2]) * depth0 / K0[0, 0]
    y = (v - K0[1, 2]) * depth0 / K0[1, 1]
    z = depth0

    # transform 3D points from cam0 to world
    pts0 = np.stack([x, y, z], axis=1) # Nx3
    pts0 = np.concatenate([pts0, np.ones_like(pts0[:, :1])], axis=1) # Nx4
    # logger.success(f"pts0 shape is {pts0.shape} ")
    pts0 = np.matmul(pose0, pts0.T).T # Nx4

    # transform 3D points from world to cam1
    pts1 = np.matmul(pose1.I, pts0.T).T # Nx4
    z1_compute = pts1[:, 2]

    # transform 3D points from cam1 to image1
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    u1 = (x1 * K1[0, 0] / z1_compute) + K1[0, 2]
    v1 = (y1 * K1[1, 1] / z1_compute) + K1[1, 2]
    u1 = u1.astype(int)
    v1 = v1.astype(int)
    
    # check if pts1 is in the image
    H, W = depth1.shape
    co_visible_mask = (u1 > 0) & (u1 < W-1) & (v1 > 0) & (v1 < H-1)
    u1 = u1[co_visible_mask]
    v1 = v1[co_visible_mask]
    z1_compute = z1_compute[co_visible_mask]
    # logger.success(f"after depth1 mask get {u1.shape} pts")

    # check depth consistency
    depth1 = depth1[v1, u1]
    # depth1 /= depth_factor
    depth_diff = np.abs(depth1 - z1_compute) / z1_compute
    depth_diff_mask = depth_diff < 0.2

    final_u1 = u1[depth_diff_mask].T
    final_v1 = v1[depth_diff_mask].T

    # logger.success(f"after depth consistency get {final_u1.shape} pts")
    if len(final_u1) < 10: raise ValueError

    # get the area1
    u1_min = final_u1.min()
    u1_max = final_u1.max()
    v1_min = final_v1.min()
    v1_max = final_v1.max()

    area1 = [u1_min, u1_max, v1_min, v1_max]

    # logger.critical(f"warp from \narea0 {area0} to \narea1 {area1} ")

    return area1, area0

def calc_area_intersection(area0, area1):
    """ return the intersection area bbox of area0 and area1
        which should be inside the area0
    """
    u_min0, u_max0, v_min0, v_max0 = area0
    u_min1, u_max1, v_min1, v_max1 = area1
    
    u_min = max(u_min0, u_min1)
    u_max = min(u_max0, u_max1)
    v_min = max(v_min0, v_min1)
    v_max = min(v_max0, v_max1)

    if u_min >= u_max or v_min >= v_max:
        return [0,0,0,0]
    
    return [u_min, u_max, v_min, v_max]

def calc_area_overlap_MC(area0, area1, depth0, depth1, K0, K1, pose0, pose1, depth_factor=1, sample_step=2):
    """
    """
    # warp area0 to img1
    area0_1, _ = warp_area_by_MC(area0, depth0, depth1, K0, K1, pose0, pose1, sample_step, depth_factor)

    # calc the overlap area
    overlap_area = calc_area_intersection(area0_1, area1)

    # calc the overlap ratio
    overlap_ratio = (overlap_area[1] - overlap_area[0]) * (overlap_area[3] - overlap_area[2]) / ((area0[1] - area0[0]) * (area0[3] - area0[2]) + 1e-5)

    return overlap_area, overlap_ratio

def calc_area_match_performence_eff_MC(area0s, area1s, img0, img1, K0, K1, pose0, pose1, depth0, depth1, depth_factor, out_path=""):
    """
    Returns:
        area_cover_ratio: float
        area_overlap_ratio: float
    """
    # 1. calc the overlap ratio of img0 in img1 and img1 in img0
    W, H = depth0.shape[1], depth0.shape[0]

    # resize img0 and img1 to same size of depth
    img0 = cv2.resize(img0, (W, H)) 
    img1 = cv2.resize(img1, (W, H))

    area0_all_img = [5, W-5, 5, H-5]
    area1_all_img = [5, W-5, 5, H-5]

    try:
        overlap_img0_1, _ = calc_area_overlap_MC(area0_all_img, area1_all_img, depth0, depth1, K0, K1, pose0, pose1, depth_factor)
    except ValueError as e:
        logger.exception(e)
        return 0, 0
    
    try:
        overlap_img1_0, _ = calc_area_overlap_MC(area1_all_img, area0_all_img, depth1, depth0, K1, K0, pose1, pose0, depth_factor)
    except ValueError as e:
        logger.exception(e)
        return 0, 0

    # # draw the overlap area in img0 and img1
    # overlap_img0_1 = [int(num) for num in overlap_img0_1]
    # overlap_img1_0 = [int(num) for num in overlap_img1_0]
    # from .vis import draw_matched_area_list
    # draw_matched_area_list(img0, img1, [overlap_img1_0], [overlap_img0_1], W, H, out_path, name0="img0", name1="img1")

    area_cover_0 = fuse_areas(area0s)
    acr_0 = calc_area_intersection(area_cover_0, overlap_img1_0)
    covis_size_0 = (overlap_img1_0[1] - overlap_img1_0[0]) * (overlap_img1_0[3] - overlap_img1_0[2])
    acr_0_ratio = ((acr_0[1] - acr_0[0]) * (acr_0[3] - acr_0[2])) / (covis_size_0 + 1e-5)

    area_cover_1 = fuse_areas(area1s)
    acr_1 = calc_area_intersection(area_cover_1, overlap_img0_1)
    covis_size_1 = (overlap_img0_1[1] - overlap_img0_1[0]) * (overlap_img0_1[3] - overlap_img0_1[2])
    acr_1_ratio = ((acr_1[1] - acr_1[0]) * (acr_1[3] - acr_1[2])) / (covis_size_1 + 1e-5)

    ACR = (acr_0_ratio + acr_1_ratio) / 2

    # 2. calc the overlap ratio of each area pair
    area_overlap_ratios = []
    for area0, area1 in zip(area0s, area1s):
        try:
            _, overlap_ratio01 = calc_area_overlap_MC(area0, area1, depth0, depth1, K0, K1, pose0, pose1, depth_factor)
            _, overlap_ratio10 = calc_area_overlap_MC(area1, area0, depth1, depth0, K1, K0, pose1, pose0, depth_factor)
        except ValueError as e:
            area_overlap_ratios.append(0)
            continue

        area_overlap_ratio = (overlap_ratio01 + overlap_ratio10) / 2
        area_overlap_ratios.append(area_overlap_ratio)

    AOR = area_overlap_ratios

    return ACR, AOR


def fuse_areas(areas):
    """ fuse a list of areas
    """
    areas_np = np.array(areas)
    u_min = np.min(areas_np[:, 0])
    u_max = np.max(areas_np[:, 1])
    v_min = np.min(areas_np[:, 2])
    v_max = np.max(areas_np[:, 3])

    return [u_min, u_max, v_min, v_max]


def R_t_err_calc(T_0to1, R, t, ignore_gt_t_thr=0.0):
    """ same as metric in paper Sparse-to-Local_Dense
        t_err = normed(t_gt)^T * t_est
        R_err = 100/scale_t * angle_err(R)
    """
    t_gt = T_0to1[:3, 3]

    # calc the absolute error of t
    scale_t = np.linalg.norm(t_gt)
    t_err = np.linalg.norm(t_gt) * np.dot(t_gt, t) # / (np.linalg.norm(t_gt) * np.linalg.norm(t))

    # calc the angle error of R
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    angle_err_R = np.rad2deg(np.abs(np.arccos(cos)))

    R_err = 100 / scale_t * angle_err_R

    return float(t_err), float(R_err)

def aggregate_pose_auc_simp(rt_errors, mode="ScanNet"):
    """
    Args:
        rt_errors: np.array Nx2
    """
    thds = [5, 10, 20]

    pose_errs = np.max(rt_errors, axis=1)
    logger.info(f"pose errs max  = {pose_errs.shape}")
    # aucs = error_auc(pose_errs, thds)
    aucs = pose_auc(pose_errs, thds) # superglue

    return aucs

def pose_auc(errors, thresholds):
    logger.info(f"get {errors.shape} errors" )
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    # return aucs
    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def recover_pts_offset_scales(pts, offset, scale):
    """
        Args:
            pts: [[u, v]s]: NOTE np.ndarray Nx2
            offsets: [u_offset, v_offset]
            scales: [u_scale, v_scale]
        Returns:
            ori_pts: [[u, v]s] NOTE np.ndarray Nx2
    """
    uv = pts
    logger.info(f"fuse with offset: {offset} and scales: {scale}")
    # print("uv0 size:", uv0.shape)
    # print("uv1 size:", uv1.shape)

    u_offset, v_offset = offset
    u_scale, v_scale = scale
    # print("offsets: ", u0_offset, v0_offset, u1_offset, v1_offset)

    ori_pts = []
    N = uv.shape[0]
    for i in range(N):
        u, v = uv[i]
        ori_pts.append([u*u_scale+u_offset, v*v_scale+v_offset])
    
    logger.info(f"fuse {len(ori_pts)} pts")

    ori_pts = np.array(ori_pts)
    # print("ori match\n", ori_matches)
    return ori_pts



