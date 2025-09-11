'''
Author: Eason
Date: 2022-07-08 15:05:44
LastEditTime: 2025-09-10 17:29:35
LastEditors: Easonyesheng preacher@sjtu.edu.cn
Description: utils for image processing
FilePath: /SA2M/hydra-mesa/utils/img_process.py
'''

from loguru import logger
import numpy as np
import cv2

def img_crop(ori_img, crop_list):
    """
        -------> x(u)
        |
        |  img
        |
        v
        y(v)
    Args:
        crop_list: [x_l, x_r, y_u, y_d] == [u_min, u_max, v_min, v_max]
    """
    crop_list_ = [int(x) for x in crop_list]
    u_min, u_max, v_min, v_max = crop_list_
    crop_img = ori_img[v_min:v_max, u_min:u_max] # NOTE the inverse operation
    return crop_img

def img_crop_with_resize(ori_img, crop_list, size):
    """ crop & resize
    Args:
        ori_img: cv2.img
        size = [W, H]
    Returns:
        crop_resized: 
        scales: [W_ori/W_re, H_ori/H_re] -- (u, v)_resized * scale = (u, v)_ori
        offsets: [u_offset, v_offset]
    """
    W_resized, H_resized = size

    crop_list_ = [int(x) for x in crop_list]
    x_l, x_r, y_u, y_d = crop_list_
    offsets = [x_l, y_u]
    crop_img = ori_img[y_u:y_d, x_l:x_r] # NOTE the inverse operation
    H_ori, W_ori = crop_img.shape[0], crop_img.shape[1]
    scales = [W_ori/W_resized, H_ori/H_resized]
    crop_resized = cv2.resize(crop_img, tuple(size))

    return crop_resized, scales, offsets
    
def img_crop_without_Diffscale(ori_img, area, size):
    """ crop a square size 
    Args:
        size=W=H: int:the crop size
    Returns:
        crop_resized: 
        scales: [W_ori/W_re, H_ori/H_re] -- (u, v)_resized * scale = (u, v)_ori
        offset: [u_offset, v_offset]
    """
    H, W = ori_img.shape[0], ori_img.shape[1]
    u_min, u_max, v_min, v_max = area
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2

    u_radius = u_max - u_min
    v_radius = v_max - v_min

    max_len = max(u_radius, v_radius)
    max_len = max(size, max_len)
    max_radius = max_len / 2
    
    # tune center
    if (u_center - max_radius) < 0 and (u_center + max_radius) >= W:
        u_min_f = 0
        u_max_f = W
    elif (u_center - max_radius) < 0:
        u_min_f = 0
        u_max_f = min(u_min_f + max_len, W)
    elif (u_center + max_radius) >= W:
        u_max_f = W
        u_min_f = max(0, u_max_f - max_len) 
    else:
        u_min_f = u_center - max_radius
        u_max_f = u_center + max_radius
    
    if (v_center - max_radius) < 0 and (v_center + max_radius) >= H:
        v_min_f = 0
        v_max_f = H
    elif (v_center - max_radius) < 0:
        v_min_f = 0
        v_max_f = min(H, v_min_f+max_len)
    elif (v_center + max_radius) >= H:
        v_max_f = H
        v_min_f = max(0, v_max_f - max_len)
    else:
        v_min_f = v_center - max_radius
        v_max_f = v_center + max_radius
    
    square_area = [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

    offset = [square_area[0], square_area[2]]

    crop, scale, _ = img_crop_with_resize(ori_img, square_area, [size, size])
    
    return crop, scale, offset

def img_crop_fix_aspect_ratio(ori_img, area, crop_W, crop_H, spread_ratio=1.2):
    """ crop area from ori image
        spread area toward same aspect ratio of crop size
    """
    ori_W, ori_H = ori_img.shape[1], ori_img.shape[0]

    u_min, u_max, v_min, v_max = area
    aspect_ratio = crop_W / crop_H
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2

    # fix the longer side, spread the shorter side
    if (u_max - u_min) / (v_max - v_min) > aspect_ratio:
        W_ori_len = (u_max - u_min)*spread_ratio
        H_ori_len = W_ori_len / aspect_ratio
    else:
        H_ori_len = (v_max - v_min)*spread_ratio
        W_ori_len = H_ori_len * aspect_ratio
    
    # tune the center, ensure the crop area is in the image
    if (u_center - W_ori_len/2) < 0 and (u_center + W_ori_len/2) >= ori_W:
        u_min_f = 0
        u_max_f = ori_W
    elif (u_center - W_ori_len/2) < 0:
        u_min_f = 0
        u_max_f = min(u_min_f + W_ori_len, ori_W)
    elif (u_center + W_ori_len/2) >= ori_W:
        u_max_f = ori_W
        u_min_f = max(0, u_max_f - W_ori_len)
    else:
        u_min_f = u_center - W_ori_len/2
        u_max_f = u_center + W_ori_len/2
    
    if (v_center - H_ori_len/2) < 0 and (v_center + H_ori_len/2) >= ori_H:
        v_min_f = 0
        v_max_f = ori_H
    elif (v_center - H_ori_len/2) < 0:
        v_min_f = 0
        v_max_f = min(v_min_f + H_ori_len, ori_H)
    elif (v_center + H_ori_len/2) >= ori_H:
        v_max_f = ori_H
        v_min_f = max(0, v_max_f - H_ori_len)
    else:
        v_min_f = v_center - H_ori_len/2
        v_max_f = v_center + H_ori_len/2

    crop_area = [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

    offset = [crop_area[0], crop_area[2]]

    crop, scale, _ = img_crop_with_resize(ori_img, crop_area, [crop_W, crop_H])

    return crop, scale, offset

def img_crop_with_padding_expand_square_rt_area(ori_img, area, crop_W, crop_H, spread_ratio=1.2):
    """ @24-03-04 expand the area to a square, put it in the left-top corner and pad to the crop size
    Args:
        ori_img: original image with crop_from_size
        area: [u_min, u_max, v_min, v_max] in crop_from_size
        crop_W, crop_H: crop size, NOTE in here, these sizes are actually the size of evaluation
        spread_ratio: the ratio of spreading the crop area in original image
    Returns:
        crop_area_img: the cropped and resized image
        scale: the scale of crop area in original image - [W_ori/W_re, H_ori/H_re]
        offset: the offset of crop area in original image
    """
    ori_H, ori_W = ori_img.shape[0], ori_img.shape[1]

    if len(ori_img.shape) == 2:   
        crop_area_img = np.zeros((crop_H, crop_W), dtype=np.uint8)
    elif len(ori_img.shape) == 3:
        crop_area_img = np.zeros((crop_H, crop_W, 3), dtype=np.uint8)
    else:
        raise ValueError(f"unsupported image shape {ori_img.shape}")

    # spread the crop area
    area_spread = spread_area(area, ori_W, ori_H, spread_ratio)
    # print(f"spread area {area} to {area_spread}")
    u_min, u_max, v_min, v_max = area_spread

    # expand the area to a square
    square_area = expand_area_to_square(area_spread, ori_W, ori_H)
    u_min, u_max, v_min, v_max = square_area

    # decide whether to resize
    W_ori_len = u_max - u_min
    H_ori_len = v_max - v_min

    if W_ori_len <= crop_W and H_ori_len <= crop_H:
        # no resize, just pad
        scale = [1, 1]
        offset = [u_min, v_min]
        crop_area_img[:v_max-v_min, :u_max-u_min] = ori_img[v_min:v_max, u_min:u_max]
    else:
        # large scale resize
        scale_W = W_ori_len / crop_W
        scale_H = H_ori_len / crop_H

        if scale_W > scale_H:
            scale = [scale_W, scale_W]
            resize_H = int(H_ori_len / scale_W)
            resize_W = crop_W
            offset = [u_min, v_min]
            crop_area_img[:resize_H, :resize_W] = cv2.resize(ori_img[v_min:v_max, u_min:u_max], (resize_W, resize_H))
        else:
            scale = [scale_H, scale_H]
            resize_W = int(W_ori_len / scale_H)
            resize_H = crop_H
            offset = [u_min, v_min]
            crop_area_img[:resize_H, :resize_W] = cv2.resize(ori_img[v_min:v_max, u_min:u_max], (resize_W, resize_H))
        
    return crop_area_img, scale, offset, [u_min, u_max, v_min, v_max]

def img_crop_with_padding_expand_square(ori_img, area, crop_W, crop_H, spread_ratio=1.2):
    """ @24-03-04 expand the area to a square, put it in the left-top corner and pad to the crop size
    Args:
        ori_img: original image with crop_from_size
        area: [u_min, u_max, v_min, v_max] in crop_from_size
        crop_W, crop_H: crop size, NOTE in here, these sizes are actually the size of evaluation
        spread_ratio: the ratio of spreading the crop area in original image
    Returns:
        crop_area_img: the cropped and resized image
        scale: the scale of crop area in original image - [W_ori/W_re, H_ori/H_re]
        offset: the offset of crop area in original image
    """
    ori_H, ori_W = ori_img.shape[0], ori_img.shape[1]

    if len(ori_img.shape) == 2:   
        crop_area_img = np.zeros((crop_H, crop_W), dtype=np.uint8)
    elif len(ori_img.shape) == 3:
        crop_area_img = np.zeros((crop_H, crop_W, 3), dtype=np.uint8)
    else:
        raise ValueError(f"unsupported image shape {ori_img.shape}")

    # spread the crop area
    area_spread = spread_area(area, ori_W, ori_H, spread_ratio)
    # print(f"spread area {area} to {area_spread}")
    u_min, u_max, v_min, v_max = area_spread

    # expand the area to a square
    square_area = expand_area_to_square(area_spread, ori_W, ori_H)
    u_min, u_max, v_min, v_max = square_area

    # decide whether to resize
    W_ori_len = u_max - u_min
    H_ori_len = v_max - v_min

    if W_ori_len <= crop_W and H_ori_len <= crop_H:
        # no resize, just pad
        scale = [1, 1]
        offset = [u_min, v_min]
        crop_area_img[:v_max-v_min, :u_max-u_min] = ori_img[v_min:v_max, u_min:u_max]
    else:
        # large scale resize
        scale_W = W_ori_len / crop_W
        scale_H = H_ori_len / crop_H

        if scale_W > scale_H:
            scale = [scale_W, scale_W]
            resize_H = int(H_ori_len / scale_W)
            resize_W = crop_W
            offset = [u_min, v_min]
            crop_area_img[:resize_H, :resize_W] = cv2.resize(ori_img[v_min:v_max, u_min:u_max], (resize_W, resize_H))
        else:
            scale = [scale_H, scale_H]
            resize_W = int(W_ori_len / scale_H)
            resize_H = crop_H
            offset = [u_min, v_min]
            crop_area_img[:resize_H, :resize_W] = cv2.resize(ori_img[v_min:v_max, u_min:u_max], (resize_W, resize_H))
        
    return crop_area_img, scale, offset

def expand_area_to_square(area, ori_W, ori_H):
    """ expand the area to a square
    Args:
        area: [u_min, u_max, v_min, v_max]
        ori_W, ori_H: the size of original image
    Returns:
        square_area: [u_min, u_max, v_min, v_max]
    """
    u_min, u_max, v_min, v_max = area
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2
    u_len = u_max - u_min
    v_len = v_max - v_min

    max_len = max(u_len, v_len)
    max_radius = max_len / 2

    if u_len > v_len:
        # tune center
        if (v_center - max_radius) < 0 and (v_center + max_radius) >= ori_H:
            v_min_f = 0
            v_max_f = ori_H
        elif (v_center - max_radius) < 0:
            v_min_f = 0
            v_max_f = min(ori_H, v_min_f + max_len)
        elif (v_center + max_radius) >= ori_H:
            v_max_f = ori_H
            v_min_f = max(0, v_max_f - max_len)
        else:
            v_min_f = v_center - max_radius
            v_max_f = v_center + max_radius

        u_min_f = u_center - max_radius
        u_max_f = u_center + max_radius
    else:
        # tune center
        if (u_center - max_radius) < 0 and (u_center + max_radius) >= ori_W:
            u_min_f = 0
            u_max_f = ori_W
        elif (u_center - max_radius) < 0:
            u_min_f = 0
            u_max_f = min(ori_W, u_min_f + max_len)
        elif (u_center + max_radius) >= ori_W:
            u_max_f = ori_W
            u_min_f = max(0, u_max_f - max_len)
        else:
            u_max_f = u_center + max_radius
            u_min_f = u_center - max_radius

        v_min_f = v_center - max_radius
        v_max_f = v_center + max_radius
    
    return [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

def img_crop_with_padding_improve_resolution(ori_img, area, crop_W, crop_H, spread_ratio=1.2):
    """ @24-01-24 put the area in the left-top corner and pad to the crop size
    Args:
        ori_img: original image with crop_from_size
        area: [u_min, u_max, v_min, v_max] in crop_from_size
        crop_W, crop_H: crop size, NOTE in here, these sizes are actually the size of evaluation
        spread_ratio: the ratio of spreading the crop area in original image
    Returns:
        crop_area_img: the cropped and resized image
        scale: the scale of crop area in original image - [W_ori/W_re, H_ori/H_re]
        offset: the offset of crop area in original image
    """
    ori_H, ori_W = ori_img.shape[0], ori_img.shape[1]

    if len(ori_img.shape) == 2:   
        crop_area_img = np.zeros((crop_H, crop_W), dtype=np.uint8)
    elif len(ori_img.shape) == 3:
        crop_area_img = np.zeros((crop_H, crop_W, 3), dtype=np.uint8)
    else:
        raise ValueError(f"unsupported image shape {ori_img.shape}")

    # spread the crop area
    area_spread = spread_area(area, ori_W, ori_H, spread_ratio)
    # print(f"spread area {area} to {area_spread}")
    u_min, u_max, v_min, v_max = area_spread

    # decide whether to resize
    W_ori_len = area_spread[1] - area_spread[0]
    H_ori_len = area_spread[3] - area_spread[2]

    if W_ori_len <= crop_W and H_ori_len <= crop_H:
        # no resize, just pad
        scale = [1, 1]
        offset = [u_min, v_min]
        crop_area_img[:v_max-v_min, :u_max-u_min] = ori_img[v_min:v_max, u_min:u_max]
    else:
        # large scale resize
        scale_W = W_ori_len / crop_W
        scale_H = H_ori_len / crop_H

        if scale_W > scale_H:
            scale = [scale_W, scale_W]
            resize_H = int(H_ori_len / scale_W)
            resize_W = crop_W
            offset = [u_min, v_min]
            crop_area_img[:resize_H, :resize_W] = cv2.resize(ori_img[v_min:v_max, u_min:u_max], (resize_W, resize_H))
        else:
            scale = [scale_H, scale_H]
            resize_W = int(W_ori_len / scale_H)
            resize_H = crop_H
            offset = [u_min, v_min]
            crop_area_img[:resize_H, :resize_W] = cv2.resize(ori_img[v_min:v_max, u_min:u_max], (resize_W, resize_H))
    
    return crop_area_img, scale, offset

def img_crop_with_mask_expand_square(ori_img, area, crop_W, crop_H, spread_ratio=1.2):
    """ @24-03-04 expand the area to a square, then resize the original image to the crop size, get the area in the resized image
    """
    ori_H, ori_W = ori_img.shape[0], ori_img.shape[1]

    # spread the crop area
    area_spread = spread_area(area, ori_W, ori_H, spread_ratio)
    # print(f"spread area {area} to {area_spread}")
    u_min, u_max, v_min, v_max = area_spread

    # expand the area to a square
    square_area = expand_area_to_square(area_spread, ori_W, ori_H)
    u_min, u_max, v_min, v_max = square_area

    # resize the original image to the crop size and get the scale factor
    resized_ori_img = cv2.resize(ori_img, (crop_W, crop_H))
    scale = [ori_W/crop_W, ori_H/crop_H]
    resized_area = [int(u_min/scale[0]), int(u_max/scale[0]), int(v_min/scale[1]), int(v_max/scale[1])]

    # get the area in the resized image
    crop_area_img = np.zeros_like(resized_ori_img)
    crop_area_img[resized_area[2]:resized_area[3], resized_area[0]:resized_area[1]] = resized_ori_img[resized_area[2]:resized_area[3], resized_area[0]:resized_area[1]]

    return crop_area_img, scale, [0, 0]

def spread_area(area, ori_W, ori_H, spread_ratio):
    """ spread the area
    """
    u_min, u_max, v_min, v_max = area
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2
    # print(f"area ori center {u_center}, {v_center}")

    W_ori_len = (u_max - u_min)*spread_ratio
    H_ori_len = (v_max - v_min)*spread_ratio
    # print(f"spread area {area} to {W_ori_len}x{H_ori_len}")

    max_W_len = min(W_ori_len, ori_W)
    max_W_radius = max_W_len / 2
    max_H_len = min(H_ori_len, ori_H)
    max_H_radius = max_H_len / 2
    # print(f"max len {max_W_len},{max_H_len}")

    # tune center
    if (u_center - max_W_radius) < 0 and (u_center + max_W_radius) >= ori_W:
        u_min_f = 0
        u_max_f = ori_W
    elif (u_center - max_W_radius) < 0:
        u_min_f = 0
        u_max_f = min(u_min_f + max_W_len, ori_W)
    elif (u_center + max_W_radius) >= ori_W:
        u_max_f = ori_W
        u_min_f = max(0, u_max_f - max_W_len) 
    else:
        u_min_f = u_center - max_W_radius
        u_max_f = u_center + max_W_radius
    
    # print(f"tune center {u_center}, {v_center} to {u_min_f},{u_max_f}")

    # H
    if (v_center - max_H_radius) < 0 and (v_center + max_H_radius) >= ori_H:
        v_min_f = 0
        v_max_f = ori_H
    elif (v_center - max_H_radius) < 0:
        v_min_f = 0
        v_max_f = min(ori_H, v_min_f+max_H_len)
    elif (v_center + max_H_radius) >= ori_H:
        v_max_f = ori_H
        v_min_f = max(0, v_max_f - max_H_len)
    else:
        v_min_f = v_center - max_H_radius
        v_max_f = v_center + max_H_radius

    return [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

def img_crop_direct(ori_img, area, crop_W, crop_H, spread_ratio=1.2, dfactor=8):
    """crop img with specific size
    Funcs:
        small -> spread to crop size
        big -> resize to crop size
    """
    ori_H, ori_W = ori_img.shape[0], ori_img.shape[1]

    logger.info(f"crop size {crop_W}x{crop_H} of area {area} from img size {ori_W}x{ori_H}")

    u_min, u_max, v_min, v_max = area
    
    u_center = (u_max + u_min) / 2
    v_center = (v_max + v_min) / 2

    W_ori_len = (u_max - u_min)*spread_ratio
    H_ori_len = (v_max - v_min)*spread_ratio

    max_W_len = max(W_ori_len, crop_W)
    max_W_radius = max_W_len / 2
    max_H_len = max(H_ori_len, crop_H)
    max_H_radius = max_H_len / 2

    # tune center
    if (u_center - max_W_radius) < 0 and (u_center + max_W_radius) >= ori_W:
        u_min_f = 0
        u_max_f = ori_W
    elif (u_center - max_W_radius) < 0:
        u_min_f = 0
        u_max_f = min(u_min_f + max_W_len, ori_W)
    elif (u_center + max_W_radius) >= ori_W:
        u_max_f = ori_W
        u_min_f = max(0, u_max_f - max_W_len) 
    else:
        u_min_f = u_center - max_W_radius
        u_max_f = u_center + max_W_radius
    # H
    if (v_center - max_H_radius) < 0 and (v_center + max_H_radius) >= ori_H:
        v_min_f = 0
        v_max_f = ori_H
    elif (v_center - max_H_radius) < 0:
        v_min_f = 0
        v_max_f = min(ori_H, v_min_f+max_H_len)
    elif (v_center + max_H_radius) >= ori_H:
        v_max_f = ori_H
        v_min_f = max(0, v_max_f - max_H_len)
    else:
        v_min_f = v_center - max_H_radius
        v_max_f = v_center + max_H_radius

    crop_area = [int(u_min_f), int(u_max_f), int(v_min_f), int(v_max_f)]

    logger.info(f"acctually crop as {crop_area}")

    offset = [crop_area[0], crop_area[2]]

    crop, scale, _ = img_crop_with_resize(ori_img, crop_area, [crop_W, crop_H])
    
    return crop, scale, offset

def img_to_color(img):
    """
    Args:
        img: np or cv2
    """
    if len(img.shape) == 2 or img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return img

def patch_adjust_with_square_min_limit(crop_list, W, H, min_size):
    """ achieve a square patch
    """
    u_min, u_max, v_min, v_max = crop_list
    u_len = u_max - u_min
    v_len = v_max - v_min

    final_radius = max(min_size, max(u_len, v_len) // 2)

    center_u = (u_max+u_min)/2
    center_v = (v_max+v_min)/2

    u_min_ = max(0, center_u - final_radius)
    u_max_ = min(W, center_u + final_radius)
    v_min_ = max(0, center_v - final_radius)
    v_max_ = min(H, center_v + final_radius)

    return [u_min_, u_max_, v_min_, v_max_]

def patch_adjust_with_size_limits(crop_list, W, H, radius_thd):
    """adjust patch to be square
    """
    u_min, u_max, v_min, v_max = crop_list
    size_max, size_min = radius_thd

    radius_max = max((u_max - u_min)/2, (v_max - v_min)/2)
    radius_max = max(radius_max, size_min)
    radius_max = min(radius_max, size_max)

    center_u = (u_max+u_min)/2
    center_v = (v_max+v_min)/2

    u_min_ = max(0, center_u - radius_max)
    u_max_ = min(W-1, center_u + radius_max)

    v_min_ = max(0, center_v - radius_max)
    v_max_ = min(H-1, center_v + radius_max)

    return [u_min_, u_max_, v_min_, v_max_]

def patch_adjust_fix_size(crop_list, W, H, fix_size=256):
    """ crop with fix size
    """
    u_min, u_max, v_min, v_max = crop_list

    center_u = (u_max+u_min)/2
    center_v = (v_max+v_min)/2

    radius = fix_size // 2
    if center_u - radius < 0:
        center_u = radius
    if center_u + radius > W:
        center_u = W - radius
    
    if center_v - radius < 0:
        center_v = radius
    if center_v + radius > H:
        center_v = H - radius
    
    u_min_ = center_u - radius
    u_max_ = center_u + radius
    v_min_ = center_v - radius
    v_max_ = center_v + radius

    return [u_min_, u_max_, v_min_, v_max_]
    
def resize_im(wo, ho, imsize=None, dfactor=1, value_to_scale=max, enforce=False):
    wt, ht = wo, ho

    # Resize only if the image is too big
    resize = imsize and value_to_scale(wo, ho) > imsize and imsize > 0
    if resize or enforce:
        scale = imsize / value_to_scale(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))

    # Make sure new sizes are divisible by the given factor
    wt, ht = map(lambda x: int(x // dfactor * dfactor), [wt, ht])
    scale = [wo / wt, ho / ht]
    return wt, ht, scale

def read_im(im_path, imsize=None, dfactor=1):
    im = Image.open(im_path)
    im = im.convert('RGB')

    # Resize
    wo, ho = im.width, im.height
    wt, ht, scale = resize_im(wo, ho, imsize=imsize, dfactor=dfactor)
    im = im.resize((wt, ht), Image.BICUBIC)
    return im, scale

def read_im_gray(im_path, imsize=None):
    im, scale = read_im(im_path, imsize)
    return im.convert('L'), scale

def load_gray_scale_tensor(im_path, device, imsize=None, dfactor=1):
    im_rgb, scale = read_im(im_path, imsize, dfactor=dfactor)
    gray = np.array(im_rgb.convert('L'))
    gray = transforms.functional.to_tensor(gray).unsqueeze(0).to(device)
    return gray, scale

def load_gray_scale_tensor_cv(im_path, device, imsize=None, value_to_scale=min, dfactor=1, pad2sqr=False):
    '''Image loading function applicable for LoFTR & Aspanformer. '''

    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    ho, wo = im.shape
    wt, ht, scale = resize_im(
        wo, ho, imsize=imsize, dfactor=dfactor,
        value_to_scale=value_to_scale,
        enforce=pad2sqr
    )
    im = cv2.resize(im, (wt, ht))
    mask = None
    if pad2sqr and (wt != ht):
        # Padding to square image
        im, mask = pad_bottom_right(im, max(wt, ht), ret_mask=True)
        mask = torch.from_numpy(mask).to(device)
    im = transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    return im, scale, mask

def load_img_padding_rt_size(img_path, imsize, dfactor=8, color=False):
    """ load img & padding to square & resize to multiple of dfactor & return size
    """
    if not color:
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ho, wo = im.shape
    else:
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # logger.warning(f"load color image {im.shape}")
        ho, wo = im.shape[0], im.shape[1]

    assert imsize[0] == imsize[1], f"imsize[0] != imsize[1] {imsize}"
    wt, ht, scale = resize_im(
        wo, ho, imsize[0], dfactor=dfactor,
        value_to_scale=max,
        enforce=True
    )
    im = cv2.resize(im, (wt, ht))
    # logger.warning(f"resized image shape {im.shape}")
    im, mask = pad_bottom_right(im, max(wt, ht), ret_mask=True)
    # logger.warning(f"padded image shape {im.shape}")
    return im, mask, (wt, ht)

def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        # cv image is HWC, not CHW
        # test if cv image
        if inp.shape[2] != 3:
            cv_flag = False
            pass
        else:
            # it's a cv image!
            inp = inp.transpose(2, 0, 1)
            cv_flag = True

        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
        
        if cv_flag:
            padded = padded.transpose(1, 2, 0)
            mask = mask.transpose(1, 2, 0)

    else:
        raise NotImplementedError()
    return padded, mask

def load_im_tensor(im_path, device, imsize=None, normalize=True,
                   with_gray=False, raw_gray=False, dfactor=1):
    im_rgb, scale = read_im(im_path, imsize, dfactor=dfactor)

    # RGB  
    im = transforms.functional.to_tensor(im_rgb)
    if normalize:
        im = transforms.functional.normalize(
            im , mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    im = im.unsqueeze(0).to(device)
    
    if with_gray:
        # Grey
        gray = np.array(im_rgb.convert('L'))
        if not raw_gray:
            gray = transforms.functional.to_tensor(gray).unsqueeze(0).to(device)
        return im, gray, scale
    return im, scale

    