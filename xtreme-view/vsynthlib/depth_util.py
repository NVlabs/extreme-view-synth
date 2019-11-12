"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/NVlabs/extreme-view-synth.
Authors: Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H. Kim, and Jan Kautz
"""

import numpy as np
import cv2
from scipy.signal import argrelextrema

def my_comparator_greater(x1, x2):
    res_1 = np.greater_equal(x1, x2)
    # res_2 = x1 > DT_THRESHOLD
    # res = np.logical_and(res_1, res_2)
    return res_1


def generate_depthmap(depth_prob, min_disp, disp_step, depht_at_inifinity,
                      color_max_val=None, use_argmax=True):
    if use_argmax:
        depth_idx = np.argmax(depth_prob, axis=0)
    else:
        depth_idx = np.argmin(depth_prob, axis=0)


    img_depth = depth_idx*disp_step
    zero_disp = (depth_idx == 0)
    img_depth = 1.0 / (img_depth + min_disp)
    img_depth[zero_disp] = depht_at_inifinity

    if color_max_val is None:
        img_depth_colored, color_max_val = apply_colormap_to_depth(img_depth, depht_at_inifinity)
        return img_depth, img_depth_colored, zero_disp, color_max_val

    else:
        img_depth_colored  = apply_colormap_to_depth(img_depth, depht_at_inifinity, max_depth=color_max_val)
        return img_depth, img_depth_colored, zero_disp



def apply_colormap_to_depth(img_depth, depth_at_infinity, max_depth=None, max_percent=95, RGB=True):
    img_depth_colored = img_depth.copy()
    m = np.min(img_depth_colored)
    M = np.max(img_depth_colored)

    if max_depth is None:
        valid_mask = img_depth_colored < depth_at_infinity
        valid_mask = np.logical_and(valid_mask, np.logical_not(np.isinf(img_depth)))
        valid_mask = np.logical_and(valid_mask, img_depth != 0.0)
        list_data = img_depth[valid_mask]

        hist, bins = np.histogram(list_data, bins=20)
        n_data = len(list_data)
        threshold_max = n_data * float(max_percent)/100.0
        sum_hist = 0

        for bin_idx, hist_val in enumerate(hist):
            sum_hist += hist_val
            if sum_hist > threshold_max:
                M = bins[bin_idx + 1]
                break
    else:
        M = max_depth

    img_depth_colored[img_depth_colored > M] = M
    img_depth_colored = (img_depth_colored - m) / (M - m)
    img_depth_colored = (img_depth_colored * 255).astype(np.uint8)
    img_depth_colored = cv2.applyColorMap(img_depth_colored, cv2.COLORMAP_JET)

    if RGB:
        img_depth_colored = cv2.cvtColor(img_depth_colored, cv2.COLOR_BGR2RGB)

    if max_depth is None:
        return img_depth_colored, M
    else:
        return img_depth_colored



def fetch_patches_VNP(y, x, p_size, dest_cam,
                  img_synth, list_src_img, list_src_cam,
                  depth_map_P1, depth_map_P2, return_None=True):

    #########################
    # define the input and the output
    #########################
    # t_input = np.zeros(shape=(p_size, p_size, 27))
    # list_src_cam_IDs_ref = dest_cam['list_src_cam_IDs_ref']
    chs_for_fg_patches = 3*len(list_src_img)
    t_input = np.zeros(shape=(p_size, p_size, 3 + 2*chs_for_fg_patches))

    t_input_synth = np.zeros(shape=(p_size, p_size, 3))
    list_t_candi_patch = []



    #########################
    # set output
    #########################
    X_grid, Y_grid = np.meshgrid(np.arange(x, x + p_size),
                                 np.arange(y, y + p_size))


    #########################
    # set input
    #########################
    synth_patch = img_synth[Y_grid, X_grid]
    t_input_synth = synth_patch
    # cv2.imshow('synth_patch', synth_patch)

    # get the reference camera params
    inv_int_dest = np.linalg.inv(dest_cam['intrinsic'])
    inv_ext_dest = np.linalg.inv(dest_cam['extrinsic'])


    for count in range(len(list_src_img)):
        # get the target camera params
        cam_i = list_src_cam[count]
        ext_i = cam_i['extrinsic']
        int_i = cam_i['intrinsic']

        planar_patch_P1_i = backward_warp_center_depth(y, x, depth_map_P1, list_src_img,
                                          p_size, ext_i, int_i, count,
                                                       inv_int_dest, inv_ext_dest)
        planar_patch_P2_i = backward_warp_center_depth(y, x, depth_map_P2, list_src_img,
                                          p_size, ext_i, int_i, count,
                                                       inv_int_dest, inv_ext_dest)

        if return_None:
            if planar_patch_P1_i is None or planar_patch_P2_i is None:
                return None, None

        list_t_candi_patch.append(planar_patch_P1_i)

        z_1 = depth_map_P1[y, x]
        z_2 = depth_map_P2[y, x]
        diff_z = np.abs(z_1 - z_2)/z_1*100

        if diff_z > 2:
            list_t_candi_patch.append(planar_patch_P2_i)


    # change shape and subtract 0.5
    t_input_synth = np.moveaxis(t_input_synth, -1, 0)
    t_input_synth -= 0.5

    for i in range(len(list_t_candi_patch)):
        t_candi_patch = list_t_candi_patch[i]
        t_candi_patch = np.moveaxis(t_candi_patch, -1, 0)
        t_candi_patch -= 0.5
        list_t_candi_patch[i] = t_candi_patch

    #########################
    return t_input_synth, list_t_candi_patch




def backward_warp_center_depth(y_coord, x_coord, dmap, list_src_img,
                              patch_size, ext_i, int_i, src_idx,
                              inv_int_ref, inv_ext_ref):

    z_coord = dmap[int(y_coord + patch_size/2), int(x_coord + patch_size/2)]
    # if z_coord == 0.0:
    #     return None

    height, width = dmap.shape
    X_grid, Y_grid = np.meshgrid(np.arange(x_coord, x_coord + patch_size),
                                 np.arange(y_coord, y_coord + patch_size))
    Z_grid = np.ones(shape=X_grid.shape) * z_coord
    X_grid = np.multiply(X_grid, Z_grid)
    Y_grid = np.multiply(Y_grid, Z_grid)

    points = np.array([X_grid.reshape(-1), Y_grid.reshape(-1), Z_grid.reshape(-1)])
    points = np.matmul(inv_int_ref, points)
    points = np.vstack([points, np.ones((1, patch_size *patch_size))])
    points = np.matmul(inv_ext_ref, points)
    points = np.matmul(ext_i, points)
    points = points[0:3]
    points = np.matmul(int_i, points)
    Xi = points[0] /points[2]
    Yi = points[1] /points[2]
    Xi = Xi.reshape((patch_size, patch_size))
    Yi = Yi.reshape((patch_size, patch_size))

    # handle some exceptions
    invalid_Xi_zero = Xi < 0
    Xi[invalid_Xi_zero] = 0
    invalid_Xi_width = Xi >= width
    Xi[invalid_Xi_width] = width - 1
    invalid_Xi = np.logical_or(invalid_Xi_zero, invalid_Xi_width)

    invalid_Yi_zero = Yi < 0
    Yi[invalid_Yi_zero] = 0
    invalid_Yi_height = Yi >= height
    Yi[invalid_Yi_height] = height - 1
    invalid_Yi = np.logical_or(invalid_Yi_zero, invalid_Yi_height)
    invalid_XYi = np.logical_or(invalid_Xi, invalid_Yi)

    # do warping
    img_i = list_src_img[src_idx]
    Xi = Xi.astype(np.int)
    Yi = Yi.astype(np.int)


    planar_patch_i = img_i[Yi, Xi]
    planar_patch_i[invalid_XYi] = 0

    return planar_patch_i
