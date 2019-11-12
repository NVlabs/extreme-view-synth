"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/NVlabs/extreme-view-synth.
Authors: Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H. Kim, and Jan Kautz
"""

import os
import numpy as np
import cv2
from scipy.signal import argrelextrema
import imageio
from shutil import copyfile
import torch
import sys

from vsynthlib import deepmvs_wrapper
from vsynthlib import depth_util
from vsynthlib import refinement

class VSynth(object):

    DT_THRESHOLD = 0.075
    # DT_THRESHOLD = 0.010
    VISIBILITY_TEST_THRESHOLD = 0.10
    MHW_THRESHOLD = 0.05
    DEPTH_AT_INFINITY = 9999999999.0

    def __init__(self, list_src_img, list_src_cams, out_dir, deepmvs_obj,
                 list_cam_params_for_vpath=[],
                 n_virtual_cams=10,
                 list_src_names=[], list_depth=[], n_depths=100,
                 vcam_mode='extrap_naive',
                 write_out=True,
                 mode_colmap=False):

        self.list_src_img = list_src_img
        self.list_src_cams = list_src_cams
        self.list_depth = list_depth
        self.out_dir = out_dir

        self.depth_estimator = deepmvs_obj

        if len(list_cam_params_for_vpath) == 0:
            self.list_src_cams_for_vpath = list_src_cams
        else:
            self.list_src_cams_for_vpath = list_cam_params_for_vpath

        if len(list_src_names) == 0:
            self.list_src_names = []
            for i in range(len(list_src_img)):
                self.list_src_names.append('%04d'%i)
        else:
            self.list_src_names = list_src_names

        self.params = dict()
        self.params['n_virtual_cams'] = n_virtual_cams
        self.params['n_depths'] = n_depths
        height, width, _ = list_src_img[0].shape
        self.params['height'] = height
        self.params['width'] = width
        self.params['vcam_mode'] = vcam_mode
        self.params['write_out'] = write_out
        self.params['mode_colmap'] = mode_colmap

        if self.params['write_out']:
            self.set_out_dirs()
            self.save_inputs()


    def set_out_dirs(self):
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.out_dir_dp = self.out_dir + '/dp'
        self.out_dir_input = self.out_dir + '/input'
        self.out_dir_output = self.out_dir + '/output'
        self.out_dir_synth_obj = self.out_dir + '/synth_obj'
        self.out_dir_vcams = self.out_dir + '/vcam'
        self.out_dir_refinement = self.out_dir + '/refinement'
        self.out_dir_2nd_synth_obj = self.out_dir + '/acc_synth_obj'
        self.out_dir_2nd_output = self.out_dir + '/acc_output'
        self.out_dir_2nd_refinement = self.out_dir + '/acc_refinement'
        self.out_dir_back_to_front_synth = self.out_dir + '/back_to_front'

        if not os.path.exists(self.out_dir_back_to_front_synth):
            os.mkdir(self.out_dir_back_to_front_synth)


    def save_inputs(self):
        if not os.path.exists(self.out_dir_input):
            os.mkdir(self.out_dir_input)

        for i in range(len(self.list_src_img)):
            img_i = self.list_src_img[i]
            cam_i = self.list_src_cams[i]

            # save dp_i to a file
            f_img_i = '%s/img_%s.png'%(self.out_dir_input, self.list_src_names[i])
            f_cam_i = '%s/cam_%s.npy'%(self.out_dir_input, self.list_src_names[i])

            np.save(f_cam_i, cam_i)
            imageio.imwrite(f_img_i, img_i)

    def save_cam_params(self, vcam_path):
        if not os.path.exists(self.out_dir_vcams):
            os.mkdir(self.out_dir_vcams)

        for cam_idx, vcam in enumerate(vcam_path):
            f_vcam = '%s/%04d.npy' % (self.out_dir_vcams, cam_idx)
            np.save(f_vcam, vcam)




    def compute_depth_probability(self, load_if_exists=True,
                                  LF_dataset_obj=None, hint=''):
        if self.params['write_out']:
            if not os.path.exists(self.out_dir_dp):
                os.mkdir(self.out_dir_dp)

        # compute the depth range
        self.compute_depth_range(hint=hint)


        if self.params['write_out'] and load_if_exists:
            # check if there are precomputed depth probabilities
            list_loaded_dp = []
            all_loaded = True
            for i in range(len(self.list_src_img)):
                filename = '%s/dp_%s.npy' % (self.out_dir_dp, self.list_src_names[i])
                if os.path.exists(filename):
                    dp_i = np.load(filename)
                    print(filename)
                    list_loaded_dp.append(dp_i)

                    if i == 0:
                        _, dmap_color_i, _, color_depth_max = depth_util.generate_depthmap(dp_i,
                                                                                           self.params['min_disp'],
                                                                                           self.params['disp_step'],
                                                                                           self.DEPTH_AT_INFINITY)
                        self.color_depth_max = color_depth_max

                else:
                    all_loaded = False

            # if exists, load and return
            if all_loaded:
                print('The depth probabilities are loaded!')
                self.list_depth_prob = list_loaded_dp
                return

        self.list_depth_prob = []

        for i in range(len(self.list_src_img)):


            import time
            start_time = time.time()

            dp_i = self.depth_estimator.compute(self.list_src_img, self.list_src_cams, i,
                                           self.params['min_disp'],
                                           self.params['disp_step'],
                                           self.DEPTH_AT_INFINITY)

            elapsed_time = time.time() - start_time
            print('DeepMVS: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

            self.list_depth_prob.append(dp_i)

            # save dp_i to a file
            if self.params['write_out']:
                filename = '%s/dp_%s.npy' % (self.out_dir_dp, self.list_src_names[i])
                print(filename)
                np.save(filename, dp_i)

                # gen depth map
                if i == 0:
                    _, dmap_color_i, _, color_depth_max = depth_util.generate_depthmap(dp_i,
                                                                        self.params['min_disp'],
                                                                        self.params['disp_step'],
                                                                        self.DEPTH_AT_INFINITY)
                    self.color_depth_max = color_depth_max
                else:
                    _, dmap_color_i, _ = depth_util.generate_depthmap(dp_i,
                                                                        self.params['min_disp'],
                                                                        self.params['disp_step'],
                                                                        self.DEPTH_AT_INFINITY,
                                                                        color_max_val=color_depth_max)
                filename = '%s/dmap_%s.png' % (self.out_dir_dp, self.list_src_names[i])
                imageio.imwrite(filename, dmap_color_i)


    def compute_depth_range(self, hint=''):
        max_depth = 0.0
        min_depth = 9999999.0


        list_data = np.array([])
        for img_depth in self.list_depth:
            valid_mask = np.logical_not(np.isinf(img_depth))
            valid_mask = np.logical_and(valid_mask, img_depth != 0.0)
            list_data = np.append(list_data, img_depth[valid_mask])

        hist, bins = np.histogram(list_data, bins=100)
        n_data = len(list_data)
        threshold_max = n_data*0.98
        threshold_min = n_data*0.02
        sum_hist = 0
        min_depth = np.min(list_data)
        max_depth = np.max(list_data)
        print('min: %f / max: %f (before histogram)'%(min_depth, max_depth))

        min_found = False
        for bin_idx, hist_val in enumerate(hist):
            sum_hist += hist_val
            if not min_found  and sum_hist > threshold_min:
                if bin_idx >= 1:
                    min_depth = bins[bin_idx - 1]
                else:
                    min_depth = bins[bin_idx]
                min_found = True

            if sum_hist > threshold_max:
                max_depth = bins[bin_idx + 1]
                break

        # museum (2.5, 15.0)
        # our_0046 (30, 800)
        # min_depth = 2.5
        # max_depth = 15.0

        if hint == 'museum1':
            min_depth = 2.5
            max_depth = 15.0

        print('min: %f / max: %f (after histogram)' % (min_depth, max_depth))


        print('max depth: %f' % (max_depth))
        print('min depth: %f' % (min_depth))
        max_disp = 1.0 / (min_depth + 1e-06)
        min_disp = 1.0 / (max_depth + 1e-06)
        disp_step = (max_disp - min_disp) / (self.params['n_depths'] - 1)
        print('disp step: ' + str(disp_step))

        # save to params
        self.params['max_depth'] = max_depth
        self.params['min_depth'] = min_depth
        self.params['max_disp'] = max_disp
        self.params['min_disp'] = min_disp
        self.params['disp_step'] = disp_step

        return max_depth, min_depth, max_disp, min_disp, disp_step


    def do(self, list_idx=None, MHW=False, save_dp=False, MHW_SRCV_WEIGHT=False, list_todo_index=[],
           winner_takes_all=False):

        if self.params['write_out']:
            # create directories
            if not os.path.exists(self.out_dir_output):
                os.mkdir(self.out_dir_output)

            if not os.path.exists(self.out_dir_synth_obj):
                os.mkdir(self.out_dir_synth_obj)


        for cam_idx, cam in enumerate(self.list_vcams):

            if list_todo_index != [] and  not (cam_idx in list_todo_index):
                continue
            synth_obj = self.do_single_image(cam, cam_idx,
                                                 winner_takes_all)

            if self.params['write_out']:
                if list_idx is None:
                    id = cam_idx
                else:
                    id = list_idx[cam_idx]

                # save
                f_synth_obj = '%s/%04d.npz'%(self.out_dir_synth_obj, id)
                np.savez_compressed(f_synth_obj, synth_obj)
                f_img_synth = '%s/vsynth_%04d.png'%(self.out_dir_output, id)
                imageio.imwrite(f_img_synth, synth_obj['img_synth'])
                f_depth_synth = '%s/dmap_%04d.png' % (self.out_dir_output, id)

                if not MHW:
                    imageio.imwrite(f_depth_synth,
                                    depth_util.apply_colormap_to_depth(synth_obj['depth_map'],
                                                                       self.DEPTH_AT_INFINITY,
                                                                       max_depth=self.color_depth_max))
                if save_dp:
                    f_dp_synth = '%s/dp_%04d.npy'%(self.out_dir_output, id)
                    np.save(f_dp_synth, synth_obj['dp'])


    def do_single_image(self, dest_cam, idx=0,
                        winner_takes_all=False,
                        save_dp=False):
        # transform the depth probability

        import time
        start = time.time()


        dp_dest, list_warped_prob\
            = transform_cost_volume_cuda(self.list_src_cams, self.list_depth_prob,
                                        dest_cam,
                                        self.params['n_depths'],
                                        self.params['height'],
                                        self.params['width'],
                                        self.params['min_disp'],
                                        self.params['disp_step'],
                                        self.params['max_depth'],
                                        self.params['min_depth'])

        end = time.time()
        print("Transform_cost_volume() took " + str(end - start))


        # generate PSV
        start = time.time()
        PSV_dest = build_PSV(self.list_src_img, self.list_src_cams,
                             dest_cam,
                             self.params['n_depths'],
                             self.params['height'],
                             self.params['width'],
                             self.params['min_disp'],
                             self.params['disp_step'],
                             self.params['max_depth'],
                             USE_DICT=True)
        end = time.time()
        print("build_PSV() took " + str(end - start))

        # perform view synthesis
        start = time.time()
        img_synth, list_new_vies, visibility_map,\
        depth_map_P1, depth_map_color_P1,\
        depth_map_P2, depth_map_color_P2\
            = synthesize_a_view(dest_cam, PSV_dest, dp_dest,
                                self.list_src_img, self.list_src_cams,
                                self.list_depth_prob,
                                self.params['min_disp'],
                                self.params['disp_step'],
                                self.DEPTH_AT_INFINITY,
                                self.params['height'],
                                self.params['width'],
                                self.params['n_depths'],
                                with_ULR_weight=True,
                                color_max_depth=self.color_depth_max,
                                winner_takes_all=winner_takes_all)



        end = time.time()
        print("synthesize_a_view() took " + str(end - start))


        # save
        synth_obj = {'img_synth': img_synth,
                     'visibility_map': visibility_map,
                     'depth_map': depth_map_P1,
                     'depth_map_P1': depth_map_P1,
                     'depth_map_P2': depth_map_P2,
                     'view_idx': idx,
                     'dest_cam': dest_cam}

        if save_dp:
            synth_obj['dp'] = dp_dest

        return synth_obj

    def refine(self, filename_weight, do_stereo=False,
               patch_size=64, list_todo_index=[],
               custom_outdir=''):
        refiner = refinement.DeepViewRefiner(filename_weight,
                                            self.out_dir,
                                            self.out_dir_refinement,
                                             patch_size=patch_size)

        for cam_idx, cam in enumerate(self.list_vcams):
            if list_todo_index != [] and  not (cam_idx in list_todo_index):
                continue
            f_synth_obj = '%s/%04d.npz'%(self.out_dir_synth_obj, cam_idx)
            synth_obj = np.load(f_synth_obj, allow_pickle=True)
            synth_obj = synth_obj['arr_0'].item()

            refiner.do(synth_obj, self.list_src_img, self.list_src_cams, cam_idx,
                       do_stereo=do_stereo, custom_outdir=custom_outdir)



def build_PSV(list_src_img, list_src_cams,
              cam_dest, num_depths, height, width,
              min_disp, disp_step, max_depth, USE_DICT=False):

    n_neighbors = len(list_src_img)

    if USE_DICT:
        PSV = {}
    else:
        PSV = np.zeros(shape=[n_neighbors, num_depths, height, width, 3], dtype=np.float32)

    int_dest = cam_dest['intrinsic']
    fx_dest = int_dest[0, 0]
    fy_dest = int_dest[1, 1]
    cx_dest = int_dest[0, 2]
    cy_dest = int_dest[1, 2]
    ext_dest = cam_dest['extrinsic']
    inv_ext_dest = np.linalg.inv(ext_dest)

    # for each neighbor image
    counter_img = 0
    for i in range(len(list_src_img)):
        img_i = list_src_img[i]
        cam_i = list_src_cams[i]
        # get the parameters
        int_i = cam_i['intrinsic']
        fx_i = int_i[0, 0]
        fy_i = int_i[1, 1]
        cx_i = int_i[0, 2]
        cy_i = int_i[1, 2]
        ext_i = cam_i['extrinsic']

        # 4 Corners on the virtual camera to get te 4 rays that intersect with the depth plane
        src_pts = np.reshape([0, 0,
                              width, 0,
                              width, height,
                              0, height], (4, 2))

        if USE_DICT:
            PSV_i = np.zeros(shape=[num_depths, height, width, 3], dtype=np.float32)

        # for each depth plane
        for d in range(num_depths):

            disp = d * disp_step + min_disp
            if d == 0:
                depth = max_depth
            else:
                depth = 1.0 / disp

            # print(depth)

            # compute dst points
            dst_pts = np.zeros((4, 2))
            counter_pt = 0
            for p in src_pts:
                p_3D_ref = np.asarray([(depth * p[0] - depth * cx_dest) / fx_dest,
                                       (depth * p[1] - depth * cy_dest) / fy_dest,
                                       depth])
                p_4D_ref = np.array([p_3D_ref[0], p_3D_ref[1], p_3D_ref[2], 1.0])
                p_4D_world = inv_ext_dest.dot(p_4D_ref)
                p_4D_i = ext_i.dot(p_4D_world)
                dst = np.asarray([cx_i + fx_i * p_4D_i[0] / p_4D_i[2], cy_i + fy_i * p_4D_i[1] / p_4D_i[2]])
                dst_pts[counter_pt, :] = dst.squeeze()
                counter_pt += 1

            # compute homography
            M, mask = cv2.findHomography(dst_pts, src_pts)
            # warp the image
            result = cv2.warpPerspective(img_i, M, (width, height),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            # cv2.imshow("img_ref", img_ref)
            # cv2.imshow("PSV of img %02d" % (i), result)
            # cv2.waitKey()
            if USE_DICT:
                PSV_i[d, :, :, :] = result
            else:
                PSV[counter_img, d, :, :, :] = result

        if USE_DICT:
            PSV[i] = PSV_i
        counter_img += 1

    return PSV



def transform_cost_volume_cuda(list_src_cams, list_src_DPs,
                              dest_cam,
                              num_depths, height, width,
                              min_disp, disp_step, max_depth, min_depth,
                              do_normalization=True):


    int_dest = dest_cam['intrinsic']
    ext_dest = dest_cam['extrinsic']
    inv_int_dest = np.linalg.inv(int_dest)
    torch_inv_int_dest = torch.from_numpy(inv_int_dest)
    torch_inv_int_dest = torch_inv_int_dest.cuda()
    inv_ext_dest = np.linalg.inv(ext_dest)
    torch_inv_ext_dest = torch.from_numpy(inv_ext_dest)
    torch_inv_ext_dest = torch_inv_ext_dest.cuda()

    list_warped_prob = []
    sum_warped_prob = np.zeros(shape=(num_depths, height, width))
    view_counter = np.zeros(shape=(num_depths, height, width))

    # define the voxel grid
    Z, Y, X = np.meshgrid(np.arange(0, num_depths), np.arange(0, height), np.arange(0, width), indexing='ij')
    Z = Z * disp_step
    zero_mask = (Z == 0)
    Z[Z != 0] += min_disp
    Z[Z != 0] = 1.0 / Z[Z != 0]
    # Z[Z != 0] += min_disp
    Z[zero_mask] = max_depth
    X = X * Z
    Y = Y * Z
    points = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    points = np.transpose(points)

    torch_points = torch.from_numpy(points)
    torch_points = torch_points.cuda()


    for src_idx, cam_param_src in enumerate(list_src_cams):
        import time
        start = time.time()
        start_multiply = time.time()
        print('Transforming the cost volume of %02d' % (src_idx))
        src_prob = list_src_DPs[src_idx]

        # get the parameters
        ext_i = cam_param_src['extrinsic']
        torch_ext_i = torch.from_numpy(ext_i)
        torch_ext_i = torch_ext_i.cuda()
        int_i = cam_param_src['intrinsic']
        torch_int_i = torch.from_numpy(int_i)
        torch_int_i = torch_int_i.cuda()

        warped_prob = np.zeros(shape=(num_depths, height, width))

        transformed_points = torch.matmul(torch_points, torch_inv_int_dest.t())
        transformed_points = torch.matmul(transformed_points, torch_inv_ext_dest[0:3, 0:3].t())
        transformed_points = torch.add(transformed_points, torch_inv_ext_dest[0:3, 3])
        transformed_points = torch.matmul(transformed_points, torch_ext_i[0:3, 0:3].t())
        transformed_points = torch.add(transformed_points, torch_ext_i[0:3, 3])
        # transformed_points = transformed_points[:, 0:3]
        transformed_points = torch.matmul(transformed_points, torch_int_i.t())
        X_src = transformed_points[:, 0] / transformed_points[:, 2]
        Y_src = transformed_points[:, 1] / transformed_points[:, 2]
        Z_src = transformed_points[:, 2]

        X_src = X_src.cpu().numpy()
        Y_src = Y_src.cpu().numpy()
        Z_src = Z_src.cpu().numpy()


        end_multipy = time.time()
        print('\t- Multiplication Iteration Took: ' + str(end_multipy - start_multiply))

        start_round = time.time()

        X_src = X_src.reshape((num_depths, height, width))
        Y_src = Y_src.reshape((num_depths, height, width))
        Z_src = Z_src.reshape((num_depths, height, width))
        disp_src = 1.0 / Z_src - min_disp

        round_Y_src = np.round(Y_src).astype(np.int)
        round_X_src = np.round(X_src).astype(np.int)
        round_Z_src = np.round(disp_src / disp_step).astype(np.int)
        round_Z_src[Z_src >= max_depth] = 0
        round_Z_src[Z_src <= min_depth] = num_depths - 1

        valid_index = np.bitwise_and(round_Y_src >= 0, round_Y_src < height)
        valid_index = np.bitwise_and(valid_index, round_X_src >= 0)
        valid_index = np.bitwise_and(valid_index, round_X_src < width)
        valid_index = np.bitwise_and(valid_index, round_Z_src >= 0)
        valid_index = np.bitwise_and(valid_index, round_Z_src < num_depths)

        end_round = time.time()
        print('\t- Round Iteration Took: ' + str(end_round - start_round))

        start_warp = time.time()
        warped_prob[valid_index] = src_prob[round_Z_src[valid_index],
                                            round_Y_src[valid_index],
                                            round_X_src[valid_index]]
        end_warp = time.time()

        print('\t- Warp Iteration Took: ' + str(end_warp - start_warp))

        view_counter[valid_index] += 1.0
        list_warped_prob.append(warped_prob)
        sum_warped_prob += warped_prob

        end = time.time()
        print('\t\t- One Iteration Took: ' + str(end - start))

        # save to a file
        # np.save('%s/warped_prob_from_%02d.npy' % (out_dir, prob_src_id), warped_prob)

        # # save the depth probability
        # for i in range(num_depths):
        #     prob = warped_prob[i]
        #     prob_color = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     cv2.imshow("prob", prob_color)
        #     cv2.waitKey()

    if do_normalization:
        sum_warped_prob = np.multiply(sum_warped_prob, 1.0/(view_counter + 1e-10))
        dp_sq_sum = np.sqrt(np.sum(np.multiply(sum_warped_prob, sum_warped_prob), axis=0)) + 1e-10
        sum_warped_prob = sum_warped_prob / dp_sq_sum

        # ## remove some inconfident ray
        # confident_ray = np.sum(sum_warped_prob, axis=0) >= 0.75
        # confident_ray = np.expand_dims(confident_ray, axis=0)
        # confident_ray = np.tile(confident_ray, [num_depths, 1, 1])
        # # valid_index = np.bitwise_and(valid_index, confident_ray)
        # warped_prob[np.logical_not(confident_ray)] = 0.0
        # ##



    return sum_warped_prob, list_warped_prob


def synthesize_a_view(cam_dest, PSV, depth_prob,
                     list_img, list_cam_params, list_depth_prob,
                     min_disp, disp_step, depth_at_infinity,
                     height, width, num_depths,
                     list_validity_maps=[], with_ULR_weight=False,
                     color_max_depth=None,
                     winner_takes_all=False):
    my_comparator = depth_util.my_comparator_greater

    int_dest = cam_dest['intrinsic']
    inv_int_dest = np.linalg.inv(int_dest)
    ext_dest = cam_dest['extrinsic']
    inv_ext_dest = np.linalg.inv(cam_dest['extrinsic'])
    campos_dest = inv_ext_dest[0:3, 3]
    camdir_dest = ext_dest[2, 0:3]

    # check zero_prob_idx
    # sum_prob = np.sqrt(np.sum(depth_prob*depth_prob, axis=0))
    # nonzero_prob_idx = sum_prob > 0.35
    # nonzero_prob_idx = np.expand_dims(nonzero_prob_idx, -1)
    # nonzero_prob_idx = np.tile(nonzero_prob_idx, [1, 1, 3])
    abs_max = np.max(depth_prob, axis=0)
    valid_prob_1d = abs_max >= 0.10
    valid_prob = np.expand_dims(valid_prob_1d, -1)
    valid_prob = np.tile(valid_prob, [1, 1, 3])
    if with_ULR_weight:
        ULR_weight_sum = 0.0

    avg_new_view = np.zeros((height, width, 3), dtype=np.float)
    normalizer = np.zeros((height, width, 3), dtype=np.float)
    visibility_map = np.zeros((height, width), dtype=np.float)

    # for each PSV_k, perfrom view synthesis
    dict_new_views = dict()
    list_new_views = []

    # compute the weight
    import time
    start_time = time.time()
    # weight_volume = np.zeros((num_depths, height, width, 3), dtype=np.float)
    # for j in range(height):
    #     for i in range(width):
    #         data = depth_prob[:, j, i]
    #         idx_local_max = argrelextrema(data, my_comparator, order=5)
    #         max_index = np.argmax(data)
    #         global_max_value = data[max_index]
    #
    #         if len(idx_local_max) == 0:
    #             depth_idx = max_index
    #         else:
    #             idx_local_max = idx_local_max[0]
    #             if len(idx_local_max) == 0 or len(idx_local_max) == 1:
    #                 depth_idx = max_index
    #             else:
    #                 for idx in reversed(idx_local_max):
    #                     local_max_value = data[idx]
    #                     if local_max_value >= global_max_value * VSynth.DT_THRESHOLD:
    #                         depth_idx = idx
    #                         break
    #
    #         weight_volume[depth_idx, j, i, :] = 1.0

    weight_volume = np.zeros((num_depths, height, width, 3), dtype=np.float)
    weight_volume_2 = np.zeros((num_depths, height, width, 3), dtype=np.float)
    obj_local_max = argrelextrema(depth_prob, my_comparator, order=3, mode='wrap')
    obj_global_max = np.argmax(depth_prob, axis=0)

    first_closest_peak = np.ones(shape=(height, width))*(-1)
    second_closest_peak = np.ones(shape=(height, width))*(-1)

    for i in reversed(range(len(obj_local_max[0]))):
        idx_d = obj_local_max[0][i]
        idx_y = obj_local_max[1][i]
        idx_x = obj_local_max[2][i]

        local_max_value = depth_prob[idx_d, idx_y, idx_x]
        global_max_value = depth_prob[obj_global_max[idx_y, idx_x], idx_y, idx_x]

        if local_max_value >= global_max_value * VSynth.DT_THRESHOLD:
            fp_exists = first_closest_peak[idx_y, idx_x] != -1
            sp_exists = second_closest_peak[idx_y, idx_x] != -1
            if sp_exists:
                continue
            if not sp_exists and not fp_exists:
                first_closest_peak[idx_y, idx_x] = idx_d
                weight_volume[idx_d, idx_y, idx_x] = 1.0
            elif fp_exists and not sp_exists:
                second_closest_peak[idx_y, idx_x] = idx_d
                weight_volume_2[idx_d, idx_y, idx_x] = 1.0

    fp_empty = first_closest_peak == -1
    sp_empty = second_closest_peak == -1
    j_grid, i_grid = np.meshgrid(range(0, height), range(0, width), indexing='ij')
    weight_volume[obj_global_max[fp_empty], j_grid[fp_empty], i_grid[fp_empty]] = 1.0
    weight_volume_2[obj_global_max[sp_empty], j_grid[sp_empty], i_grid[sp_empty]] = 1.0

    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # compute the depth map
    ref_depth_map, ref_depth_map_colored, zero_disp \
        = depth_util.generate_depthmap(weight_volume[:,:,:,0],
                                       min_disp, disp_step, depth_at_infinity, color_max_val=color_max_depth)
    ref_depth_map_2, ref_depth_map_colored_2, _\
        = depth_util.generate_depthmap(weight_volume_2[:,:,:,0],
                                       min_disp, disp_step, depth_at_infinity, color_max_val=color_max_depth)

    # filter the depthmap
    invalid_prob_1d = np.logical_not(valid_prob_1d)
    invalid_prob = np.logical_not(valid_prob)
    ref_depth_map[invalid_prob_1d] = 0.0
    ref_depth_map_colored[invalid_prob] = 0.0
    ref_depth_map_2[invalid_prob_1d] = 0.0
    ref_depth_map_colored_2[invalid_prob] = 0.0


    # imageio.imwrite('./ref_depth_map_colored.png', ref_depth_map_colored)
    # imageio.imwrite('./ref_depth_map_colored_2.png', ref_depth_map_colored_2)

    # compute weights
    if with_ULR_weight:

        # compute max distance and min distance
        max_dist = -9999999.0
        min_dist = 9999999.0
        min_idx = -1
        for src_idx in range(len(list_img)):
            # get the parameters ready
            cam_param_src = list_cam_params[src_idx]
            ext_i = cam_param_src['extrinsic']

            # positional weight
            inv_ext_i = np.linalg.inv(ext_i)
            campos_i = inv_ext_i[0:3, 3]
            campos_diff = campos_i - campos_dest
            campos_dist = np.sqrt(np.sum(campos_diff * campos_diff))
            if campos_dist > max_dist:
                max_dist = campos_dist
            if campos_dist < min_dist:
                min_dist = campos_dist
                min_idx = src_idx

        # compute weights
        list_ULR_weights = []
        for src_idx in range(len(list_img)):
            # get the parameters ready
            cam_param_src = list_cam_params[src_idx]
            ext_i = cam_param_src['extrinsic']

            # positional weight
            inv_ext_i = np.linalg.inv(ext_i)
            campos_i = inv_ext_i[0:3, 3]
            campos_diff = campos_i - campos_dest
            campos_dist = np.sqrt(np.sum(campos_diff * campos_diff))/max_dist
            # campos_weight = np.exp(-campos_dist / (0.40 * 0.40))
            campos_weight = 100*np.exp(-campos_dist / (0.2 * 0.2))



            # directional weight
            camdir_i = ext_i[2, 0:3]
            camdir_dot = camdir_i * camdir_dest
            camdir_dist = np.sqrt(np.sum(camdir_dot * camdir_dot))
            if camdir_dist < 0.5:
                camdir_weight = 0.0
            else:
                # camdir_weight = np.exp(-camdir_dist / (0.8 * 0.8))
                camdir_weight = 100*np.exp(-camdir_dist / (0.4 * 0.4))

            if winner_takes_all:
                if src_idx == min_idx:
                    campos_weight = 1.0
                    camdir_weight = 1.0
                else:
                    campos_weight = 0.1
                    camdir_weight = 0.1

            ULR_weight = campos_weight * camdir_weight
            print("%f / %f / %f"%(campos_weight, camdir_weight, ULR_weight))
            list_ULR_weights.append(ULR_weight)

    for src_idx in range(len(list_img)):
        # compute the depth map
        depth_map_k, _, _, _ \
            = depth_util.generate_depthmap(list_depth_prob[src_idx], min_disp, disp_step, depth_at_infinity)

        # get the parameters ready
        cam_param_src = list_cam_params[src_idx]
        # get the parameters
        ext_i = cam_param_src['extrinsic']
        int_i = cam_param_src['intrinsic']

        # reproject to
        Y, X = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')
        Z = ref_depth_map
        X = X * Z
        Y = Y * Z
        points = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
        points = np.matmul(inv_int_dest, points)
        points = np.vstack([points, np.ones((1, height * width))])
        points = np.matmul(inv_ext_dest, points)
        points = np.matmul(ext_i, points)
        points = points[0:3]
        points = np.matmul(int_i, points)
        X_src = points[0] / points[2]
        Y_src = points[1] / points[2]
        Z_src = points[2]

        X_src = X_src.reshape((height, width))
        Y_src = Y_src.reshape((height, width))
        Z_src = Z_src.reshape((height, width))
        # make an exception for sky
        Z_src[zero_disp] = depth_at_infinity
        Z_src[Z_src > depth_at_infinity] = depth_at_infinity

        round_Y_src = np.round(Y_src).astype(np.int)
        round_X_src = np.round(X_src).astype(np.int)
        valid_index = np.bitwise_and(round_Y_src >= 0, round_Y_src < height)
        valid_index = np.bitwise_and(valid_index, round_X_src >= 0)
        valid_index = np.bitwise_and(valid_index, round_X_src < width)

        # warped_depth_map_k = np.zeros(shape=(height, width))
        # warped_depth_map_k[valid_index] = depth_map_k[round_Y_src[valid_index],
        #                                               round_X_src[valid_index]]
        #
        # depth_diff = ref_depth_map - warped_depth_map_k
        # invalid_depth = depth_diff > ref_depth_map*VISIBILITY_TEST_THRESHOLD
        # invalid_depth = depth_diff > 0
        depth_diff = np.zeros(shape=(height, width))
        depth_diff[valid_index] = Z_src[valid_index]\
                                    - depth_map_k[round_Y_src[valid_index],
                                                  round_X_src[valid_index]]
        invalid_depth = depth_diff > Z_src*VSynth.VISIBILITY_TEST_THRESHOLD
        valid_depth = np.logical_not(invalid_depth)
        valid_index = np.logical_and(valid_index, valid_depth)

        # get PSV
        PSV_k = PSV[src_idx]

        if list_validity_maps != []:
            validity_map = list_validity_maps[src_idx]
            check_validity = np.zeros(shape=(height, width))
            check_validity[valid_index] = validity_map[round_Y_src[valid_index],
                                                       round_X_src[valid_index]]
            valid_index = np.logical_and(valid_index, check_validity)

        # perform
        valid_index = valid_index.astype(np.float)
        valid_index = np.expand_dims(valid_index, -1)
        valid_index = np.tile(valid_index, [1, 1, 3])
        # valid_index = np.logical_and(valid_index, nonzero_prob_idx)
        valid_index = np.logical_and(valid_index, valid_prob)


        view_k = np.multiply(PSV_k, weight_volume)
        view_k = np.sum(view_k, 0)
        view_k = view_k * valid_index
        # import imageio
        # imageio.imwrite('./view_%04d.png' % src_idx, view_k)
        # imageio.imwrite('./view_%04d_mask.png' % src_idx, valid_index.astype(np.float))

        if with_ULR_weight:
            # apply the weight
            ULR_weight = list_ULR_weights[src_idx]
            ULR_weight_sum += ULR_weight

            avg_new_view += view_k*ULR_weight
            normalizer += valid_index*ULR_weight

        else:
            avg_new_view += view_k
            normalizer += valid_index

        visibility_map += valid_index.astype(np.float)[:,:,0]
        dict_new_views[src_idx] = view_k
        list_new_views.append(view_k)


        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(img_k)
        # plt.title("Source Image")
        # plt.subplot(122)
        # plt.imshow(view_k)
        # plt.title("View Synth [%02d]" % (k))

    zero_pixels = avg_new_view == 0.0
    avg_new_view = np.multiply(avg_new_view, 1.0 / (normalizer + 1e-10))
    avg_new_view[zero_pixels] = 0.0


    # remove area where only one view sees
    # one_view_map = (visibility_map == 1.0)
    # one_view_map = np.expand_dims(one_view_map, -1)
    # one_view_map = np.tile(one_view_map, (1, 1, 3))
    # avg_new_view[one_view_map] = 0.0

    validity_map = visibility_map > 1.0
    visibility_map /= float(len(list_img))

    # remove outliers
    avg_new_view[avg_new_view > 1.0] = 1.0
    avg_new_view[avg_new_view < 0.0] = 0.0

    # imageio.imwrite('./view_merged.png', avg_new_view)
    # fig = plt.figure()
    # plt.imshow(visibility_map)
    # plt.show()
    if list_validity_maps != []:
        return avg_new_view, list_new_views, visibility_map, validity_map, \
               ref_depth_map, ref_depth_map_colored, \
               ref_depth_map_2, ref_depth_map_colored_2
    else:
        return avg_new_view, list_new_views, visibility_map,\
               ref_depth_map, ref_depth_map_colored,\
               ref_depth_map_2, ref_depth_map_colored_2


