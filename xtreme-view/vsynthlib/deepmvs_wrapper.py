"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/NVlabs/extreme-view-synth.
Authors: Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H. Kim, and Jan Kautz
"""

import torch
import torchvision as vision
import torch.nn.functional as F
from torch.autograd import Variable
import pydensecrf.densecrf as dcrf
import numpy as np
import cv2

from DeepMVS.model import DeepMVS




class DeepMVSWrapper(object):
    def __init__(self, filename_DeepMVS,
                 n_depths=100,
                 enable_CUDA=True,
                 do_filter=True):

        self.dev_id = 0
        if torch.cuda.device_count() > 1:
            self.dev_id = 1

        self.model_deepMVS = DeepMVS(n_depths, use_gpu=enable_CUDA, gpu_id=self.dev_id)
        self.model_deepMVS.load_state_dict(torch.load(filename_DeepMVS))
        self.model_deepMVS.share_memory()
        print('DeepMVS model loaded!', filename_DeepMVS)

        if enable_CUDA:
            self.model_VGGNet = vision.models.vgg19(pretrained=True).cuda(self.dev_id)
        else:
            self.model_VGGNet = vision.models.vgg19(pretrained=True)

        self.model_VGGNet.share_memory()
        self.model_VGGNet_normalize\
            = vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print('VGGNET model loaded!')

        # Constants for DenseCRF.
        self.dict_DenseCRF = dict()

        ######################################
        # default from DeepMVS
        ######################################
        # self.dict_DenseCRF['sigma_xy'] = 80.0
        # self.dict_DenseCRF['sigma_rgb'] = 15.0
        # self.dict_DenseCRF['sigma_d'] = 10.0
        # self.dict_DenseCRF['iteration_num'] = 5
        # compat = np.zeros((n_depths, n_depths), dtype=np.float32)
        # for row in range(0, n_depths):
        #     for col in range(0, n_depths):
        #         compat[row, col] = (row - col) ** 2 / self.dict_DenseCRF['sigma_d'] ** 2 / 2
        # self.dict_DenseCRF['compat'] = compat
        #####################################

        ######################################
        # For museum and others
        ######################################
        self.dict_DenseCRF['sigma_xy'] = 30.0
        self.dict_DenseCRF['sigma_rgb'] = 3
        self.dict_DenseCRF['iteration_num'] = 20
        self.dict_DenseCRF['compat'] = 10.0

        # for high res
        # self.dict_DenseCRF['sigma_xy'] = 60
        # self.dict_DenseCRF['sigma_rgb'] = 3.0
        # self.dict_DenseCRF['iteration_num'] = 20
        # self.dict_DenseCRF['compat'] = 10.0

        ######################################
        # For bikes of StereoMagnificiation
        ######################################
        # self.dict_DenseCRF['sigma_xy'] = 25.0
        # self.dict_DenseCRF['sigma_rgb'] = 10.0
        # self.dict_DenseCRF['iteration_num'] = 5
        # self.dict_DenseCRF['compat'] = 5.0

        self.n_depths = n_depths
        self.patch_size = 128
        self.stride = int(self.patch_size/2)
        self.do_filter = do_filter

    def build_PSV(self, list_src_img, list_src_cam, ref_idx,
                  height, width,
                  min_disp, disp_step, max_depth):

        n_neighbors = len(list_src_img) - 1

        PSV = np.zeros(shape=[n_neighbors, self.n_depths, height, width, 3], dtype=np.float32)

        cam_param_ref = list_src_cam[ref_idx]
        int_mat_ref = cam_param_ref['intrinsic']
        fx_ref = int_mat_ref[0, 0]
        fy_ref = int_mat_ref[1, 1]
        cx_ref = int_mat_ref[0, 2]
        cy_ref = int_mat_ref[1, 2]
        ext_ref = cam_param_ref['extrinsic']
        inv_ext_ref = np.linalg.inv(ext_ref)

        # for each neighbor image
        counter_img = 0
        for i in range(len(list_src_img)):
            if i == ref_idx:
                continue

            img_i = list_src_img[i]
            cam_param_i = list_src_cam[i]
            # get the parameters
            int_mat = cam_param_i['intrinsic']
            fx_i = int_mat[0, 0]
            fy_i = int_mat[1, 1]
            cx_i = int_mat[0, 2]
            cy_i = int_mat[1, 2]
            ext_i = cam_param_i['extrinsic']

            # 4 Corners on the virtual camera to get te 4 rays that intersect with the depth plane
            src_pts = np.reshape([0, 0,
                                  width, 0,
                                  width, height,
                                  0, height], (4, 2))

            # for each depth plane
            for d in range(self.n_depths):

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
                    p_3D_ref = np.asarray([(depth * p[0] - depth * cx_ref) / fx_ref,
                                           (depth * p[1] - depth * cy_ref) / fy_ref,
                                           depth])
                    p_4D_ref = np.array([p_3D_ref[0], p_3D_ref[1], p_3D_ref[2], 1.0])
                    p_4D_world = inv_ext_ref.dot(p_4D_ref)
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

                PSV[counter_img, d, :, :, :] = result

            counter_img += 1

        return PSV

    def perform_DeepMVS(self, list_img, ref_idx, PSV,
                        height, width, batch_size=1, use_gpu=True):

        # Generate VGG features.
        with torch.no_grad():
            VGG_tensor = Variable(
                self.model_VGGNet_normalize(torch.FloatTensor(list_img[ref_idx].copy())).permute(2, 0, 1).unsqueeze(0))

        if use_gpu:
            VGG_tensor = VGG_tensor.cuda(self.dev_id)
        VGG_scaling_factor = 0.01
        for i in range(0, 4):
            VGG_tensor = self.model_VGGNet.features[i].forward(VGG_tensor)
        if use_gpu:
            feature_input_1x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
        else:
            feature_input_1x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
        for i in range(4, 9):
            VGG_tensor = self.model_VGGNet.features[i].forward(VGG_tensor)
        if use_gpu:
            feature_input_2x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
        else:
            feature_input_2x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
        for i in range(9, 14):
            VGG_tensor = self.model_VGGNet.features[i].forward(VGG_tensor)
        if use_gpu:
            feature_input_4x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
        else:
            feature_input_4x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
        for i in range(14, 23):
            VGG_tensor = self.model_VGGNet.features[i].forward(VGG_tensor)
        if use_gpu:
            feature_input_8x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
        else:
            feature_input_8x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
        for i in range(23, 32):
            VGG_tensor = self.model_VGGNet.features[i].forward(VGG_tensor)
        if use_gpu:
            feature_input_16x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
        else:
            feature_input_16x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
        del VGG_tensor

        # Stride through entire reference image.
        predict_raw = torch.zeros(self.n_depths, height, width)
        border_x = (self.patch_size - self.stride) / 2
        border_y = (self.patch_size - self.stride) / 2
        col_total = int((width - 2 * border_x - 1) / self.stride) + 1
        row_total = int((height - 2 * border_y - 1) / self.stride) + 1

        for row_idx in range(0, row_total):
            for col_idx in range(0, col_total):

                # Compute patch location for this patch and next patch.
                if col_idx != col_total - 1:
                    start_x = col_idx * self.stride
                else:
                    start_x = width - self.patch_size

                if row_idx != row_total - 1:
                    start_y = row_idx * self.stride
                else:
                    start_y = height - self.patch_size

                # Read plane-sweep volume and start next patch.
                ref_img = list_img[ref_idx][start_y:(start_y + self.patch_size), start_x:(start_x + self.patch_size),
                          :].copy() - 0.5
                sweep_volume = PSV[:, :, start_y:(start_y + self.patch_size), start_x:(start_x + self.patch_size),
                               :].copy() - 0.5
                num_neighbors = len(list_img) - 1

                # Prepare the inputs.
                data_in_tensor = torch.FloatTensor(batch_size, 1, self.n_depths, 2, 3, self.patch_size, self.patch_size)
                ref_img_tensor = torch.FloatTensor(ref_img).permute(2, 0, 1).unsqueeze(0)
                data_in_tensor[0, 0, :, 0, ...] = ref_img_tensor.expand(self.n_depths, -1, -1, -1)
                with torch.no_grad():
                    feature_input_1x \
                        = Variable(
                        feature_input_1x_whole[..., start_y:start_y + self.patch_size, start_x:start_x + self.patch_size])
                    feature_input_2x \
                        = Variable(
                        feature_input_2x_whole[..., int(start_y / 2):int(start_y / 2) + int(self.patch_size / 2),
                        int(start_x / 2):int(start_x / 2) + int(self.patch_size / 2)])
                    feature_input_4x \
                        = Variable(
                        feature_input_4x_whole[..., int(start_y / 4):int(start_y / 4) + int(self.patch_size / 4),
                        int(start_x / 4):int(start_x / 4) + int(self.patch_size / 4)])
                    feature_input_8x \
                        = Variable(
                        feature_input_8x_whole[..., int(start_y / 8):int(start_y / 8) + int(self.patch_size / 8),
                        int(start_x / 8):int(start_x / 8) + int(self.patch_size / 8)])
                    feature_input_16x \
                        = Variable(
                        feature_input_16x_whole[..., int(start_y / 16):int(start_y / 16) + int(self.patch_size / 16),
                        int(start_x / 16):int(start_x / 16) + int(self.patch_size / 16)])
                if use_gpu:
                    feature_input_1x = feature_input_1x.cuda(self.dev_id)
                    feature_input_2x = feature_input_2x.cuda(self.dev_id)
                    feature_input_4x = feature_input_4x.cuda(self.dev_id)
                    feature_input_8x = feature_input_8x.cuda(self.dev_id)
                    feature_input_16x = feature_input_16x.cuda(self.dev_id)
                # Loop through all neighbor images.
                for neighbor_idx in range(0, num_neighbors):
                    data_in_tensor[0, 0, :, 1, ...] = torch.FloatTensor(
                        np.moveaxis(sweep_volume[neighbor_idx, ...], -1, -3))
                    with torch.no_grad():
                        data_in = Variable(data_in_tensor)
                    if use_gpu:
                        data_in = data_in.cuda(self.dev_id)
                    if neighbor_idx == 0:
                        cost_volume \
                            = self.model_deepMVS.forward_feature(data_in, [feature_input_1x, feature_input_2x, feature_input_4x,
                                                              feature_input_8x, feature_input_16x]).data[...]
                    else:
                        cost_volume \
                            = torch.max(cost_volume, self.model_deepMVS.forward_feature(data_in, [feature_input_1x, feature_input_2x,
                                                                                     feature_input_4x, feature_input_8x,
                                                                                     feature_input_16x]).data[...])
                # Make final prediction.
                with torch.no_grad():
                    predict = self.model_deepMVS.forward_predict(Variable(cost_volume[:, 0, ...]))

                # Compute copy range.
                if col_idx == 0:
                    copy_x_start = 0
                    copy_x_end = self.patch_size - border_x
                elif col_idx == col_total - 1:
                    copy_x_start = border_x + col_idx * self.stride
                    copy_x_end = width
                else:
                    copy_x_start = border_x + col_idx * self.stride
                    copy_x_end = copy_x_start + self.stride

                if row_idx == 0:
                    copy_y_start = 0
                    copy_y_end = self.patch_size - border_y
                elif row_idx == row_total - 1:
                    copy_y_start = border_y + row_idx * self.stride
                    copy_y_end = height
                else:
                    copy_y_start = border_y + row_idx * self.stride
                    copy_y_end = copy_y_start + self.stride

                # Copy the prediction to buffer.
                copy_x_start = int(copy_x_start)
                copy_x_end = int(copy_x_end)
                copy_y_start = int(copy_y_start)
                copy_y_end = int(copy_y_end)
                predict_raw[..., copy_y_start:copy_y_end, copy_x_start:copy_x_end] \
                    = predict.data[0, :, copy_y_start - start_y:copy_y_end - start_y,
                      copy_x_start - start_x:copy_x_end - start_x]

        ######################################################
        # compute the depth probability
        ######################################################
        with torch.no_grad():
            depth_prob = F.softmax(Variable(predict_raw), dim=0).data.numpy()

        ######################################################
        # Pass through DenseCRF.
        ######################################################
        with torch.no_grad():
            unary_energy = F.log_softmax(Variable(predict_raw), dim=0).data.numpy()

        crf = dcrf.DenseCRF2D(width, height, self.n_depths)
        crf.setUnaryEnergy(-unary_energy.reshape(self.n_depths, height * width))
        ref_img_full = (list_img[ref_idx] * 255.0).astype(np.uint8)
        crf.addPairwiseBilateral(sxy=(self.dict_DenseCRF['sigma_xy'], self.dict_DenseCRF['sigma_xy']),
                                 srgb=(
                                 self.dict_DenseCRF['sigma_rgb'], self.dict_DenseCRF['sigma_rgb'], self.dict_DenseCRF['sigma_rgb']),
                                 rgbim=ref_img_full,
                                 compat=self.dict_DenseCRF['compat'],
                                 kernel=dcrf.FULL_KERNEL,
                                 normalization=dcrf.NORMALIZE_SYMMETRIC)
        new_raw = crf.inference(self.dict_DenseCRF['iteration_num'])
        new_raw = np.array(new_raw).reshape(self.n_depths, height, width)

        return new_raw, depth_prob

    def compute(self, list_src_img, list_src_cam, ref_idx,
                       min_disp, disp_step, max_depth):

        img = list_src_img[0]
        height, width, n_channels = img.shape

        # build PSV
        PSVs = self.build_PSV(list_src_img, list_src_cam, ref_idx,
                              height, width,
                              min_disp, disp_step, max_depth)

        # call deepMVS
        dp_refined, dp = self.perform_DeepMVS(list_src_img, ref_idx, PSVs,
                                               height, width)

        if self.do_filter:
            return dp_refined
        else:
            return dp



