"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/NVlabs/extreme-view-synth.
Authors: Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H. Kim, and Jan Kautz
"""

import numpy as np
import torch
from torch.autograd import Variable
import os
import time
import imageio

from vsynthlib.refinet.models import Model_VNPCAT
from vsynthlib import depth_util


class DeepViewRefiner(object):

    NUM_INPUT_CHANNELS = 27

    def __init__(self, filename_model_weight, working_dir, out_dir,
                 patch_size=64,
                 with_CUDA=True):

        # define the model
        self.model = Model_VNPCAT()

        # load weight
        self.with_CUDA = with_CUDA
        self.model.load_state_dict(torch.load(filename_model_weight))
        if with_CUDA:
            self.model.cuda()

        self.working_dir = working_dir
        self.out_dir = out_dir
        self.patch_size = patch_size

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)



    pass

    def do(self, synth_obj, list_src_img, list_src_cam, count=0,
           do_stereo=False, return_val=False, custom_outdir=''):

        value = self.do_VNPCAT(synth_obj, list_src_img, list_src_cam,
                        count=count, return_val=return_val,
                        without_candi= False,
                        custom_outdir=custom_outdir)

        if return_val:
            return value

    def do_VNPCAT(self, synth_obj, list_src_img, list_src_cam, count=0,
                  return_val=False, without_candi=False, custom_outdir=''):

        func_fetch_patch = depth_util.fetch_patches_VNP

        # load synth data
        img_synth = synth_obj['img_synth']
        depth_map_P1 = synth_obj['depth_map_P1']
        depth_map_P2 = synth_obj['depth_map_P2']
        dest_cam = synth_obj['dest_cam']
        height, width, _ = img_synth.shape

        # perform refinement patch-by-patchy
        #############################################################
        # Do Testing
        #############################################################
        img_merged = np.zeros(shape=(height, width, 3))
        img_counter = np.zeros(shape=(height, width, 3))
        for j in range(0, height, int(self.patch_size / 4)):
            for i in range(0, width, int(self.patch_size / 4)):

                t_start = time.time()
                # set the model to the evaluation mode
                self.model.eval()

                # get candidate tensor
                x_top = i
                y_top = j
                if x_top + self.patch_size >= width:
                    x_top = width - self.patch_size
                if y_top + self.patch_size >= height:
                    y_top = height - self.patch_size

                t_input_synth, list_t_candi_patch = func_fetch_patch(y_top, x_top, self.patch_size, dest_cam,
                                                                    img_synth, list_src_img, list_src_cam,
                                                                    depth_map_P1, depth_map_P2)

                if t_input_synth is None:
                    print('None!')
                    continue

                # check if more than half of input pixels are valid
                t_in_slice = t_input_synth[0]
                bool_nz = t_in_slice != -0.5
                bool_nz = bool_nz.astype(np.float)
                sum_nz = np.sum(bool_nz)
                if sum_nz < self.patch_size * self.patch_size * 0.6:
                    continue

                t_input_synth = np.expand_dims(t_input_synth, axis=0)
                t_input_synth = t_input_synth.astype(np.float32)
                _, chs, _, _ = t_input_synth.shape
                n_patches = len(list_t_candi_patch)
                t_in_synth = t_input_synth

                input_synth_tensor \
                    = torch.from_numpy(t_in_synth)

                if self.with_CUDA:
                    input_synth_tensor = input_synth_tensor.cuda()
                with torch.no_grad():
                    input_synth_variable = Variable(input_synth_tensor, requires_grad=False)

                list_input_candi_variable = []
                for i in range(n_patches):
                    candi_patch = list_t_candi_patch[i]
                    candi_patch = np.expand_dims(candi_patch, axis=0)
                    candi_patch = candi_patch.astype(np.float32)

                    candi_tensor = torch.from_numpy(candi_patch)

                    if self.with_CUDA:
                        candi_tensor = candi_tensor.cuda()

                    with torch.no_grad():
                        input_candi_variable = Variable(candi_tensor)

                    list_input_candi_variable.append(input_candi_variable)


                # do forward pass
                if without_candi:
                    output_variable = self.model(input_synth_variable)
                    output_to_show = output_variable[0].cpu().data[0]
                else:
                    output_variable = self.model(input_synth_variable, list_input_candi_variable)
                    output_to_show = output_variable.cpu().data[0]

                output_to_show = output_to_show + 0.5
                output_to_show = output_to_show.permute(1, 2, 0).numpy()
                output_to_show[output_to_show < 0.0] = 0.0
                output_to_show[output_to_show > 1.0] = 1.0
                output_to_show = output_to_show * 255.0
                output_to_show = output_to_show.astype(np.uint8)

                img_merged[y_top:(y_top + self.patch_size), x_top:(x_top + self.patch_size), :] += output_to_show
                img_counter[y_top:(y_top + self.patch_size), x_top:(x_top + self.patch_size), :] += 1

                t_current = time.time()
                t_elapsed_row = t_current - t_start

                # delete variables
                del input_synth_variable
                for var in list_input_candi_variable:
                    self.var = var
                    del self.var


        img_merged = img_merged / (img_counter + 1e-10)
        img_merged /= 255.0
        if return_val:
            return img_merged[0:height, 0:width]
        else:
            filename_out_prefix = 'refined_vsynth_%04d' % (count)
            if custom_outdir != '':
                imageio.imwrite('%s/%s.png' % (custom_outdir, filename_out_prefix), img_merged[0:height, 0:width])
            else:
                imageio.imwrite('%s/%s.png' % (self.out_dir, filename_out_prefix), img_merged[0:height, 0:width])
