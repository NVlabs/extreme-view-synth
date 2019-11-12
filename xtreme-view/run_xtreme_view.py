"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/NVlabs/extreme-view-synth.
Authors: Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H. Kim, and Jan Kautz
"""

from copy import deepcopy
from vsynthlib import deepmvs_wrapper
from vsynthlib import core
from dataloader import colmap_loader

import os
import sys
import argparse
import numpy as np

class XtremeViewRunner():

    virtual_cams = {}

    # scene 0
    virtual_cams['0000'] = { 'src_indx' : 0,
                    'view_offsets' : [np.array([-6, 0, 0, 0]),
                                    np.array([-4.5,0,  0, 0]),
                                    np.array([-3,0,  0, 0]),
                                    np.array([ 3, 0, 0, 0]),
                                    np.array([ 4.5,0,  0, 0]),
                                    np.array([ 6, 0, 0, 0])]
                }

    # Figure 10 - top row
    virtual_cams['0005'] = { 'src_indx': 0,
                    'view_offsets':  [np.array([6, 0, -0.5, 0]),
                                      np.array([-6, 0, -0.5, 0])]
                }

    # Figure 10 - second row
    virtual_cams['0009'] = { 'src_indx': 1,
                    'view_offsets': [np.array([-2, 0, -0.5, 0]), # absent
                                    np.array([-5,0, -0.5, 0]), # absent
                                    np.array([-8, 0, -0.5, 0]), # absent
                                    np.array([-11, 0, -0.5, 0])]
                }

    # Figure 10 - third row
    virtual_cams['0020'] = {'src_indx': 1,
                    'view_offsets' :[
                    np.array([0, 0, -3.0, 0]),
                    np.array([0, 0, -4.0, 0]),
                    np.array([0, 0, -8.0, 0]),
                    ]
                }

    # Figure 10 - fourth row
    virtual_cams['0027'] = { 'src_indx': 1,
                    'view_offsets': [np.array([0.25,0, 0 , 0]),
                        np.array([0.5,0,  0, 0]),
                        np.array([1.0, 0, 0, 0]),
                        np.array([1.5, 0, 0, 0])]
                }

    virtual_cams['default'] = { 'src_indx': 1,
                        'view_offsets': [np.array([0.25,0, 0 , 0]),
                        np.array([0.5,0,  0, 0]),
                        np.array([1.0, 0, 0, 0]),
                        np.array([1.5, 0, 0, 0])]
                }

    def __init__(self, args):

        ####################################
        # Create a DeepMVS wrapper object
        ####################################
        filename_DeepMVS = os.path.join(args.models_path, 'DeepMVS_final.model')
        self.models_path = args.models_path
        self.refine_model_path = os.path.join(args.models_path, 'Model_VNPCAT_E33.pth')
        self.deepmvs_obj = deepmvs_wrapper.DeepMVSWrapper(filename_DeepMVS, do_filter=True)

        self.dense_crf_params = {'default': {'sigma_xy': 45.0, 'sigma_rgb': 30.0, 'iteration_num': 5, 'compat': 10.0}}


    def run(self, colmap_seq_path, input_views=[]):

        print('Processing sequence: ', colmap_seq_path)
        seq_name = os.path.basename(os.path.normpath(colmap_seq_path))

        outDir = os.path.join(colmap_seq_path, 'xtreme-view')
        if not os.path.exists(outDir):
            os.mkdir(outDir)

        # Adjust the dense crf parameters if needed
        if seq_name in self.dense_crf_params:
            self.deepmvs_obj.dict_DenseCRF = self.dense_crf_params[colmap_seq_path]
        else:
            self.deepmvs_obj.dict_DenseCRF = self.dense_crf_params['default']

        list_img, list_depth, list_cam_params \
            = colmap_loader.COLMAPData.read_data_to_list(colmap_seq_path)

        if len(input_views) > 0:
            list_img = [list_img[i] for i in input_views]
            list_depth = [list_depth[i] for i in input_views]
            list_cam_params = [list_cam_params[i] for i in input_views]

        #############################
        # Create our vsynth object
        #############################
        view_synthesizer = core.VSynth(list_img, list_cam_params, outDir,
                                    self.deepmvs_obj, list_depth=list_depth,
                                    mode_colmap=True)

        #####################################################
        # Compute the depth probability ( = perform DeepMVS)
        # When the depth probability are stored in the working dir,
        # it will skip without performing DeepMVS
        #####################################################
        view_synthesizer.compute_depth_probability()

        #############################################
        # Create the virtual cameras
        #############################################
        view_synthesizer.list_vcams = []

        if seq_name in self.virtual_cams:
            view_offsets = self.virtual_cams[seq_name]['view_offsets']
            src_indx = self.virtual_cams[seq_name]['src_indx']
        else:
            view_offsets = self.virtual_cams['default']['view_offsets']
            src_indx = self.virtual_cams['default']['src_indx']

        for view_offset in view_offsets:
            new_vcam = deepcopy(view_synthesizer.list_src_cams[src_indx])
            new_vcam['extrinsic'][:,3] = new_vcam['extrinsic'][:,3] + view_offset
            view_synthesizer.list_vcams.append(new_vcam)

        list_todo_index=[] # generate all the cameras
        view_synthesizer.do(MHW_SRCV_WEIGHT=False, list_todo_index=list_todo_index)
        view_synthesizer.refine(self.refine_model_path, list_todo_index=list_todo_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_path', help='the path to the sequence of images.')
    parser.add_argument('--models_path', help='the path where the pre-trained models have been downloaded to.',  default='/models')
    parser.add_argument('--input_views', help='comma-separated list of the indices in the sequence to use as inputs')
    args = parser.parse_args()
    print(args)

    runner = XtremeViewRunner(args)
    input_views = []
    if args.input_views is not None:
        input_views = [int(i) for i in args.input_views.split(',')]
    runner.run(args.seq_path, input_views)