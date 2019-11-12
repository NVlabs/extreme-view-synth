"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/NVlabs/extreme-view-synth
Authors: Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H. Kim, and Jan Kautz
"""


import numpy as np
import cv2
import imageio
import json
import os
import sys
from pyquaternion import Quaternion

colmap_root = os.getenv('COLMAP_ROOT', '/workspace/colmap')
sys.path.append(os.path.join(colmap_root, 'scripts', 'python'))
import read_model


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

class COLMAPData():

    @staticmethod
    def read_data_to_list(seq_path):

        list_img = []
        list_depth = []
        list_cam_params = []

        # read camera params
        c_cams, c_images, c_points3D = read_model.read_model('%s/dense/0/sparse' % seq_path, '.bin')

        # read image and depth
        img_dir = os.path.join(seq_path, 'dense', '0', 'images')
        img_list = os.listdir(img_dir)
        img_list.sort()

        for idx, img_name in enumerate(img_list):
            filename_img = os.path.join(img_dir, img_name)
            filename_depth = os.path.join(seq_path, 'dense', '0', 'stereo', 'depth_maps', '%s.geometric.bin' % img_name)

            # read images
            img = imageio.imread(filename_img).astype(np.float32) / 255.0
            list_img.append(img)

            # read depths
            depth = read_array(filename_depth)

            min_depth, max_depth = np.percentile(depth, [5, 90])
            depth[depth < min_depth] = min_depth
            depth[depth > max_depth] = max_depth
            list_depth.append(depth)

            # fetch the camera params
            for key in c_images:
                image_key = c_images[key]
                image_name = image_key.name
                if image_name == img_name:
                    key_to_fetch_for_cam = image_key.camera_id
                    key_to_fetch_for_image = key

            params = {}
            c_cam = c_cams[key_to_fetch_for_cam]
            params['f_x'] = c_cam.params[0]
            params['f_y'] = c_cam.params[1]
            params['c_x'] = c_cam.params[2]
            params['c_y'] = c_cam.params[3]

            c_image = c_images[key_to_fetch_for_image]
            q = Quaternion(c_image.qvec)
            e = np.zeros(shape=(4, 4))
            e[0:3, 0:3] = q.rotation_matrix
            e[0:3, 3] = c_image.tvec
            e[3, 3] = 1.0
            int_mat = np.array([[params['f_x'], 0.0, params['c_x']],
                                [0.0, params['f_y'], params['c_y']],
                                [0.0, 0.0, 1.0]])
            cam = {}
            cam['extrinsic'] = e
            cam['intrinsic'] = int_mat
            list_cam_params.append(cam)

        return list_img, list_depth, list_cam_params
