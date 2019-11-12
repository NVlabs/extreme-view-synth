"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/NVlabs/extreme-view-synth.
Authors: Inchang Choi, Orazio Gallo, Alejandro Troccoli, Min H. Kim, and Jan Kautz
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.init as init

#
# VNPCAT = Variable Number of Patch using concatenation
#

class Model_VNPCAT_Encoder(nn.Module):
    # Based on Unet and inpainting network
    def __init__(self, num_in_features = 3):
        super(Model_VNPCAT_Encoder, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(num_in_features, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_bnorm = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_bnorm = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_bnorm = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv5_bnorm = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv6_bnorm = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, 3, 2, 1)
        self.conv7_bnorm = nn.BatchNorm2d(512)

        self.apply(self.initialize_weight)

    def forward(self, x):

        # encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2_bnorm(self.conv2(x1)))

        x3 = self.relu(self.conv3_bnorm(self.conv3(x2)))
        x4 = self.relu(self.conv4_bnorm(self.conv4(x3)))

        x5 = self.relu(self.conv5_bnorm(self.conv5(x4)))
        x6 = self.relu(self.conv6_bnorm(self.conv6(x5)))

        x7 = self.relu(self.conv7_bnorm(self.conv7(x6)))

        return [x2, x4, x6, x7]

    def initialize_weight(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Model_VNPCAT_Decoder(nn.Module):
    # Based on Unet and inpainting network
    def __init__(self):
        super(Model_VNPCAT_Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv2d(512*2, 512, 3, 1, 1)
        self.conv1_bnorm = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2_bnorm = nn.BatchNorm2d(512)
        self.conv2_up = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2_up_bnorm = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512*3, 512, 3, 1, 1)
        self.conv3_bnorm = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_bnorm = nn.BatchNorm2d(512)
        self.conv4_up = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv4_up_bnorm = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256*3, 256, 3, 1, 1)
        self.conv5_bnorm = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv6_bnorm = nn.BatchNorm2d(256)
        self.conv6_up = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv6_up_bnorm = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128*3, 128, 3, 1, 1)
        self.conv7_bnorm = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv8_bnorm = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 3, 3, 1, 1)

        self.apply(self.initialize_weight)

    def forward(self, list_F_synth, list_F_max):

        # encoder
        F_synth_3 = list_F_synth[3]
        F_max_3 = list_F_max[3]
        x0 = torch.cat((F_synth_3, F_max_3), 1)
        x1 = self.relu(self.conv1_bnorm(self.conv1(x0)))
        x2 = self.relu(self.conv2_bnorm(self.conv2(x1)))
        x2_up = self.relu(self.conv2_up_bnorm(self.conv2_up(self.upsample(x2))))

        F_synth_2 = list_F_synth[2]
        F_max_2 = list_F_max[2]
        x2_cat = torch.cat((x2_up, F_synth_2, F_max_2), 1)
        x3 = self.relu(self.conv3_bnorm(self.conv3(x2_cat)))
        x4 = self.relu(self.conv4_bnorm(self.conv4(x3)))
        x4_up = self.relu(self.conv4_up_bnorm(self.conv4_up(self.upsample(x4))))

        F_synth_1 = list_F_synth[1]
        F_max_1 = list_F_max[1]
        x4_cat = torch.cat((x4_up, F_synth_1, F_max_1), 1)
        x5 = self.relu(self.conv5_bnorm(self.conv5(x4_cat)))
        x6 = self.relu(self.conv6_bnorm(self.conv6(x5)))
        x6_up = self.relu(self.conv6_up_bnorm(self.conv6_up(self.upsample(x6))))

        F_synth_0 = list_F_synth[0]
        F_max_0 = list_F_max[0]
        x6_cat = torch.cat((x6_up, F_synth_0, F_max_0), 1)
        x7 = self.relu(self.conv7_bnorm(self.conv7(x6_cat)))
        x8 = self.relu(self.conv8_bnorm(self.conv8(x7)))
        x9 = self.conv9(x8)

        return x9

    def initialize_weight(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Model_VNPCAT(nn.Module):
    # Based on Unet and inpainting network
    def __init__(self):
        super(Model_VNPCAT, self).__init__()
        self.E = Model_VNPCAT_Encoder()
        self.D = Model_VNPCAT_Decoder()
        self.apply(self.initialize_weight)

    def forward(self, x_synth, list_x_candi):

        # encoder
        list_F_synth = self.E(x_synth)
        list_list_F_candi = []
        for x_candi in list_x_candi:
            list_F_candi = self.E(x_candi)
            list_list_F_candi.append(list_F_candi)

        # do max pool
        list_F0 = []
        list_F1 = []
        list_F2 = []
        list_F3 = []

        for list_F_candi in list_list_F_candi:
            list_F0.append(list_F_candi[0][None])
            list_F1.append(list_F_candi[1][None])
            list_F2.append(list_F_candi[2][None])
            list_F3.append(list_F_candi[3][None])

        concat_F0 = torch.cat(list_F0)
        concat_F1 = torch.cat(list_F1)
        concat_F2 = torch.cat(list_F2)
        concat_F3 = torch.cat(list_F3)

        F0_max, _ = torch.max(concat_F0, dim=0)
        F1_max, _ = torch.max(concat_F1, dim=0)
        F2_max, _ = torch.max(concat_F2, dim=0)
        F3_max, _ = torch.max(concat_F3, dim=0)

        list_F_max = [F0_max, F1_max, F2_max, F3_max]

        # decoder
        x_refined = self.D(list_F_synth, list_F_max)

        return x_refined

    def initialize_weight(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
