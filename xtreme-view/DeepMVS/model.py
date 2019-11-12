'''
BSD 2-Clause License

Copyright (c) 2018, Po-Han Huang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMVS(nn.Module):
    def __init__(self, num_depths, use_gpu = True, gpu_id = 0):
        super(DeepMVS, self).__init__()
        # Patch Matching
        self.layer_0 = nn.Sequential(
                nn.Conv2d(3, 64, (5, 5), stride = (1, 1), padding = (2, 2)),
                nn.SELU()
            )
        self.layer_1 = nn.Sequential(
                nn.Conv2d(128, 96, (5, 5), stride = (1, 1), padding = (2, 2)),
                nn.SELU(),
                nn.Conv2d(96, 32, (5, 5), stride = (1, 1), padding = (2, 2)),
                nn.SELU(),
                nn.Conv2d(32, 4, (5, 5), stride = (1, 1), padding = (2, 2)),
                nn.SELU()
            )
        # Encoder
        self.layer_2_e1x = nn.Sequential(
                nn.Conv2d(4 * num_depths, 200, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
                nn.Conv2d(200, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU()
            )
        self.layer_2_e2x = nn.Sequential(
                nn.Conv2d(100, 100, (2, 2), stride = (2, 2), padding = (0, 0)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU()
            )
        self.layer_2_e4x = nn.Sequential(
                nn.Conv2d(100, 100, (2, 2), stride = (2, 2), padding = (0, 0)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
            )
        self.layer_2_e8x = nn.Sequential(
                nn.Conv2d(100, 100, (2, 2), stride = (2, 2), padding = (0, 0)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
            )
        self.layer_2_e16x = nn.Sequential(
                nn.Conv2d(100, 100, (2, 2), stride = (2, 2), padding = (0, 0)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU()
            )
        # Buffer layers for VGG features
        self.layer_b1x = nn.Sequential(
                nn.Conv2d(64, 64, (1, 1), stride = (1, 1), padding = (0, 0)),
                nn.SELU(),
            )
        self.layer_b2x = nn.Sequential(
                nn.Conv2d(128, 100, (1, 1), stride = (1, 1), padding = (0, 0)),
                nn.SELU(),
            )
        self.layer_b4x = nn.Sequential(
                nn.Conv2d(256, 100, (1, 1), stride = (1, 1), padding = (0, 0)),
                nn.SELU(),
            )
        self.layer_b8x = nn.Sequential(
                nn.Conv2d(512, 100, (1, 1), stride = (1, 1), padding = (0, 0)),
                nn.SELU(),
            )
        self.layer_b16x = nn.Sequential(
                nn.Conv2d(512, 100, (1, 1), stride = (1, 1), padding = (0, 0)),
                nn.SELU(),
            )
        # Decoder
        self.layer_2_d16x = nn.Sequential(
                nn.Conv2d(200, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
            )
        self.layer_2_d8x = nn.Sequential(
                nn.Conv2d(300, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU()
            )
        self.layer_2_d4x = nn.Sequential(
                nn.Conv2d(300, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU()
            )
        self.layer_2_d2x = nn.Sequential(
                nn.Conv2d(300, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
                nn.Conv2d(100, 100, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU()
            )
        self.layer_2_d1x = nn.Sequential(
                nn.Conv2d(264, 400, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
                nn.Conv2d(400, 800, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU()
            )
        # Inter-Volume Aggregation
        self.layer_3 = nn.Sequential(
                nn.Conv2d(800, 400, (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.SELU(),
                nn.Conv2d(400, num_depths, (3, 3), stride = (1, 1), padding = (1, 1))
            )
        self.layer_loss = nn.CrossEntropyLoss(ignore_index=-1)

        if use_gpu:
            self.layer_0 = self.layer_0.cuda(gpu_id)
            self.layer_1 = self.layer_1.cuda(gpu_id)
            self.layer_2_e1x = self.layer_2_e1x.cuda(gpu_id)
            self.layer_2_e2x = self.layer_2_e2x.cuda(gpu_id)
            self.layer_2_e4x = self.layer_2_e4x.cuda(gpu_id)
            self.layer_2_e8x = self.layer_2_e8x.cuda(gpu_id)
            self.layer_2_e16x = self.layer_2_e16x.cuda(gpu_id)
            self.layer_b1x = self.layer_b1x.cuda(gpu_id)
            self.layer_b2x = self.layer_b2x.cuda(gpu_id)
            self.layer_b4x = self.layer_b4x.cuda(gpu_id)
            self.layer_b8x = self.layer_b8x.cuda(gpu_id)
            self.layer_b16x = self.layer_b16x.cuda(gpu_id)
            self.layer_2_d16x = self.layer_2_d16x.cuda(gpu_id)
            self.layer_2_d8x = self.layer_2_d8x.cuda(gpu_id)
            self.layer_2_d4x = self.layer_2_d4x.cuda(gpu_id)
            self.layer_2_d2x = self.layer_2_d2x.cuda(gpu_id)
            self.layer_2_d1x = self.layer_2_d1x.cuda(gpu_id)
            self.layer_3 = self.layer_3.cuda(gpu_id)
            self.layer_loss = self.layer_loss.cuda(gpu_id)

    # Shape of 'volume_input': batch_size * num_neighbors (or num_sources) * num_depths * 2 * num_channels * height * width
    # 'feature_inputs' is a list of five VGG feature tensors, each of shape: batch_size * num_features * height * width
    def forward(self, volume_input, feature_inputs):
        (aggregated_feature, _) = torch.max(self.forward_feature(volume_input, feature_inputs), 1)
        return self.forward_predict(aggregated_feature)

    def forward_feature(self, volume_input, feature_inputs):
        if volume_input.dim() != 7 or volume_input.size(3) != 2:
            raise ValueError("'volume_input' must be a tensor of shape: batch_size * num_neighbors (or num_sources) * num_depths * 2 * num_channels * height * width")
        if len(feature_inputs) != 5:
            raise ValueError("'feature_inputs' is a list of five VGG feature tensors of shape: batch_size * num_features * height * width")
        for feature in feature_inputs:
            if feature.dim() != 4:
                raise ValueError("'feature_inputs' is a list of five VGG feature tensors of shape: batch_size * num_features * height * width")
        batch_size = volume_input.size(0)
        num_neighbors = volume_input.size(1)
        num_depths = volume_input.size(2)
        num_channels = volume_input.size(4)
        height = volume_input.size(5)
        width = volume_input.size(6)
        layer_0_output = self.layer_0(
            volume_input.view(batch_size * num_neighbors * num_depths * 2, num_channels, height, width))
        layer_1_output = self.layer_1(
            layer_0_output.view(batch_size * num_neighbors * num_depths, 2 * 64, height, width))
        layer_2_e1x_out = self.layer_2_e1x(layer_1_output.view(batch_size * num_neighbors, num_depths * 4, height, width))
        layer_2_e2x_out = self.layer_2_e2x(layer_2_e1x_out)
        layer_2_e4x_out = self.layer_2_e4x(layer_2_e2x_out)
        layer_2_e8x_out = self.layer_2_e8x(layer_2_e4x_out)
        layer_2_e16x_out = self.layer_2_e16x(layer_2_e8x_out)
        layer_b1x_out = self.layer_b1x(feature_inputs[0])
        layer_b2x_out = self.layer_b2x(feature_inputs[1])
        layer_b4x_out = self.layer_b4x(feature_inputs[2])
        layer_b8x_out = self.layer_b8x(feature_inputs[3])
        layer_b16x_out = self.layer_b16x(feature_inputs[4])
        if num_neighbors != 1:
            # We need to copy the features for each neighbor image. When batch_size = 1, use expand() instead of repeat() to save memory.
            if batch_size == 1:
                layer_b1x_out = layer_b1x_out.expand(batch_size * num_neighbors, -1, -1, -1)
                layer_b2x_out = layer_b2x_out.expand(batch_size * num_neighbors, -1, -1, -1)
                layer_b4x_out = layer_b4x_out.expand(batch_size * num_neighbors, -1, -1, -1)
                layer_b8x_out = layer_b8x_out.expand(batch_size * num_neighbors, -1, -1, -1)
                layer_b16x_out = layer_b16x_out.expand(batch_size * num_neighbors, -1, -1, -1)
            else:
                layer_b1x_out = layer_b1x_out.repeat(num_neighbors, 1, 1, 1)
                layer_b2x_out = layer_b2x_out.repeat(num_neighbors, 1, 1, 1)
                layer_b4x_out = layer_b4x_out.repeat(num_neighbors, 1, 1, 1)
                layer_b8x_out = layer_b8x_out.repeat(num_neighbors, 1, 1, 1)
                layer_b16x_out = layer_b16x_out.repeat(num_neighbors, 1, 1, 1)
        layer_2_d16x_out = self.layer_2_d16x(torch.cat((layer_2_e16x_out, layer_b16x_out), 1))
        layer_2_d8x_out = self.layer_2_d8x(torch.cat((layer_2_e8x_out, F.upsample(layer_2_d16x_out, scale_factor=2, mode='bilinear'), layer_b8x_out), 1))
        layer_2_d4x_out = self.layer_2_d4x(torch.cat((layer_2_e4x_out, F.upsample(layer_2_d8x_out, scale_factor=2, mode='bilinear'), layer_b4x_out), 1))
        layer_2_d2x_out = self.layer_2_d2x(torch.cat((layer_2_e2x_out, F.upsample(layer_2_d4x_out, scale_factor=2, mode='bilinear'), layer_b2x_out), 1))
        layer_2_d1x_out = self.layer_2_d1x(torch.cat((layer_2_e1x_out, F.upsample(layer_2_d2x_out, scale_factor=2, mode='bilinear'), layer_b1x_out), 1))
        return layer_2_d1x_out.view(batch_size, num_neighbors, 800, height, width)

    def forward_predict(self, aggregated_feature):
        layer_3_output = self.layer_3(aggregated_feature)
        return layer_3_output

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        m.bias.data.fill_(0)