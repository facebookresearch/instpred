# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn

from logging import getLogger
logger = getLogger()

feat_ind = {
    'fpn_res5_2_sum': 0,
    'fpn_res4_5_sum': 1,
    'fpn_res3_3_sum': 2,
    'fpn_res2_2_sum': 3
}

downsampler = torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

def format_variable_length_output(outputs, n_outputs, train_single_level=None):
    if train_single_level is None:
        r_outputs = {'fpn_res5_2_sum' : outputs[0]}
        if n_outputs == 1:
            return r_outputs
        else:
            r_outputs['fpn_res4_5_sum'] = outputs[1]
            if n_outputs == 2:
                return r_outputs
            else:
                r_outputs['fpn_res3_3_sum'] = outputs[2]
                if n_outputs == 3:
                    return r_outputs
                else:
                    r_outputs['fpn_res2_2_sum'] = outputs[3]
                    return r_outputs
    else:
        return {train_single_level : outputs}

def update_autoregressive_multiscale_inputs(inputs, outputs, nb_feat, nI, nb_scales):
    new_inputs = {}
    for k, v in inputs.items():
        new_inputs[k] = []
        ns = nb_scales[feat_ind[k]]
        out = outputs[k][ns-1]
        for sc in range(ns-1, -1, -1):
            inp = v[sc]
            if sc < ns-1:
                out = downsampler(out)
            assert nb_feat * nI == inp.size(1), \
                '%d, %d are not equal and should be' % (nb_feat * nI, v.size(1))
            assert inp.dim() == 4 == out.dim()
            st, en = nb_feat * 1, nb_feat * nI
            new_inputs[k].insert(0, torch.cat((inp[:, st:en, :, :], out), 1))
            assert new_inputs[k][0].size() == inp.size()
    return new_inputs

def transform_into_list(seq_outputs):
    # Returns a list where
    rslt = []
    # for each feature
    for k in seq_outputs[0]:
        # we list the predictions for that feature
        for t in range(len(seq_outputs)):
            rslt.append(seq_outputs[t][k])
    return rslt

class Autoregressive(nn.Module):
    def __init__(self, config, single_frame_model):
        super(Autoregressive, self).__init__()
        self.ffpn_levels = config['FfpnLevels']
        self.single_frame_model = single_frame_model
        self.n_target_frames_ar = config['n_target_frames_ar']
        self.nb_feat = config['nb_features']
        self.n_input_frames = config['n_input_frames']
        self.train_single_level = config['train_single_level']
        self.nb_scales = config['nb_scales']

    def forward(self, inputs):
        """
        At each time step: takes the input features, uses the single frame model
        given to predict the next future features, then updates the inputs to
        prepare for the next step.
        """
        seq_outputs = []
        for t in range(self.n_target_frames_ar):
            features_inputs = inputs
            if not self.train_single_level is None:
                features_inputs = features_inputs[self.train_single_level]
            outputs = format_variable_length_output(self.single_frame_model(features_inputs), self.ffpn_levels, train_single_level = self.train_single_level)
            seq_outputs.append(outputs)
            if t < self.n_target_frames_ar - 1:
                inputs = update_autoregressive_multiscale_inputs(inputs, outputs, self.nb_feat, self.n_input_frames, self.nb_scales)
        return transform_into_list(seq_outputs)
