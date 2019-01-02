# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from logging import getLogger

logger = getLogger()

#-------------------------------------------------------------------------------
# Individual architectures
class F2Fi_l(nn.Module):
    def __init__(self, opt):
        super(F2Fi_l, self).__init__()
        nCPI, nI = opt['n_channels_per_input'], opt['n_input_frames']
        nCPT, nT = opt['n_channels_per_target'], opt['n_target_frames']
        ms = opt['model_size']

        assert 32*ms >=   nCPT*nT, 'Too much compression before output !'

        self.conv1 = nn.Conv2d(nCPI*nI,  64*ms, 1, 1, 0)
        self.conv2 = nn.Conv2d(64*ms,    64*ms, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(64*ms,    64*ms, 3, 1, 2, 2)
        self.conv4 = nn.Conv2d(64*ms,    32*ms, 3, 1, 4, 4)
        self.conv5 = nn.Conv2d(32*ms,    32*ms, 3, 1, 4, 4)
        self.conv6 = nn.Conv2d(32*ms,    32*ms, 3, 1, 2, 2)
        self.conv7 = nn.Conv2d(32*ms,    nCPT*nT, 7, 1, 3)

        self._initialize_weights()

    def forward(self, x):
        # operations to perform
        x = self.conv1(x)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.conv7(F.relu(x))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

class F2Fi_multiscale_refine(nn.Module):
    """Architecture for F2F_i,l refining the previous coarse prediction i-1,l"""
    def __init__(self, opt):
        super(F2Fi_multiscale_refine, self).__init__()
        nCPI, nI = opt['n_channels_per_input'], opt['n_input_frames']
        nCPT, nT = opt['n_channels_per_target'], opt['n_target_frames']
        ms = opt['model_size']
        if 'return_feat' in opt:
            self.return_feat = opt['return_feat']
        else:
            self.return_feat = None
        assert 32*ms >=   nCPT*nT, 'Too much compression before output !'
        # Exactly the same as F2F_lesser_compression except one more set of feature maps in input, predicted by lower level.
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(nCPI*(nI+1),  64*ms, 1, 1, 0)
        self.conv2 = nn.Conv2d(64*ms,    64*ms, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(64*ms,    64*ms, 3, 1, 2, 2)
        self.conv4 = nn.Conv2d(64*ms,    32*ms, 3, 1, 4, 4)
        self.conv5 = nn.Conv2d(32*ms,    32*ms, 3, 1, 4, 4)
        self.conv6 = nn.Conv2d(32*ms,    32*ms, 3, 1, 2, 2)
        self.conv7 = nn.Conv2d(32*ms,    nCPT*nT, 7, 1, 3)

        self._initialize_weights()

    def forward(self, x, coarse):
        # Upsample coarse
        upcoarse = self.upsampler(coarse)
        # Concatenate inputs and coarse prediction
        ref = torch.cat([x, upcoarse], 1)
        # Compute refinement
        ref = self.conv3(F.relu(self.conv2(F.relu(self.conv1(ref)))))

        ref = self.conv4(F.relu(ref))
        if self.return_feat == 'conv4':
            return ref

        ref = self.conv5(F.relu(ref))
        if self.return_feat == 'conv5':
            return ref

        ref = self.conv6(F.relu(ref))
        if self.return_feat == 'conv6':
            return ref

        ref = self.conv7(F.relu(ref))
        if self.return_feat == 'conv7':
            return ref

        assert self.return_feat is None
        # Return the sum
        return upcoarse + ref

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


# Convention:inputs, outputs, models, targets, losses will be in order of coarsest to largest
class F2Fi_multiscale(nn.Module):
    """Architecture for F2F_0,l giving a (coarse if nb_scales >1) prediction
    for level l."""
    def __init__(self, opt):
        super(F2Fi_multiscale, self).__init__()
        self.nb_scales = opt['nb_scales']
        models = []
        opt_first = copy.deepcopy(opt)
        if opt['nb_scales'] > 1: opt_first['return_feat'] = None
        models.append(
            F2Fi_l(opt_first)
        )
        for sc in range(1, self.nb_scales):
            opt_next = copy.deepcopy(opt)
            if sc < opt['nb_scales']-1: opt_next['return_feat'] = None
            models.append(F2Fi_multiscale_refine(opt_next))
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        outputs.append(self.models[0](x[0]))
        for sc in range(1, self.nb_scales):
            outputs.append(self.models[sc](x[sc], outputs[sc-1]))
        return outputs


class F2F_multiscale(nn.Module):
    """F2F architecture, relying on all F2F_l networks. Used in evaluation."""
    def __init__(self, config):
        super(F2F_multiscale, self).__init__()
        self.opt = config
        opt_f2f5 = copy.deepcopy(self.opt)
        opt_f2f5['nb_scales'] = opt_f2f5['nb_scales'][0]
        assert config['architecture_f2fi'] == 'multiscale'
        self.f2f5 = F2Fi_multiscale(opt_f2f5)
        if self.opt['FfpnLevels'] > 1:
            if self.opt['f2f_sharing']:
                self.f2f4 = self.f2f5
            else:
                opt_f2f4 = copy.deepcopy(self.opt)
                opt_f2f4['nb_scales'] = opt_f2f4['nb_scales'][1]
                self.f2f4 = F2Fi_multiscale(opt_f2f4)
        if self.opt['FfpnLevels'] > 2:
            if self.opt['f2f_sharing']:
                self.f2f3 = self.f2f5
            else:
                opt_f2f3 = copy.deepcopy(self.opt)
                opt_f2f3['nb_scales'] = opt_f2f3['nb_scales'][2]
                self.f2f3 = F2Fi_multiscale(opt_f2f3)
        if self.opt['FfpnLevels'] > 3:
            if self.opt['f2f_sharing']:
                self.f2f2 = self.f2f5
            else:
                opt_f2f2 = copy.deepcopy(self.opt)
                opt_f2f2['nb_scales'] = opt_f2f2['nb_scales'][3]
                self.f2f2 = F2Fi_multiscale(opt_f2f2)

    def forward(self, x):
        y = {}
        y[u'fpn_res5_2_sum'] = self.f2f5(x[u'fpn_res5_2_sum'])
        if self.opt['FfpnLevels'] == 1:
            return [y[u'fpn_res5_2_sum']]
        else:
            y[u'fpn_res4_5_sum'] = self.f2f4(x[u'fpn_res4_5_sum'])
            if self.opt['FfpnLevels'] == 2:
                return y[u'fpn_res5_2_sum'], y[u'fpn_res4_5_sum']
            else:
                y[u'fpn_res3_3_sum'] = self.f2f3(x[u'fpn_res3_3_sum'])
                if self.opt['FfpnLevels'] == 3:
                    return y[u'fpn_res5_2_sum'], y[u'fpn_res4_5_sum'], y[u'fpn_res3_3_sum']
                else:
                    y[u'fpn_res2_2_sum'] = self.f2f2(x[u'fpn_res2_2_sum'])
                    return y[u'fpn_res5_2_sum'], y[u'fpn_res4_5_sum'], y[u'fpn_res3_3_sum'], y[u'fpn_res2_2_sum']
