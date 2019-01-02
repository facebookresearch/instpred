# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

import os
#-------------------------------------------------------------------------------
def create_multiscale_criterion_table(nb_scales, critname):
    crittable = []
    for sc in range(nb_scales):
        crittable.append(critname())
    return crittable


class FfpnCriterion(nn.Module):
    def __init__(self, config):
        super(FfpnCriterion, self).__init__()
        self.opt = config
        self.criterions = {}
        self.gpu_id = config['gpu_id']

        levelNames = ['fpn_res5_2_sum', 'fpn_res4_5_sum', 'fpn_res3_3_sum', 'fpn_res2_2_sum']
        for l in range(self.opt['FfpnLevels']):
            if not config['train_single_level'] is None and levelNames[l] != str(config['train_single_level']):
                continue
            crit = {}
            if 'mse' in self.opt['loss_features']:
                if config['nb_scales'] is None:
                    crit['mse'] = nn.MSELoss()
                else:
                    crit['mse'] = create_multiscale_criterion_table(config['nb_scales'], nn.MSELoss)

            if 'l1' in self.opt['loss_features']:
                if config['nb_scales'] is None:
                    crit['l1'] = nn.L1Loss()
                else:
                    crit['l1'] = create_multiscale_criterion_table(config['nb_scales'], nn.L1Loss)

            # Add here any other auxiliary loss you could want to train predicted features i,l with

            self.criterions[levelNames[l]] = crit

        self.levelNames = levelNames
        self.train_single_level = config['train_single_level']
        self.nb_scales = config['nb_scales']


    def forward(self, inputs, targets, encoder_activations=None):
        l = 0
        loss_terms = {}
        if self.train_single_level is None:
            assert len(inputs) == self.opt['FfpnLevels']
        else:
            assert len(inputs) == 1
        for i, k in enumerate(inputs):
            for loss_type in self.opt['loss_features']:
                if self.nb_scales is None:
                    val = self.criterions[k][loss_type](inputs[k], targets[k])
                    loss_terms['%s-%s' % (k, loss_type)] = val.item()
                    l += val
                else:
                    for sc in range(self.nb_scales):
                        val = self.criterions[k][loss_type][sc](inputs[k][sc], targets[k][sc])
                        loss_terms['%s-%s-%s' % (k, loss_type, sc)] = val.item()
                        l += val

        return l, loss_terms

    def __str__(self, indent=''):
        from mytorch.printing import bcolors
        s = indent + 'FfpnCriterion {\n'

        for l in range(self.opt['FfpnLevels']):
            lev = self.levelNames[l]
            if not self.train_single_level is None and lev != str(self.train_single_level):
                continue
            s += indent + '\t' + bcolors.PURPLE + lev + bcolors.END + ' : {\n \t\t '
            ct = 0
            for k, v in self.criterions[lev].items():
                coeff = 1
                if self.nb_scales is None:
                    s+= indent + '%s: ' %k + bcolors.CYAN + str(v).split(' ')[0]
                    ct += 1
                    if ct < len(self.criterions[lev]) :
                        s += '()' + bcolors.END + ', lambda : %f\n \t\t ' % coeff
                else:
                    for sc in range(self.nb_scales):
                        s+= indent + '%s: ' %k + bcolors.CYAN + str(v[sc]).split(' ')[0]
                        if sc < self.nb_scales-1 or ct < len(self.criterions[lev])-1:
                            s+= '()' + bcolors.END + ', lambda : %f, scale: %d\n \t\t ' % (coeff, sc)
                    ct += 1
                if ct == len(self.criterions[lev]) :
                    s += '()' + bcolors.END + ', lambda : %f\n\t' % coeff +indent+'}'
        s += '\n'+indent + '}'
        return s

    def __repr__(self):
        return self.criterions


class AutoregressiveCriterion(nn.Module):
    def __init__(self, config, single_frame_criterion):
        super(AutoregressiveCriterion, self).__init__()
        self.n_target_frames_ar = config['n_target_frames_ar']
        self.single_frame_criterion = single_frame_criterion

    def forward(self, inputs, targets):
        """ Forward assumes that inputs and targets are in the shape of a list
        of n_target_frames_autoregressive dictionaries containing the features
        for the trained levels of the ffpn.
        It also assumes that the same criterion is applied at each time step.
        The final loss is the sum over the times steps of each single time step
        component.
        """
        assert len(inputs)  == self.n_target_frames_ar
        assert len(targets) == self.n_target_frames_ar
        all_loss_terms = []
        final_loss = 0
        for t in range(self.n_target_frames_ar):
            loss, loss_terms = self.single_frame_criterion(inputs[t], targets[t])
            final_loss += loss
            all_loss_terms.append(loss_terms)
        return final_loss, all_loss_terms

    def __str__(self):
        from mytorch.printing import bcolors
        s = 'AutoregressiveCriterion {\n'
        for t in range(self.n_target_frames_ar):
            s += '\t' + bcolors.GREEN + 't+%d' % (t+1) + bcolors.END + ' : {\n'
            s += self.single_frame_criterion.__str__(indent = '\t')
            s += '\n}'
        return s
