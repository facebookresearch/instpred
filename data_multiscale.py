# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import torch
from logging import getLogger
from torch.autograd import Variable

logger = getLogger()

#-------------------------------------------------------------------------------

class featurePredictionSampler(object):
    """Samples elements from the dataset and returns input and target features
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, features, nb_scales):
        self.data_source = data_source
        self.features = features
        self.current = 0
        self.high = len(self.data_source)
        self.nb_scales = nb_scales
        # Just like in the mask rcnn code, use simple spatial subsampling
        self.downsampler = torch.nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def __iter__(self):
        return self

    def next(self):
        if self.current >= self.high:
            raise StopIteration
        else:
            sample = self.data_source[self.current]
            inputs, targets = {}, {}
            high_lev_feat = ['fpn_res5_2_sum', 'fpn_res4_5_sum', 'fpn_res3_3_sum', 'fpn_res2_2_sum']
            for ifeat, feat in enumerate(high_lev_feat):
                if feat in self.features:
                    inputs[feat] = [sample[u'input_features_' + feat]]
                    targets[feat] = [sample[u'target_features_' + feat]]
                    for sc in range(1, self.nb_scales[ifeat]):
                        downsampled_input = self.downsampler(Variable(inputs[feat][0])).data
                        inputs[feat].insert(0, downsampled_input)
                        downsampled_target = self.downsampler(Variable(targets[feat][0])).data
                        targets[feat].insert(0, downsampled_target)

            self.current += 1
            return inputs, targets, sample['seqIDs']

    def __len__(self):
        return len(self.data_source)

    def reset(self, reshuffle = False):
        self.current = 0
        if reshuffle:
            logger.info('Reshuffling dataset.')
            dataset_to_resample = self.data_source
            while not hasattr(dataset_to_resample, 'resample'):
                dataset_to_resample = dataset_to_resample.dataset
            dataset_to_resample.resample()

#-------------------------------------------------------------------------------
from data import create_cityscapes_datasource_train, create_cityscapes_datasource_val
# Main loading functions
def load_cityscapes_train(opt):
    dataset, required_fpn_features, loaded_model = create_cityscapes_datasource_train(opt)
    dataset_loader = featurePredictionSampler(dataset, required_fpn_features, opt['nb_scales'])
    return dataset_loader, loaded_model


def load_cityscapes_val(opt):
    if opt['loaded_model'] is None:
        dataset, required_fpn_features, loaded_model = create_cityscapes_datasource_val(opt)
    else:
        dataset, required_fpn_features = create_cityscapes_datasource_val(opt)
    dataset_loader = featurePredictionSampler(dataset, required_fpn_features, opt['nb_scales'])
    if opt['loaded_model'] is None:
        return dataset_loader, loaded_model
    else:
        return dataset_loader
