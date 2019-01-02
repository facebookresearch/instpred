# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import torch
from torchnet.dataset import ShuffleDataset, BatchDataset, TransformDataset
from cityscapesDatasetAndFeatures import CityscapesDatasetAndFeatures
from logging import getLogger

logger = getLogger()

#-------------------------------------------------------------------------------
class featurePredictionSampler(object):
    """Samples elements from the dataset and returns input and target features
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, features):
        self.data_source = data_source
        self.features = features
        self.current = 0
        self.high = len(self.data_source)

    def __iter__(self):
        return self

    def next(self):
        if self.current >= self.high:
            raise StopIteration
        else:
            sample = self.data_source[self.current]
            inputs, targets = {}, {}
            for feat in self.features:
                inputs[feat] = sample[u'input_features_' + feat]
                targets[feat] = sample[u'target_features_' + feat]
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
FPNfeatures = [u'fpn_res5_2_sum', u'fpn_res4_5_sum', u'fpn_res3_3_sum', u'fpn_res2_2_sum']

def intersect(a, b):
    return list(set(a) & set(b))

def create_cityscapes_datasource_train(opt):
    required_fpn_features = intersect(opt['features'], FPNfeatures)
    cityscapes = CityscapesDatasetAndFeatures(
        split = 'train',
        frame_ss = opt['frame_ss'],
        nSeq = opt['n_input_frames'] + opt['n_target_frames'],
        features = opt['features'],
        savedir = opt['save'],
        size = opt['nIt'] * opt['batchsize']
    )
    loaded_model = cityscapes.model

    def form_input_and_target_features(sample):
        for feat in required_fpn_features:
            sz = sample[feat].size()
            nI, nCPI = opt['n_input_frames'], opt['n_channels_per_input']
            nT, nCPT = opt['n_target_frames'], opt['n_channels_per_target']
            sample[u'input_features_' + feat] = sample[feat][0:nI, :, :, :]
            sample[u'input_features_' + feat] = \
                sample[u'input_features_' + feat].view((nI *nCPI, sz[2], sz[3]))
            sample[u'target_features_' + feat] = sample[feat][nI:, :, :, :]
            sample[u'target_features_' + feat] = \
                sample[u'target_features_' + feat].view((nT *nCPT, sz[2], sz[3]))

        return sample

    shuffled_cityscapes = ShuffleDataset(dataset = cityscapes)

    dataset = BatchDataset( # batches
        dataset = TransformDataset( # forms input and target features
            dataset = shuffled_cityscapes,
            transforms = form_input_and_target_features
        ),
        batchsize = opt['batchsize'],
    )

    return dataset, required_fpn_features, loaded_model


# Main loading functions
def load_cityscapes_train(opt):
    dataset, required_fpn_features, loaded_model = create_cityscapes_datasource_train(opt)
    dataset_loader = featurePredictionSampler(dataset, required_fpn_features)
    return dataset_loader, loaded_model


def create_cityscapes_datasource_val(opt):
    required_fpn_features = intersect(opt['features'], FPNfeatures)
    split = 'test' if opt['test_set'] else 'val'
    cityscapes = CityscapesDatasetAndFeatures(
        split = split,
        frame_ss = opt['frame_ss'],
        nSeq = opt['n_input_frames'] + opt['n_target_frames'],
        features = opt['features'],
        savedir = opt['save'],
        size = opt['nIt'] * opt['batchsize'],
        loaded_model = opt['loaded_model']
    )
    if opt['loaded_model'] is None:
        loaded_model = cityscapes.model

    def form_input_and_target_features(sample):
        for feat in required_fpn_features:
            sz = sample[feat].size()
            nI, nCPI = opt['n_input_frames'], opt['n_channels_per_input']
            nT, nCPT = opt['n_target_frames'], opt['n_channels_per_target']
            sample[u'input_features_' + feat] = sample[feat][0:nI, :, :, :]
            sample[u'input_features_' + feat] = \
                sample[u'input_features_' + feat].view((nI *nCPI, sz[2], sz[3]))
            sample[u'target_features_' + feat] = sample[feat][nI:, :, :, :]
            sample[u'target_features_' + feat] = \
                sample[u'target_features_' + feat].view((nT *nCPT, sz[2], sz[3]))

        return sample

    dataset = BatchDataset( # batches
        dataset = TransformDataset( # forms input and target features
            dataset = cityscapes,
            transforms = form_input_and_target_features
        ),
        batchsize = opt['batchsize']
    )

    if opt['loaded_model'] is None:
        return dataset, required_fpn_features, loaded_model
    else:
        return dataset, required_fpn_features

def load_cityscapes_val(opt):
    if opt['loaded_model'] is None:
        dataset, required_fpn_features, loaded_model = create_cityscapes_datasource_val(opt)
    else:
        dataset, required_fpn_features = create_cityscapes_datasource_val(opt)
    dataset_loader = featurePredictionSampler(dataset, required_fpn_features)
    if opt['loaded_model'] is None:
        return dataset_loader, loaded_model
    else:
        return dataset_loader
