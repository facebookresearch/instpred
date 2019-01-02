# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""This script is used to precompute features extracted by the Mask R-CNN
backbone."""

import os
import pprint
import copy
import numpy as np
from detectron.utils.io import save_object

from mypython.logger import create_logger

#-------------------------------------------------------------------------------
# Load config
from config import make_config
opt, configs, _ = make_config()
assert opt['precompute_features'], 'please specify which types of features to compute, ex : fpn_res5_2_sum'

#-------------------------------------------------------------------------------
# Start logging
logger = create_logger(os.path.join(opt['logs'], 'compute_features.log'))
logger.info('============ Initialized logger ============')

#-------------------------------------------------------------------------------
# Create dataloaders
trainsetConfig = configs['trainset']
valsetConfig = configs['valset']

from data import load_cityscapes_train, load_cityscapes_val
logger.info(pprint.pprint(trainsetConfig))
trainset, loaded_model = load_cityscapes_train(trainsetConfig)
# to remove shuffling of the dataset
trainset.data_source.dataset.dataset = trainset.data_source.dataset.dataset.dataset
trainLoader = iter(trainset)

valsetConfig['loaded_model'] = loaded_model
logger.info(pprint.pprint(valsetConfig))
valset = load_cityscapes_val(valsetConfig)
valLoader = iter(valset)

#-------------------------------------------------------------------------------
# Precompute features

def precompute_maskrcnn_backbone_features(config, dataset, split):
    feat_type = config['feat_type']
    # assert that the dimensions are ok otherwise break
    s, nI, nT = min(len(dataset), config['it']), config['n_input_frames'], config['n_target_frames']

    # Automatically get feature dimensions
    sample_input, _, _ = dataset.next()
    sample_features = sample_input[feat_type]
    assert sample_features.dim() == 4, "Batch mode is expected"
    sz = sample_features.size()
    assert(sz[0] == 1, 'This function assumes batch mode, but a single example per batch')
    c, h, w = sz[1]/nI, sz[2], sz[3]
    assert c == config['nb_features']
    dataset.reset()
    # Check that the dataset to compute will be under 100GB - floating point takes 4B - check
    assert s * (nI+nT) * c * h * w * 4 <= 1e11, \
        'The dataset to compute will take over 100 GB - aborting'
    # Initialize tensors
    seq_features = np.empty((s, (nI+nT), c, h, w), dtype = np.float32)
    seq_ids = ['' for _ in range(s)]
    for i, data in enumerate(dataset):
        inputs, targets, _ = data
        correspondingSample = dataset.data_source[i]
        # insert in the dataset
        inp_feat = inputs[feat_type].view((nI, c, h, w)).numpy().astype(np.float32)
        tar_feat = targets[feat_type].view((nT, c, h, w)).numpy().astype(np.float32)
        seq_features[i] = np.concatenate((inp_feat, tar_feat), 0)
        seq_ids[i] = correspondingSample[u'annID'][0]
        if i >= (config['it']-1) :
            break

    # save the precomputed features
    fr = config['frame_ss']
    filename = os.path.join(opt['save'] , '__'.join((split, feat_type, 'nSeq%d'%(nI+nT), 'fr%d'%fr)))
    logger.info('Precomputed features saved to : %s'%filename)
    save_object(
        dict(
            sequence_ids = seq_ids
        ),
        filename + '__ids.pkl'
    )
    np.save(filename + '__features.npy', seq_features)

assert opt['trainbatchsize'] == opt['valbatchsize'] == 1
cfg_train = {
    'n_input_frames' : opt['n_input_frames'],
    'n_target_frames' : opt['n_target_frames'],
    'nb_features' : opt['nb_features'],
    'feat_type' : opt['precompute_features'],
    'frame_ss' : opt['frame_subsampling'],
    'it' : opt['ntrainIt']
}
cfg_val = copy.copy(cfg_train)
cfg_val['it'] = opt['nvalIt']
print(cfg_train)
print(cfg_val)

precompute_maskrcnn_backbone_features(cfg_val, valLoader, 'val')
precompute_maskrcnn_backbone_features(cfg_train, trainLoader, 'train')
