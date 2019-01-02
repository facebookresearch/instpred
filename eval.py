# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
""" This script evaluates full F2F models on the val/test sets."""
import cv2
import os
import sys
import pprint

import torch
from mypython.logger import create_logger

#-------------------------------------------------------------------------------
from config import make_config
opt, configs, _ = make_config()
assert opt['train_single_level'] is None
assert opt['n_target_frames'] == 1 # >1 means batch prediction of >1 time steps
assert opt['n_target_frames_ar'] >= 1
#-------------------------------------------------------------------------------
# Start logging
logger = create_logger(os.path.join(opt['logs'], 'eval.log'))
logger.info('============ Initialized logger ============')
logger.info(pprint.pformat(opt))
#-------------------------------------------------------------------------------
# Create dataloader
valsetConfig = configs['valset']
from data_multiscale import load_cityscapes_val

logger.info(pprint.pprint(valsetConfig))
valsetConfig['loaded_model'] = None
valset = load_cityscapes_val(valsetConfig)
valLoader, loaded_model = iter(valset)

#-------------------------------------------------------------------------------
# Create model
modelsConfig = configs['models']

if opt['architecture'] == 'parallel':
    if opt['architecture_f2fi'] == 'multiscale':
        from models import F2F_multiscale as F2F
    else:
        raise NotImplementedError('architecture F2F not implemented: %r' % opt['architecture_f2fi'])
else:
    raise NotImplementedError('architecture not implemented: %r' % opt['architecture'])

single_frame_model = F2F(modelsConfig)

autoregressiveConfig = {
    'FfpnLevels'            : opt['FfpnLevels'],
    'n_target_frames_ar'    : opt['n_target_frames_ar'],
    'nb_features'           : opt['nb_features'],
    'n_input_frames'        : opt['n_input_frames'],
    'train_single_level'    : opt['train_single_level'],
    'nb_scales'             : opt['nb_scales']
}

from autoregressive import Autoregressive
model = Autoregressive(autoregressiveConfig, single_frame_model)

assert not opt['model'] is None, 'This script is only for evaluation of a given model'
model.single_frame_model.load_state_dict(torch.load(opt['model']))
logger.info('loaded' + opt['model'])

model.cuda(opt['id_gpu_model'])
logger.info(model)
from mytorch.implementation_utils import get_nb_parameters
logger.info('This model has %d parameters.' % (get_nb_parameters(model)))

#-------------------------------------------------------------------------------
# Evaluate
from evaluation_functions import evaluate
evalConfig = configs['eval']
evaluate(evalConfig, model, valLoader)
