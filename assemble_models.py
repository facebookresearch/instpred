# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""This script is useful to assemble F2Fi models into a F2F model.
NB: assumes multiscale architecture_f2fi and parallel architecture,
etc. see modelsConfig below"""
import os
import torch
import sys

from mypython.logger import create_logger
from torch.nn.parameter import Parameter
from mytorch.implementation_utils import get_nb_parameters

#-------------------------------------------------------------------------------
from config import make_config
opt, configs, checkpoint = make_config()
assert opt['architecture'] == 'parallel'
assert opt['architecture_f2fi'] == 'multiscale'

#-------------------------------------------------------------------------------
# Start logging
logger = create_logger(os.path.join(opt['logs'], 'assemble_models.log'))
logger.info('============ Initialized logger ============')

f2f_paths = {
    'f2f5': os.environ['F2F5_WEIGHTS'],
    'f2f4': os.environ['F2F4_WEIGHTS'],
    'f2f3': os.environ['F2F3_WEIGHTS'],
    'f2f2': os.environ['F2F2_WEIGHTS']
}

output_name = 'F2F_model.net'

#-------------------------------------------------------------------------------
# Create model
modelsConfig = configs['models']
assert opt['architecture_f2fi'] == 'multiscale'
from models import F2F_multiscale as F2F
model = F2F(modelsConfig)

logger.info('This model has %d parameters.' % (get_nb_parameters(model)))
logger.info('F2F5 has %d parameters.' % (get_nb_parameters(model.f2f5)))
logger.info('F2F4 has %d parameters.' % (get_nb_parameters(model.f2f4)))
logger.info('F2F3 has %d parameters.' % (get_nb_parameters(model.f2f3)))
logger.info('F2F2 has %d parameters.' % (get_nb_parameters(model.f2f2)))


submodels = {
    'f2f5': model.f2f5,
    'f2f4': model.f2f4,
    'f2f3': model.f2f3,
    'f2f2': model.f2f2
}

for lvl in f2f_paths:
    params = torch.load(f2f_paths[lvl])
    params = {'.'.join(k.split('.')[1:]) : v for k,v in params.items()}
    submodels[lvl].load_state_dict(params)

#-------------------------------------------------------------------------------
# Save model
output_path = os.path.join(opt['save'], output_name)
torch.save(model.state_dict(), output_path)
logger.info('Model saved in %s' % output_path)
