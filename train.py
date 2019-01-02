# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Main for training a single subnetwork of F2F.
"""

import os
import pprint

import torch
import torch.optim as optim

from mypython.logger import create_logger

#-------------------------------------------------------------------------------
from config import make_config
opt, configs, checkpoint = make_config()
assert not opt['train_single_level'] is None and \
    (opt['train_single_level'] == 'fpn_res5_2_sum' or \
    opt['train_single_level'] == 'fpn_res4_5_sum' or \
    opt['train_single_level'] == 'fpn_res3_3_sum' or \
    opt['train_single_level'] == 'fpn_res2_2_sum'), \
    'This script should be used to train a single level of the feature '\
    'pyramid in autoregressive fashion. But train_single_level is '\
    '%r' % opt['train_single_level']

#-------------------------------------------------------------------------------
# Start logging
logger = create_logger(os.path.join(opt['logs'], 'train.log'))
logger.info('============ Initialized logger ============')

#-------------------------------------------------------------------------------
# Create dataloader
trainsetConfig = configs['trainset']
valsetConfig = configs['valset']

from data_multiscale import load_cityscapes_train, load_cityscapes_val
logger.info(pprint.pprint(trainsetConfig))
trainset, loaded_model = load_cityscapes_train(trainsetConfig)
trainLoader = iter(trainset)

valsetConfig['loaded_model'] = loaded_model
logger.info(pprint.pprint(valsetConfig))
valset = load_cityscapes_val(valsetConfig)
valLoader = iter(valset)

#-------------------------------------------------------------------------------
# Create F2Fi model
modelsConfig = configs['models']
if opt['architecture_f2fi'] == 'multiscale':
    from models import F2Fi_multiscale as F2Fi
else:
    raise NotImplementedError('architecture F2F not implemented: %r' % opt['architecture_f2fi'])
modelsConfig['nb_scales'] = modelsConfig['nb_scales'][0] # they are all equal since train_single_level is set

single_frame_model = F2Fi(modelsConfig)

autoregressiveConfig = {
    'FfpnLevels'            : opt['FfpnLevels'],
    'n_target_frames_ar'    : opt['n_target_frames_ar'],
    'nb_features'           : opt['nb_features'],
    'n_input_frames'        : opt['n_input_frames'],
    'train_single_level'    : opt['train_single_level'],
    'nb_scales'            : opt['nb_scales']
}

from autoregressive import Autoregressive
model = Autoregressive(autoregressiveConfig, single_frame_model)

# Note : when resume is used, optimiser state is also loaded to resume the
# training ; when model is used, only the model weights are loaded
if checkpoint is None:
    if opt['model'] is not None:
        saved = torch.load(opt['model'])
        # only load the parameters in the model performing the coarsest prediction
        saved = { '.'.join(k.split('.')[-2:]): v  for k,v in saved.items()}
        model.single_frame_model.models[0].load_state_dict(saved)
        logger.info('loaded' + opt['model'])
else:
    model.load_state_dict(checkpoint['state_dict'])

params = model.parameters()

model.cuda(opt['id_gpu_model'])
logger.info(model)
from mytorch.implementation_utils import get_nb_parameters
logger.info('This model has %d parameters.' % (get_nb_parameters(model)))

#-------------------------------------------------------------------------------
# Criterion and optimizer
from criterion import FfpnCriterion, AutoregressiveCriterion
criterionConfig = configs['criterions']
criterionConfig['nb_scales'] = criterionConfig['nb_scales'][0] # they are all equal since train_single_level is set

single_frame_criterion = FfpnCriterion(criterionConfig)
logger.info('%s' % single_frame_criterion)

autoregressiveCritConfig = {'n_target_frames_ar' : opt['n_target_frames_ar']}
criterion = AutoregressiveCriterion(autoregressiveCritConfig, single_frame_criterion)
logger.info('%s' % criterion)

#-------------------------------------------------------------------------------
# Optimizer
optimConfig = configs['optim']
if optimConfig['optim_algo'] == 'nesterov-sgd':
    optimizer = optim.SGD(params, lr = optimConfig['learning_rate'], momentum = optimConfig['momentum'], nesterov = True)
    logger.info('Nesterov momentum SGD with learning rate :'+
        ' %6f, momentum : %2f, weight decay : %10f' % (
        optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['momentum'],
        optimizer.param_groups[0]['weight_decay']))

elif optimConfig['optim_algo'] == 'adam':
    optimizer = optim.Adam(params, lr = optimConfig['learning_rate'], betas = (
        optimConfig['beta1'], optimConfig['beta2']))
    logger.info('Adam with learning rate :'+
        ' %.6f, beta1 : %.1f, beta2 : %.3f' % (
        optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['betas'][0],
        optimizer.param_groups[0]['betas'][1]))
else:
    logger.error('Unknown optim_algo : %r' % optimConfig['optim_algo'])
    exit()

if not checkpoint is None:
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Need to do this for the tensors to be on the same GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(opt['id_gpu_model'])

# necessary for re-launching jobs
if not os.path.isfile(os.path.join(opt['save'], 'checkpoint.pth.tar')):
    from mytorch.checkpointing import save_checkpoint
    save_checkpoint({
        'epoch': checkpoint['epoch'] if not checkpoint is None else 0,
        'iter': checkpoint['iter'] if not checkpoint is None else 0,
        'opt_path': os.path.join(opt['logs'], 'params.pkl'),
        'state_dict': model.state_dict(),
        'best_prec1': checkpoint['best_prec1'] if not checkpoint is None else None,
        'optimizer' : optimizer.state_dict(),
        }, False, savedir = opt['save'])

#-------------------------------------------------------------------------------
# Train
from autoregressive_training import train_multiscale, val_multiscale, save
training_config = configs['train']
val_config = configs['eval']

# For checkpointing purposes
bestModelPerf = checkpoint['best_prec1'] if not checkpoint is None else None
if not bestModelPerf is None:
    logger.info('Starting from best model perf: %f' % bestModelPerf)
start_epoch = checkpoint['epoch'] if not checkpoint is None else 0
start_iter = checkpoint['iter'] if not checkpoint is None else 0

stats = {}
for epoch in range(start_epoch, opt['nEpoches']):
    train_multiscale(training_config, model, trainLoader, criterion, optimizer, epoch, stats, bestModelPerf, start_iter = start_iter)
    start_iter = 0 # important otherwise train will skip start_iter iterations at each epoch
    isBest, bestModelPerf = val_multiscale(val_config, model, valLoader, criterion, epoch, stats, bestModelPerf, optimizer)
    if opt['save'] is not None:
        save(model, optimizer, epoch, opt, stats, isBest, bestModelPerf)

os.system('touch %s' %os.path.join(opt['save'], 'HALT')) # To avoid relaunching a job if not necessary
