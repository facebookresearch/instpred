# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.autograd import Variable

import sys
import numpy as np
import os
import time
import json
import signal

from logging import getLogger
logger = getLogger()

from mytorch.checkpointing import save_checkpoint

#-------------------------------------------------------------------------------
# To handle signals coming from slurm that job is going to be pre-empted
SIGNAL_RECEIVED = False
HALT_filename = 'HALT'
CHECKPOINT_filename = 'checkpoint.pth.tar'
MAIN_PID = os.getpid()
RESUME_PATH = ''
NUM_GPUS = None

def signalHandler(a, b):
    global SIGNAL_RECEIVED
    logger.info('Signal received %r %r' %( a, time.time()))
    SIGNAL_RECEIVED = True

    ''' If HALT file exists, which means the job is done, exit peacefully.
    Just an additional security, in practice, should not be used.
    '''
    if os.path.isfile(os.path.join(RESUME_PATH, HALT_filename)):
        logger.info('Job is done, exiting')
        exit(0)


    ''' Submit a new job to resume from checkpoint.
    => implement whatever makes sense in your environment below.'''
    # ==========================================================================

    if os.path.isfile(os.path.join(RESUME_PATH, CHECKPOINT_filename)):
        logger.info('time is up, back to queue')
        SCRIPT_PATH = sys.argv[0]
        # USERTODO : Implement command to launch a new job resuming this one
        # command =
    # ==========================================================================

        logger.info('Executing %s' % command)
        if os.system(command):
            raise RuntimeError('launch failed')
        time.sleep(3)
        logger.info('New job submitted to the queue, saving checkpoint')
    return

''' Install signal handler
'''
signal.signal(signal.SIGUSR1, signalHandler)

feat_ind = {
    'fpn_res5_2_sum': 0,
    'fpn_res4_5_sum': 1,
    'fpn_res3_3_sum': 2,
    'fpn_res2_2_sum': 3
}

#-------------------------------------------------------------------------------
# Elementary functions
def prepareMultiscaleForForwardOnGpu(*tensors, **kwargs):
    assert 'nb_scales' in kwargs.keys()
    if 'gpu_id' not in kwargs.keys():
        kwargs['gpu_id'] = 0
    rslt = []
    def prepareTensor(tensor, gpu_id):
        return Variable(tensor.cuda(gpu_id), requires_grad = False)
    for ind, tens in enumerate(tensors):
        rslt.append({})
        assert isinstance(tens, dict), \
            'No other cases considered for multiscale for now.'

        for k, v in tens.items():
            rslt[ind][k] = []
            for sc in range(kwargs['nb_scales'][feat_ind[k]]):
                rslt[ind][k].append(prepareTensor(v[sc], gpu_id = kwargs['gpu_id']))
    return rslt


def resetTrainStatsSingleFrameMultiscale(opt):
    rstats = {}
    levelNames = ['fpn_res5_2_sum', 'fpn_res4_5_sum', 'fpn_res3_3_sum', 'fpn_res2_2_sum']
    for l in range(opt['FfpnLevels']):
        lev = levelNames[l]
        for loss_type in opt['loss_features']:
            for sc in range(opt['nb_scales'][l]):
                rstats['train_%s-%s-%s' % (lev, loss_type, sc)] = []

    return rstats


def resetValStatsSingleFrameMultiscale(opt):
    rstats = {}
    levelNames = ['fpn_res5_2_sum', 'fpn_res4_5_sum', 'fpn_res3_3_sum', 'fpn_res2_2_sum']
    for l in range(opt['FfpnLevels']):
        lev = levelNames[l]
        for loss_type in opt['loss_features']:
            for sc in range(opt['nb_scales'][l]):
                rstats['val_%s-%s-%s' % (lev, loss_type, sc)] = []

    return rstats


def resetTrainProgressMultiscale(opt, train_loader, stats):
    runningTrainLoss = 0.0
    train_loader.reset(reshuffle = True)
    # Stats
    for t in range(opt['n_target_frames']):
        stats['t+%d' % (t+1)] = {}
        stats['t+%d' % (t+1)].update(resetTrainStatsSingleFrameMultiscale(opt))
    stats['train_ae_loss_values'] = []
    return runningTrainLoss


def resetValProgressMultiscale(opt, val_loader, stats):
    totalValLoss = 0.0
    ctValIt = 0
    val_loader.reset()
    # Stats
    for t in range(opt['n_target_frames']):
        if not stats.has_key('t+%d' % (t+1)): stats['t+%d' % (t+1)] = {}
        stats['t+%d' % (t+1)].update(resetValStatsSingleFrameMultiscale(opt))
    stats['val_ae_loss_values'] = []
    return totalValLoss, ctValIt

def reshapeMultiscaleTargetsForCriterion(targets, nT, nb_feat, nb_scales):
    seq_targets = []
    for t in range(nT):
        rtargets = {}
        for k, v in targets.items():
            rtargets[k] = []
            for sc in range(nb_scales[feat_ind[k]]):
                assert v[sc].dim() == 4
                assert v[sc].size(1) == nT * nb_feat
                st, en = t * nb_feat, (t+1) * nb_feat
                rtargets[k].append(v[sc][:, st:en, :, :])
        seq_targets.append(rtargets)
    return seq_targets

def updateTrainProgress(opt, runningTrainLoss, lossdata, loss_terms, stats, i, rtl_period, epoch):
    stats['train_ae_loss_values'].append(lossdata)
    for kt, vt in enumerate(loss_terms):
        for ks, vs in vt.items() :
            stats['t+%d' % (kt+1)]['train_'+ks].append(vs)
    runningTrainLoss +=  lossdata
    if i % rtl_period == (rtl_period -1):
        avgRunningTrainLoss = runningTrainLoss / rtl_period
        logger.info('[%d, %5d] running train loss: %.3f' %
             (epoch + 1, i + 1, avgRunningTrainLoss))
        runningTrainLoss = 0.0

    return runningTrainLoss

def updateValProgress(totalValLoss, ctValIt, lossdata, loss_terms, stats, epoch, i, rtl_period):
    stats['val_ae_loss_values'].append(lossdata)
    for kt, vt in enumerate(loss_terms):
        for ks, vs in vt.items() :
            stats['t+%d' % (kt+1)]['val_'+ks].append(vs)
    totalValLoss +=  lossdata
    ctValIt += 1
    if i % rtl_period == (rtl_period -1):
        avgValLoss = totalValLoss / ctValIt
        logger.info('[%d, %5d] mean validation loss: %.3f' %
             (epoch + 1, i + 1, avgValLoss))
    return totalValLoss, ctValIt


def checkIsBest(totalValLoss, ctValIt, bestModelPerf=None):
    current_val = - totalValLoss/ctValIt
    sigma = 0.001
    logger.info('Current val : %.3f' % current_val)
    if bestModelPerf is None:
        bestModelPerf = current_val
        logger.info("Self bestModelPerf : %.3f" % bestModelPerf)
        return False, bestModelPerf
    else:
        if current_val > bestModelPerf + sigma:
            bestModelPerf = current_val
            logger.info("Self bestModelPerf : %.3f" % bestModelPerf)
            return True, bestModelPerf
        else:
            logger.info("Self bestModelPerf : %.3f" % bestModelPerf)
            return False, bestModelPerf


def format_variable_length_multiscale_sequence(outputs, ffpn_levels, nT, nb_scales):
    """ Only implemented in case single feature training..."""
    find_feature_by_dim = {
        32 : 'fpn_res5_2_sum', 64 : 'fpn_res4_5_sum',
        128 : 'fpn_res3_3_sum', 256 : 'fpn_res2_2_sum'}
    seq_outputs = []
    assert len(outputs) == nT * ffpn_levels
    current_frame = 0
    feat = None
    for f, out in enumerate(outputs):
        if len(seq_outputs) == current_frame: seq_outputs.append({})
        if feat is None: feat = find_feature_by_dim[out[-1].size(2)]
        assert len(out) == nb_scales[feat_ind[feat]]
        assert find_feature_by_dim[out[-1].size(2)] == feat
        seq_outputs[current_frame][feat] = out
        current_frame +=1
        if (f+1)%nT == 0:
            current_frame = 0
            feat = None
    return seq_outputs

#-------------------------------------------------------------------------------
# Main functions
def train_multiscale(opt, model, train_loader, criterion, optimizer, epoch, stats, best_prec1, start_iter = 0):
    global SIGNAL_RECEIVED
    from detectron.utils.timer import Timer
    t = Timer()
    model.train()

    runningTrainLoss = resetTrainProgressMultiscale(opt, train_loader, stats)
    rtl_period = max(5, int(len(train_loader)/1))
    logger.info('-------------------------- Training epoch #%d --------------------------' % (epoch+1))
    t.tic()
    # set the variables for signal_handler
    global RESUME_PATH, NUM_GPUS
    RESUME_PATH = opt['save']
    NUM_GPUS = opt['gpu_id'] + 1 # relies assumption that the model uses the last GPU

    for i, data in enumerate(train_loader):
        # Skip the iterations included in the checkpoint
        if i < start_iter: continue

        # Get and prepare data
        inputs, targets, _ = data
        inputs, targets = prepareMultiscaleForForwardOnGpu(inputs, targets, **{'gpu_id' : opt['gpu_id'], 'nb_scales': opt['nb_scales']})
        targets = reshapeMultiscaleTargetsForCriterion(targets, opt['n_target_frames'], opt['nb_features'], opt['nb_scales'])
        # Optimization
        optimizer.zero_grad()
        ffpnlevels = 1 if opt['train_single_level'] else opt['FfpnLevels']
        outputs = format_variable_length_multiscale_sequence(model(inputs), ffpnlevels, opt['n_target_frames'], opt['nb_scales'])
        loss, loss_terms = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update progress
        runningTrainLoss = updateTrainProgress(opt, runningTrainLoss, loss.item(), loss_terms, stats, i, rtl_period, epoch)

        if SIGNAL_RECEIVED:
            save_checkpoint({
                'epoch': epoch,
                'iter': i+1,
                'opt_path': os.path.join(opt['logs'], 'params.pkl'),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                }, False, savedir = opt['save'])
            logger.info('Saved checkpoint before exiting peacefully for job requeuing')
            exit(0)
        del loss, inputs, outputs, targets, loss_terms
        t.toc() ; t.tic()
        if i >= (opt['it']-1) : break
    print('Training iteration average duration : %f' % t.average_time)


def val_multiscale(opt, model, val_loader, criterion, epoch, stats, bestModelPerf, optimizer):
    global SIGNAL_RECEIVED
    from detectron.utils.timer import Timer
    t = Timer()
    model.eval()
    totalValLoss, ctValIt = resetValProgressMultiscale(opt, val_loader, stats)
    rtl_period = max(5, int(len(val_loader)/1))
    t.tic()
    coco_cityscapes_dataset = val_loader.data_source.dataset.dataset.dataset
    json_classes = coco_cityscapes_dataset.classes
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Get and prepare data
            inputs, targets, seqIDs = data
            inputs, targets = prepareMultiscaleForForwardOnGpu(inputs, targets, **{'gpu_id' : opt['gpu_id'], 'nb_scales': opt['nb_scales']})
            targets = reshapeMultiscaleTargetsForCriterion(targets, opt['n_target_frames'], opt['nb_features'], opt['nb_scales'])
            # Evaluation
            ffpnlevels = 1 if opt['train_single_level'] else opt['FfpnLevels']
            outputs = format_variable_length_multiscale_sequence(model(inputs), ffpnlevels, opt['n_target_frames'], opt['nb_scales'])
            loss, loss_terms = criterion(outputs, targets)
            # Update progress
            totalValLoss, ctValIt = updateValProgress(totalValLoss, ctValIt, loss.item(), loss_terms, stats, epoch, i, rtl_period)
            t.toc() ; t.tic()
            if SIGNAL_RECEIVED:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'iter': 0,
                    'opt_path': os.path.join(opt['logs'], 'params.pkl'),
                    'state_dict': model.state_dict(),
                    'best_prec1': bestModelPerf,
                    'optimizer' : optimizer.state_dict(),
                    }, False, savedir = opt['save'])
                logger.info('Saved checkpoint before exiting peacefully for job requeuing')
                exit(0)
            del loss, inputs, outputs, targets, loss_terms
            if i >= (opt['it']-1) : break
    logger.info('Validation iteration average duration : %f' % t.average_time)

    return checkIsBest(totalValLoss, ctValIt, bestModelPerf=bestModelPerf)


def save(model, optimizer, epoch, entireSetOptions, stats, isBestModel, bestModelPerf):
    nEs = entireSetOptions['nEpocheSave']
    logger.info('Saving results to %s' % entireSetOptions['save'])
    logger.info('Saving model to '+entireSetOptions['save'] + 'model_%dep.net' % (epoch+1))
    torch.save(model.state_dict(), entireSetOptions['save'] + 'model_%dep.net' % (epoch+1))
    save_checkpoint({
        'epoch': epoch + 1,
        'iter': 0,
        'opt_path': os.path.join(entireSetOptions['logs'], 'params.pkl'),
        'state_dict': model.state_dict(),
        'best_prec1': bestModelPerf,
        'optimizer' : optimizer.state_dict(),
        },
        isBestModel,
        savedir = entireSetOptions['save'])
    train_mean_ae_loss = np.mean(stats['train_ae_loss_values'])
    val_mean_ae_loss = np.mean(stats['val_ae_loss_values'])
    logger.info('Mean autoencoder loss throughout training epoch: %.5f' % train_mean_ae_loss)
    logger.info('Mean autoencoder loss of validation epoch: %.5f' % val_mean_ae_loss)

    logs = dict([('n_epoch', epoch+1)])
    for k, v in stats.items() :
        if isinstance(v, dict):
            for kv, vv in v.items():
                logs['_'.join((k, kv))] = np.mean(vv)
        else:
            logs[k] = np.mean(v)

    logger.info("__log__:%s" % json.dumps(logs))
