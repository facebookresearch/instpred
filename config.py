# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import pprint
import os
import pickle
import copy
import torch

#-------------------------------------------------------------------------------
# Main function processing input arguments to turn them into configurations
class futureInstancePredictionInputParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description = 'Future features prediction for instance segmentation')
        # -- Experiment related
        self.parser.add_argument('-nI', '--n_input_frames', default = 4, type = int, help = 'number of input frames')
        self.parser.add_argument('-nT', '--n_target_frames', default = 1, type = int, help = 'number of target frames')
        self.parser.add_argument('-nTR', '--n_target_frames_ar', default = None, type = int, help = 'number of target frames when model is evaluated or fine-tuned in autoregressive fashion')
        self.parser.add_argument('-pl', '--FfpnLevels', default = 4, type = int, help = 'number of levels in the feature pyramids extracted by Mask R-CNN. NB: leave this to 4 when training a single level of the pyramid.')
        # -- Data related
        self.parser.add_argument('-nf', '--nb_features', default = 256, type = int, help = 'number of feature maps per input and per target')
        self.parser.add_argument('-fr', '--frame_subsampling', default = 3, type = int, help = 'temporal subsampling rate of the sequences')
        self.parser.add_argument('-ts', '--test_set', default = False, type = bool, help = 'if True, compute results for test set instead of val set')

        # -- Model related
        self.parser.add_argument('-m', '--model', default = None, type = str, help = 'path to previously trained model to initialize parameters with')
        self.parser.add_argument('-fsh', '--f2f_sharing', default = False, type = bool, help = 'if True, shares weights of the f2f subnetwork accross levels of the future feature pyramid network')
        self.parser.add_argument('-arch', '--architecture', default = 'parallel', type = str, help = 'name of the module to load for the ffpn model\'s architecture - possible values : \'parallel\'')
        self.parser.add_argument('-ms', '--model_size', default = 8, type = int, help = 'parameter by which the capacity of each f2f model is linearly augmented')
        self.parser.add_argument('-archf2fi', '--architecture_f2fi', default = 'multiscale', type = str, help = 'name of the F2F architecture to be used. Possible values : \'multiscale\'')
        self.parser.add_argument('-sc', '--nb_scales', default = None, type = str, help = 'specify the number of scales here for each subnetwork, ex:1,1,1,3. For training single level, specify only the scale of the trained level.')
        self.parser.add_argument('-lhlf', '--loss_features', default = 'mse', type = str, help = 'specifies components to add to the loss applied on the high level features. Eg : l1_gdl or mse - should separate components with _')

        # -- Optimization related
        self.parser.add_argument('-optalg', '--optim_algo', default = 'nesterov-sgd', type = str, help = 'optimization algorithm to use, choose from : \'nesterov-sgd\', \'adam\'')
        self.parser.add_argument('-lr', '--learning_rate', default = 0.0005, type = float, help = 'learning rate of the optimization algorithm')
        self.parser.add_argument('-mom', '--momentum', default = 0.9, type = float, help = 'momentum of the SGD optimization algorithm')
        self.parser.add_argument('-b1', '--beta1', default = 0.9, type = float, help = 'beta1 of the Adam optimization algorithm')
        self.parser.add_argument('-b2', '--beta2', default = 0.999, type = float, help = 'beta2 of the Adam optimization algorithm')
        self.parser.add_argument('-trit', '--ntrainIt', default = 2975, type = int, help = 'number of training iterations per epoch')
        self.parser.add_argument('-trbs', '--trainbatchsize', default = 1, type = int, help = 'number of samples per batch for training iterations')
        self.parser.add_argument('-nEp', '--nEpoches', default = 500, type = int, help = 'number of epochs for training')
        self.parser.add_argument('-tsl', '--train_single_level', default = None, type = str, help = 'useful to specify to train a single level of the featyre pyramid. Eg, fpn_res5_2_sum, fpn_res4_5_sum, fpn_res3_3_sum, fpn_res2_2_sum.')
        # -- Validation related
        self.parser.add_argument('-vlit', '--nvalIt', default = 500, type = int, help = 'number of validation iterations per epoch')
        self.parser.add_argument('-vlbs', '--valbatchsize', default = 1, type = int, help = 'number of samples per batch for validation iterations')
        # -- Printing and logging related
        self.parser.add_argument('-s', '--save', default = 'scratch', type = str, help = 'relative path to root dirs where all results should be saved, in results and logs directories as adequate')
        self.parser.add_argument('-nEps', '--nEpocheSave', default = 5, type = int, help = 'number of epochs between each save')
        self.parser.add_argument('-rst', '--create_subdir', default = 'False', type = str, help = 'if True, restarts the experiment from scratch in a sub folder of the given saving root directory ; otherwise, considers the saving directory as the root')
        self.parser.add_argument('-disp', '--display', default = False, type = bool, help = 'if True, displays certain plots and figures in visdom')
        self.parser.add_argument('-dispthr', '--display_thresh', default = 0.5, type = float, help = 'if instance results are to be displayed, threshold on the confidence for displaying them')

        # -- Running mode related
        self.parser.add_argument('-dbg', '--debug', default = False, type = bool, help = 'if True, launches in debug mode potentially with fake precomputed features')
        self.parser.add_argument('-ngpud', '--num_gpus_data', default = 1, type = int, help = 'number of GPUs to use for extracting the input and target features')
        self.parser.add_argument('-idgpum', '--id_gpu_model', default = 0, type = int, help = 'gpu id of the gpu to run the feature prediction model on')
        self.parser.add_argument('-r', '--resume', default = None, type = str, help = 'allows to resume a run. Path to the dir containing the checkpoint of a model from which we want to resume the run. Assumes the run was writing heavy results in this directory and that logs and params.pkl were being written in the corresponding logs directory.')
        # -- Precomputing features
        self.parser.add_argument('-pc', '--precompute_features', default = None, type = str, help = 'True in case precomputing features is necessary, set here which type')

        from environment_configuration import ROOT_SAVE_HEAVY_RESULTS, ROOT_LOGS
        # root directory containing all heavy results of all experiments ran with this script
        self.root_save = ROOT_SAVE_HEAVY_RESULTS
        # root directory containing all light results of all experiments ran with this script
        self.root_logs = ROOT_LOGS


    def parse_inputs_and_infer_config(self):
        defopt = self.parser.parse_args()
        # I don't want to deal with a namespace
        opt = {}
        for arg in vars(defopt):
            opt[arg] = getattr(defopt, arg)

        if opt['resume'] is None:
            print('Global configuration :')
            pprint.pprint(opt)

            # Infer from input arguments certain configuration values
            # Saving directories
            if opt['save']:
                rel_save_path = opt['save']
                # path to root dir where all heavy results (model weights, quantitative results, etc) should be saved
                opt['save'] = os.path.join(self.root_save , rel_save_path)
                # path to root dir where all lights results (logs, quantitative results, etc) should be saved
                opt['logs'] = os.path.join(self.root_logs , rel_save_path)

            if opt['save']:
                if opt['create_subdir'] == 'True':
                    from mytorch.implementation_utils import get_dump_directory
                    # Create a new directory with a random name inside opt['save']
                    from environment_configuration import CLUSTER_JOB_ID # note: can be None
                    opt['save'], opt['exp_name'] = get_dump_directory(opt['save'], job_id=CLUSTER_JOB_ID)
                    opt['logs'] = os.path.join(opt['logs'] , opt['exp_name'])
                else:
                    assert opt['create_subdir'] == 'False', \
                        'bad value for create_subdir: %r (should be True/False)' % opt['create_subdir']
                    opt['exp_name'] = ''

            if opt['save'] and not os.path.exists(opt['logs']):
                os.makedirs(opt['logs'])
            if opt['save'] and not os.path.exists(opt['save']):
                os.makedirs(opt['save'])

            print('Saving all heavy results of this run to : %s. \nHeavy results include : model weights, qualitative results, etc.' %
                opt['save'])
            print('Saving all light results of this run to : %s. \nLight results include : logs, quantitative results, etc.' %
                opt['logs'])

            # Adequately add the names of the features corresponding to the provided FfpnLevels
            if opt['train_single_level'] is None:
                opt['features'] = [u'fpn_res5_2_sum']
                if opt['FfpnLevels'] >= 2:
                    opt['features'] = opt['features'] + [u'fpn_res4_5_sum']
                if opt['FfpnLevels'] >= 3:
                    opt['features'] = opt['features'] + [u'fpn_res3_3_sum']
                if opt['FfpnLevels'] == 4:
                    opt['features'] = opt['features'] + [u'fpn_res2_2_sum']
            else:
                opt['features'] = [unicode(opt['train_single_level'], 'utf-8')]


            def check_no_unkown_loss(losses_options, possible_losses):
                losses = copy.copy(losses_options)
                for l in possible_losses:
                    if l in losses: losses.remove(l)
                assert len(losses) == 0, 'Unknown losses %r' % losses

            opt['loss_features'] = opt['loss_features'].split('_')
            check_no_unkown_loss(opt['loss_features'], ['mse', 'gdl', 'l1'])

            assert opt['id_gpu_model'] == opt['num_gpus_data'] \
                or opt['id_gpu_model'] == opt['num_gpus_data']-1

            assert opt['num_gpus_data'] == 1 or\
            opt['num_gpus_data'] == opt['n_input_frames'] + opt['n_target_frames'],\
            'multi gpu can only be using one gpu per image to produce feats on.'

            if opt['architecture_f2fi'] == 'multiscale':
                if opt['precompute_features'] is None:
                    assert not opt['nb_scales'] is None
                    scales = opt['nb_scales'].split(',')
                    if len(opt['nb_scales']) == 1: # same nb_scale for each level or train_single_level
                        opt['nb_scales'] = [int(scales[0]) for _ in range(opt['FfpnLevels'])]
                    else:
                        opt['nb_scales'] = [int(sc) for sc in scales]
                        assert len(opt['nb_scales'])== opt['FfpnLevels']

            return opt
        else: # if opt['resume']
            checkpoint_path = os.path.join(self.root_save, opt['resume'],'checkpoint.pth.tar')
            assert os.path.isfile(checkpoint_path),\
                "=> no checkpoint found at %s" % (checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            with open(checkpoint['opt_path'], 'r') as f: opt = pickle.load(f)
            print("=> loaded checkpoint '%s' (epoch %d)" % (checkpoint_path, checkpoint['epoch']))

            return opt, checkpoint



    def create_individual_configs(self, opt):
        configs = {}
        configs['trainset'] = {
            'split'                     : 'train',
            'frame_ss'                  : opt['frame_subsampling'],
            'n_input_frames'            : opt['n_input_frames'],
            'n_target_frames'           : opt['n_target_frames_ar'] or opt['n_target_frames'],
            'n_channels_per_input'      : opt['nb_features'],
            'n_channels_per_target'     : opt['nb_features'],
            'nIt'                       : opt['ntrainIt'],
            'features'                  : opt['features'],
            'batchsize'                 : opt['trainbatchsize'],
            'debug'                     : opt['debug'],
            'save'                      : opt['save'],
            'multi_gpu'                 : opt['num_gpus_data'] > 1,
            'train_single_level'        : opt['train_single_level'],
            'nb_scales'                 : opt['nb_scales'],
        }
        configs['valset'] = {
            'split'                     : 'val',
            'frame_ss'                  : opt['frame_subsampling'],
            'n_input_frames'            : opt['n_input_frames'],
            'n_target_frames'           : opt['n_target_frames_ar'] or opt['n_target_frames'],
            'n_channels_per_input'      : opt['nb_features'],
            'n_channels_per_target'     : opt['nb_features'],
            'nIt'                       : opt['nvalIt'],
            'features'                  : opt['features'],
            'batchsize'                 : opt['valbatchsize'],
            'debug'                     : opt['debug'],
            'save'                      : opt['save'],
            'multi_gpu'                 : opt['num_gpus_data'] > 1,
            'train_single_level'        : opt['train_single_level'],
            'nb_scales'                 : opt['nb_scales'],
            'test_set'                  : opt['test_set'],
        }
        configs['models'] = {
            'model_size'                : opt['model_size'],
            'n_channels_per_input'      : opt['nb_features'],
            'n_channels_per_target'     : opt['nb_features'],
            'n_input_frames'            : opt['n_input_frames'],
            'n_target_frames'           : opt['n_target_frames'],
            'FfpnLevels'                : opt['FfpnLevels'],
            'model'                     : opt['model'],
            'f2f_sharing'               : opt['f2f_sharing'],
            'architecture_f2fi'          : opt['architecture_f2fi'],
            # 'gpu_id'                    : opt['id_gpu_model'],
            'nb_scales'                 : opt['nb_scales'],
        }
        configs['criterions'] = {
            'FfpnLevels'                : opt['FfpnLevels'],
            'loss_features'             : opt['loss_features'],
            'gpu_id'                    : opt['id_gpu_model'],
            'train_single_level'        : opt['train_single_level'],
            'n_target_frames'           : opt['n_target_frames_ar'] or opt['n_target_frames'],
            'nb_scales'                 : opt['nb_scales'],
        }
        configs['meters'] = {
            'n_target_frames'           : opt['n_target_frames_ar'] or opt['n_target_frames']
        }
        configs['optim'] = {
            'optim_algo'                : opt['optim_algo'],
            'learning_rate'             : opt['learning_rate'],
            'momentum'                  : opt['momentum'],
            'beta1'                     : opt['beta1'],
            'beta2'                     : opt['beta2'],
        }
        configs['train'] = {
            'FfpnLevels'                : opt['FfpnLevels'],
            'save'                      : opt['save'],
            'nEpocheSave'               : opt['nEpocheSave'],
            'n_input_frames'            : opt['n_input_frames'],
            'n_target_frames'           : opt['n_target_frames_ar'] or opt['n_target_frames'],
            'it'                        : opt['ntrainIt'],
            'gpu_id'                    : opt['id_gpu_model'],
            'loss_features'             : opt['loss_features'],
            'train_single_level'        : opt['train_single_level'],
            'nb_features'               : opt['nb_features'],
            'nb_scales'                 : opt['nb_scales'],
            'logs'                      : opt['logs'],
        }
        configs['eval'] = {
            'save'                      : opt['save'],
            'display'                   : opt['display'],
            'display_thresh'            : opt['display_thresh'],
            'FfpnLevels'                : opt['FfpnLevels'],
            'batchsize'                 : opt['valbatchsize'],
            'n_input_frames'            : opt['n_input_frames'],
            'n_target_frames'           : opt['n_target_frames_ar'] or opt['n_target_frames'],
            'nb_features'               : opt['nb_features'],
            'it'                        : opt['nvalIt'],
            'gpu_id'                    : opt['id_gpu_model'],
            'loss_features'             : opt['loss_features'],
            'train_single_level'        : opt['train_single_level'],
            'nb_scales'                 : opt['nb_scales'],
            'logs'                      : opt['logs'],
            'test_set'                  : opt['test_set']
        }

        return configs

def make_config():
    fipParser = futureInstancePredictionInputParser()
    opt = fipParser.parse_inputs_and_infer_config()
    # If resuming a job, then checkpoint is also returned
    if len(opt) == 2:
        opt, checkpoint = opt
    else:
        checkpoint = None
    configs = fipParser.create_individual_configs(opt)
    if opt['logs']:
        pickle.dump(opt, open(os.path.join(opt['logs'], 'params.pkl'), 'wb'))
    return opt, configs, checkpoint
