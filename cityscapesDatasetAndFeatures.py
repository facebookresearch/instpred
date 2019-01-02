# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# coding: utf-8
#-------------------------------------------------------------------------------
# Imports
from __future__ import print_function

from logging import getLogger
logger = getLogger()

from argparse import Namespace
import os
import time
import numpy as np
from PIL import Image
import cPickle as pickle
from collections import namedtuple, defaultdict
import json

from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace

from environment_configuration import PRECOMPUTED_FEATURES_DIR
from detectron.utils.timer import Timer
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file, merge_cfg_from_list
import detectron.utils.blob as blob_utils

import torch
from torchnet.dataset.dataset import Dataset
from detectron.datasets.json_dataset import JsonDataset

from mask_rcnn_model import initialize_model_from_cfg, MASK_RCNN_CONFIG
#-------------------------------------------------------------------------------
# To avoid using cv2 just for reading images...
def cv2imread_like(impath):
    im = np.array(Image.open(impath)) # open and convert to np array
    assert(im.ndim == 3)
    assert(im.shape[2] == 3)
    im = im[:, :, [2, 1, 0]] # convert to BGR
    im = im.copy() # just to recover normal striding
    return im

#-------------------------------------------------------------------------------
# Extracted and adapted from detectron
def im_extract_features(model, im, ffpn_levels, timers=None):
    """
    Extracts high level features listed in levels from model for the image im.
    """
    if timers is None:
        timers = defaultdict(Timer)
    timers['im_extract_features'].tic()

    # Get inputs to the caffe2 model
    blobs = {}
    blobs['data'], im_scale, blobs['im_info'] = \
        blob_utils.get_image_blob(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

    for k, v in blobs.items():
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))

    # Run the net forward
    workspace.RunNet(model.backbone.Proto().name)

    # Extract the features
    features = {}
    for feat in ffpn_levels:
        features[feat] = workspace.FetchBlob(core.ScopedName(feat))
    im_info = blobs['im_info']

    return features, im_info, im_scale, im.shape

#-------------------------------------------------------------------------------
# CityscapesDatasetAndFeatures definition
# Useful for get_classes
from cityscapesscripts.helpers.labels import Label, labels as cityscapes_labels
class CityscapesDatasetAndFeatures(Dataset):
    """`Cityscapes` dataset and precomputed features loader.
    See https://www.cityscapes-dataset.com/
    """

    def __init__(self, split, frame_ss, nSeq, features, savedir, size=None, loaded_model = None):
        super(CityscapesDatasetAndFeatures, self).__init__()

        self.split = split
        self.frame_ss = frame_ss
        self.nSeq = nSeq or 1
        self.features = features
        self.FPNfeatures = u'fpn_res5_2_sum' in self.features \
            or u'fpn_res4_5_sum' in self.features \
            or u'fpn_res3_3_sum' in self.features \
            or u'fpn_res2_2_sum' in self.features
        self.limitSize = size

        # Check which features have been precomputed and load them if they have been found
        logger.info('Searching for precomputed features...')
        self.precompute_features_dir = PRECOMPUTED_FEATURES_DIR

        self.potential_precomputed_feature_types = [u'fpn_res5_2_sum', u'fpn_res4_5_sum', u'fpn_res3_3_sum', u'fpn_res2_2_sum']
        self.load_precomputed = self.check_requested_features_that_could_be_precomputed_were()
        if self.load_precomputed:
            self.precomputed_features_index, self.precomputed_features = \
                self.load_requested_precomputed_features()

        self.requested_fpn_features = [] + \
            ([u'fpn_res5_2_sum'] if 'fpn_res5_2_sum' in self.features else []) + \
            ([u'fpn_res4_5_sum'] if 'fpn_res4_5_sum' in self.features else []) + \
            ([u'fpn_res3_3_sum'] if 'fpn_res3_3_sum' in self.features else []) + \
            ([u'fpn_res2_2_sum'] if 'fpn_res2_2_sum' in self.features else [])



        import detectron.utils.c2 as c2_utils
        c2_utils.import_detectron_ops()

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        dset = b'cityscapes_fine_instanceonly_seg_sequences_' + self.split
        if not cfg.is_immutable(): # just in case feature extractor has not been set up already
            # Preparing where to load data from and how using coco api

            args = Namespace(
                cfg_file = MASK_RCNN_CONFIG,
                wait = True,
                multi_gpu_testing = False,
                opts = ['OUTPUT_DIR', savedir]
            )

            merge_cfg_from_file(args.cfg_file)
            if args.opts is not None:
                merge_cfg_from_list(args.opts)
                assert_and_infer_cfg()

        assert os.path.exists(cfg.TEST.WEIGHTS), \
            'need path to pretrained instance segmentation model'
        assert not cfg.MODEL.RPN_ONLY, 'end to end model required'
        dataset = JsonDataset(dset)

        self.dataset = dataset
        self.im_list = self.dataset.get_roidb()
        # Preparing the model from which we obtain the features
        if not self.load_precomputed:
            if loaded_model is None:
                model = initialize_model_from_cfg()
                self.model = model
            else:
                self.model = loaded_model
        else:
            self.model = False

        # Store config for further use in running the Mask RCNN head
        logger.info('Cityscapes dataset, size : %d' % len(self))


    def check_requested_features_that_could_be_precomputed_were(self):
        requested_features_that_could_be_precomputed_were = True
        for feat in self.potential_precomputed_feature_types:
            if feat in self.features:
                feat_path = '__'.join((self.split, feat, 'nSeq%d'%self.nSeq, 'fr%d'%self.frame_ss, 'features.npy'))
                feat_path = os.path.join(self.precompute_features_dir, feat_path)
                if os.path.exists(feat_path):
                    logger.info('Found precomputed features for %s' % feat)
                else:
                    logger.info('Did not find precomputed features for %s at %s'
                        % (feat, feat_path))
                    requested_features_that_could_be_precomputed_were = False
                    break
        return requested_features_that_could_be_precomputed_were


    def load_requested_precomputed_features(self):
        precomputed_features_index, precomputed_features = {}, {}
        for feat in self.potential_precomputed_feature_types:
            if feat in self.features:
                # First load the ids
                ids_path = '__'.join((self.split, feat, 'nSeq%d'%self.nSeq, 'fr%d'%self.frame_ss, 'ids.pkl'))
                ids_path = os.path.join(self.precompute_features_dir, ids_path)
                assert os.path.exists(ids_path)
                with open(ids_path, 'r') as f:
                    precomputed_features_index[feat] = pickle.load(f)
                    precomputed_features_index[feat] = precomputed_features_index[feat]['sequence_ids']
                # Then load the features
                feat_path = '__'.join((self.split, feat, 'nSeq%d'%self.nSeq, 'fr%d'%self.frame_ss, 'features.npy'))
                feat_path = os.path.join(self.precompute_features_dir, feat_path)
                assert os.path.exists(feat_path)

                logger.info('Loading precomputed features for %s' % feat)
                precomputed_features[feat] = np.load(feat_path)

        return precomputed_features_index, precomputed_features


    def _get_IDs_for_sequence(self, annID):
        anndir, annfile = os.path.split(annID)
        city, vidID, frID, typeID = annfile.split('_')
        frIDn = int(frID)
        fr_st = frIDn - self.frame_ss * (self.nSeq-1)
        seqIDs = [ os.path.join(anndir, '%s_%s_%06d_%s' %(
            city, vidID, fr_st + self.frame_ss * fr, typeID))
            for fr in range(self.nSeq)]
        # check that last frame is annotated frame
        assert(os.path.split(seqIDs[-1])[1] == annfile)
        return seqIDs



    def __getitem__(self, index, gpu_id=0):
        sample = {}
        annID = self.im_list[index]['image']
        seqIDs = self._get_IDs_for_sequence(annID)

        gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
        name_scope = 'gpu_{}'.format(gpu_id)

        if u'RGBs' in self.features or u'BGRs' in self.features:
            seq = self._add_rgb_bgr_features(sample, seqIDs)
        if self.FPNfeatures:
            # First check if requested features have been precomputed and loaded
            if self.load_precomputed:
                for feat in self.precomputed_features.keys():
                    # Check that we are loading the right features
                    assert self.precomputed_features_index[feat][index] == annID
                    # Get them and put them in the expected shape
                    sample[feat] = self.precomputed_features[feat][index]
                    sample[feat] = torch.from_numpy(sample[feat]).float()
                    sz = sample[feat].size()
                    ch, h, w = sz[0]/self.nSeq, sz[1], sz[2]
            else: # else compute them
                if not u'BGRs' in sample:
                    seq = self._add_rgb_bgr_features(sample, seqIDs)
                for t in range(seq.shape[0]):
                    with core.NameScope(name_scope):
                        with core.DeviceScope(gpu_dev):
                            features, im_info, im_scales, im_shape = im_extract_features(self.model, seq[t], self.requested_fpn_features)
                            for k, v in features.items():
                                if k in self.features:
                                    if t==0:
                                        sample[k] = [[] for f in range(self.nSeq)]
                                    sample[k][t] = torch.from_numpy(v)
                                    sample[k][t] = sample[k][t]

                for k in features:
                    if k in self.features:
                        sample[k] = torch.cat(sample[k][:])


        sample[u'seqIDs'] = seqIDs
        sample[u'annID'] = annID

        return sample

    def __len__(self):
        return len(self.im_list)

    def _add_rgb_bgr_features(self, sample, seqIDs):
        seq = [np.expand_dims(cv2imread_like(imgID), 0) for imgID in seqIDs] #-- tested to be the same...
        seq = np.concatenate(seq, axis = 0)
        sample[u'BGRs'] = torch.from_numpy(seq)
        sample[u'BGRs'] = sample[u'BGRs'].permute(0, 3, 1, 2)
        sample[u'RGBs'] = torch.cat((
            sample[u'BGRs'][:,2:,:,:],
            sample[u'BGRs'][:,1:2,:,:],
            sample[u'BGRs'][:,0:1,:,:]),1)
        return seq

    def get_classes(self, task=None):
        self.cityscapes_labels = cityscapes_labels
        if task is None:
            classes = [label.name for label in cityscapes_label]
            return self.cityscapes_labels, classes
        elif task == 'semantic_segmentation' :
            return self.map_to_trainIds_for_semantic_segmentation()
        elif task == 'instance_segmentation' :
            return self.map_to_trainIds_for_instance_segmentation()
        else:
            logger.error('Unkown specified task : %r' % task)
            exit()


    def map_to_trainIds_for_semantic_segmentation(self):
        rlabels = []
        n_ignored = 0
        labels = self.cityscapes_labels
        classes = []
        for label in labels :
            if label.ignoreInEval :
                n_ignored +=1
        n_train_classes = len(cityscapes_labels) - n_ignored
        logger.info('Number of classes to train on : %d + one untagged class placed last' % n_train_classes)
        logger.info('Mapping : ')
        for label in labels:
            if label.ignoreInEval:
                rlabel_trainId = n_train_classes
            else:
                classes.append(label.name)
                rlabel_trainId = label.trainId
            rlabel = Label(label.name, label.id, rlabel_trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval, label.color)
            rlabels.append(rlabel)
            logger.info('%s \t %d' %(rlabel.name, rlabel.trainId))
        classes.append('unlabeled')
        return rlabels, classes


    def map_to_trainIds_for_instance_segmentation(self):
        rlabels = []
        n_ignored, n_background = 0, 0
        labels = self.cityscapes_labels
        classes = ['background']
        for label in labels :
            if label.ignoreInEval :
                n_ignored +=1
            elif not label.hasInstances:
                n_background += 1
        # We group all non hasInstance classes to one background class
        n_train_classes = len(cityscapes_labels) - n_ignored - n_background + 1
        logger.info('One background class placed first + number of classes of instances to train on : %d + one untagged class placed last' % (n_train_classes-1))
        logger.info('Mapping : ')
        counter = 1
        for label in labels:
            if label.ignoreInEval:
                rlabel_trainId = n_train_classes
            elif not label.hasInstances:
                rlabel_trainId = 0
            else:
                classes.append(label.name)
                rlabel_trainId = counter
                counter += 1
            rlabel = Label(label.name, label.id, rlabel_trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval, label.color)
            rlabels.append(rlabel)
            logger.info('%s \t %d' %(rlabel.name, rlabel.trainId))
        classes.append('unlabeled')
        return rlabels, classes
