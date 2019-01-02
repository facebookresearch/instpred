# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import copy
from argparse import Namespace
import numpy as np
import yaml

from logging import getLogger
logger = getLogger()

from caffe2.proto import caffe2_pb2
from caffe2.python import core as caffe2_core
from caffe2.python import workspace

from detectron.utils.timer import Timer
from detectron.core.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg)
from detectron.utils.io import save_object
from detectron.datasets.json_dataset import JsonDataset
import detectron.utils.boxes as box_utils
from detectron.core.test import (
    im_detect_mask, segm_results, box_results_with_nms_and_limit)
from detectron.core.test_engine import empty_results, extend_results
from detectron.datasets import task_evaluation

import torch

from mask_rcnn_model import initialize_model_from_cfg, MASK_RCNN_CONFIG
# ------------------------------------------------------------------------------
def im_detect_bbox_given_features(model, features, im_info, im_scales, im_shape):
    """Bounding box object detection for provided features with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        features (dictionary of ndarray): high level features from which to run detection

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    # Function simply adapted to use the input features and forward through the
    # head rather than input images through the entire network.

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True)
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    blobs = copy.copy(features)
    blobs['im_info'] = im_info
    for k, v in blobs.items():
        workspace.FeedBlob(caffe2_core.ScopedName(k), v)
    workspace.RunNet(model.faster_rnn_head.Proto().name)

    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        assert len(im_scales) == 1, \
            'Only single-image / single-scale batch implemented'
        rois = workspace.FetchBlob(caffe2_core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    # use softmax estimated probabilities
    scores = workspace.FetchBlob(caffe2_core.ScopedName('cls_prob')).squeeze()

    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(caffe2_core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4:]
        pred_boxes = box_utils.bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scales



def im_detect_all_given_features(model, subsampler, features, im_info, im_scales, im_shape, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    scores, boxes, im_scales = im_detect_bbox_given_features(model, features, im_info, im_scales, im_shape)
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        timers['im_detect_mask'].tic()
        masks = im_detect_mask(model, im_scales, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(
            cls_boxes, masks, boxes, im_shape[0], im_shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    return cls_boxes, cls_segms

# ------------------------------------------------------------------------------
# MaskRcnnHead definition
class MaskRcnnHead(object):
    def __init__(self, config, im_list, model=None, gpu_id=0): # im_list passed from cityscapes dataset
        self.nb_features = config['nb_features']
        self.split = config['split']
        self.im_list = im_list

        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        if not cfg.is_immutable(): # just in case feature extractor has not been set up already
            dset = b'cityscapes_fine_instanceonly_seg_' + self.split
            args = Namespace(
                cfg_file = MASK_RCNN_CONFIG,
                wait = True,
                multi_gpu_testing = False,
                range = None, #[0, 3],
                opts = ['OUTPUT_DIR', config['save']]
            )

            merge_cfg_from_file(args.cfg_file)
            if args.opts is not None:
                merge_cfg_from_list(args.opts)
                assert_and_infer_cfg()

        if model is None or model == False:
            self.model = initialize_model_from_cfg(instanciate_head_also=True)
        else:
            self.model = model

        gpu_dev = caffe2_core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
        name_scope = 'gpu_{}'.format(gpu_id)
        # Subsampler - originally inside the FPN network. But we don't want to predict the subsampled features.
        # Instead, we want to predict the features, and then use the same subsampling operator to obtain the subsampled features
        with caffe2_core.NameScope(name_scope):
            with caffe2_core.DeviceScope(gpu_dev):
                self.subsampler = caffe2_core.CreateOperator(
                    "MaxPool", # operator
                    ["predicted_fpn_res5_2_sum"], #input blobs
                    ["predicted_fpn_res5_2_sum_subsampled_2x"], #output blobs
                    kernel=1, pad=0, stride=2,
                    deterministic = 1
                )

        self.timers = {k: Timer() for k in
          ['im_detect_bbox', 'im_detect_mask',
           'misc_bbox', 'misc_mask', 'im_forward_backbone']}

        # For evaluation with respect to the dataset's gt, we save the prediction of the annotated frame for each sequence
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.num_images = len(self.im_list)
        self.all_boxes_ann_frame, self.all_segms_ann_frame, _ = empty_results(self.num_classes, self.num_images)
        self.id_sequences = []
        self.gpu_id = gpu_id

    def run(self, index, inputFeatures, accumulate = True, image_path = None):
        """
        index - index of the dataset entry
        inputFeatures - features input to the head
        accumulate - whether to save to predictions in self.all_... members
        image_path - path to the annotated image, to which the predictions correspond
        """
        timers = self.timers
        # Format the inputs to the mask rcnn head
        features = {}
        for k, v in inputFeatures.iteritems():
            assert v.dim() == 3, 'Batch mode not allowed'
            features[k] = np.expand_dims(v.data.cpu().numpy(), axis=0)

        gpu_dev = caffe2_core.DeviceOption(caffe2_pb2.CUDA, self.gpu_id)
        name_scope = 'gpu_{}'.format(self.gpu_id)

        # Clean the workspace to make damn sure that nothing comes from the
        # possible forwarding of target features, depending on the use of this
        # module
        parameters = [str(s) for s in self.model.params] + [ str(s) + '_momentum' for s in self.model.TrainableParams()]
        for b in workspace.Blobs():
            if not b in parameters:
                workspace.FeedBlob(b, np.array([]))

        # Produce the top level of the pyramid of features
        with caffe2_core.NameScope(name_scope):
            with caffe2_core.DeviceScope(gpu_dev):
                workspace.FeedBlob(caffe2_core.ScopedName("predicted_fpn_res5_2_sum"), features['fpn_res5_2_sum'])
                workspace.RunOperatorOnce(self.subsampler)
                features[u'fpn_res5_2_sum_subsampled_2x'] = workspace.FetchBlob(caffe2_core.ScopedName("predicted_fpn_res5_2_sum_subsampled_2x"))


        # Forward the rest of the features in the head of the model
        im_info = np.array([[1024., 2048., 1.]], dtype = np.float32)
        im_scales = np.array([1.])
        im_shape = (1024, 2048, 3)
        with caffe2_core.NameScope(name_scope):
            with caffe2_core.DeviceScope(gpu_dev):
                cls_boxes_i, cls_segms_i = im_detect_all_given_features(
                    self.model, self.subsampler, features, im_info, im_scales, im_shape, timers)

        # If required, store the results in the class's members
        if accumulate:
            extend_results(index, self.all_boxes_ann_frame, cls_boxes_i)
        if cls_segms_i is not None and accumulate:
            extend_results(index, self.all_segms_ann_frame, cls_segms_i)

        if image_path is not None and accumulate:
            self.id_sequences.append(image_path)

        if index % 10 == 0:
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            det_time = (timers['im_detect_bbox'].average_time +
                        timers['im_detect_mask'].average_time )
            misc_time = (timers['misc_bbox'].average_time +
                         timers['misc_mask'].average_time
                          )
            print(
                ('im_detect: '
                 '{:d}/{:d} {:.3f}s + {:.3f}s => avg total time: {:.3f}s').format(
                    index, self.num_images,
                    det_time, misc_time, ave_total_time))

        return cls_boxes_i, cls_segms_i

    def save_annotated_frame_results(self, config, output_dir = './quantitative_eval/', st=None, en=None):
        det_filename = 'detections'
        det_filename += '_%d' % st if not st is None else ''
        det_filename += '_%d' % en if not en is None else ''
        det_filename +='.pkl'
        det_file = os.path.join(output_dir, det_filename)
        cfg_yaml = yaml.dump(cfg)
        save_object(
            dict(all_boxes=self.all_boxes_ann_frame,
                 all_segms=self.all_segms_ann_frame,
                 cfg=cfg_yaml, all_ids = self.id_sequences, config=config),
            det_file
        )
        self.all_boxes_ann_frame, self.all_segms_ann_frame, _ = empty_results(self.num_classes, self.num_images)
        self.id_sequences = []
        logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
