# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""This file contains adapted bits of model_builder to allow access to
model.backbone and model.faster_rnn_head, so that it can be run on the extracted
features."""
import sys
import copy
import os

from detectron.core.config import cfg
from detectron.modeling import model_builder
from detectron.modeling.detector import DetectionModelHelper
import detectron.modeling.fast_rcnn_heads as fast_rcnn_heads
import detectron.modeling.rpn_heads as rpn_heads
import detectron.modeling.optimizer as optim
import detectron.utils.net as nu
from detectron.utils.c2 import gauss_fill, const_fill, SuffixNet

from caffe2.python import workspace

config_name = 'configs/maskrcnn/coco_init_e2e_mask_rcnn_R-50-FPN_with_feature_extraction.yaml'
MASK_RCNN_CONFIG = os.path.join(os.path.dirname(__file__), config_name)

def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    cls_out = model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    bbox_out = model.FC(
        blob_in,
        'bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    return cls_out, bbox_out


def _add_fast_rcnn_head_for_feature_extraction(
    model, add_roi_box_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a Fast R-CNN head to the model."""
    blob_frcn, dim_frcn = add_roi_box_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    blob_cls, blob_bbox = add_fast_rcnn_outputs(model, blob_frcn, dim_frcn)
    if model.train:
        loss_gradients = fast_rcnn_heads.add_fast_rcnn_losses(model)
    else:
        loss_gradients = None
    return loss_gradients, blob_cls, blob_bbox

def build_generic_detection_model_for_feature_extraction(
        model, add_conv_body_func, add_roi_frcn_head_func,
        add_roi_mask_head_func=None, add_roi_keypoint_head_func=None,
        freeze_conv_body=False):
    def _single_gpu_build_func(model):
        """Builds the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model."""
        # For training we define one net that contains all ops
        # For inference, we split the graph into two nets: a standard fast r-cnn
        # net and a mask prediction net; the mask net is only applied to a
        # subset of high-scoring detections
        is_inference = not model.train

        head_loss_gradients = {
            'rpn': None,
            'box': None,
            'mask': None,
            'keypoints': None,
        }

        # Add the conv body
        blob_conv, dim_conv, spatial_scale_conv = add_conv_body_func(model)
        if freeze_conv_body:
            for b in blob_ref_to_list(blob_conv):
                model.StopGradient(b, b)
        backbone_net = copy.deepcopy(model.net.Proto())
        model.backbone = model.net.Clone('backbone')

        if cfg.RPN.RPN_ON:
            # Add the RPN head
            head_loss_gradients['rpn'] = rpn_heads.add_generic_rpn_outputs(
                model, blob_conv, dim_conv, spatial_scale_conv
            )

        if cfg.FPN.FPN_ON:
            # After adding the RPN head, restrict FPN blobs and scales to
            # those used in the RoI heads
            blob_conv, spatial_scale_conv = model_builder._narrow_to_fpn_roi_levels(
                blob_conv, spatial_scale_conv
            )

        assert not cfg.MODEL.RPN_ONLY, 'only mask rcnn end2end implemented and tested'
        # Add the Fast R-CNN head
        head_loss_gradients['box'], blob_cls, blob_bbox = _add_fast_rcnn_head_for_feature_extraction(
            model, add_roi_frcn_head_func, blob_conv, dim_conv, spatial_scale_conv
        )

        # Extract the head of faster rcnn net, store it as its own network,
        # then create a new network faster_rcnn_head
        model.faster_rnn_head, [blob_cls, blob_bbox] = SuffixNet(
            'faster_rcnn_head', model.net, len(backbone_net.op), [blob_cls, blob_bbox])

        if cfg.MODEL.MASK_ON:
            head_loss_gradients['mask'] = model_builder._add_roi_mask_head(
                model, add_roi_mask_head_func, blob_conv, dim_conv,
                spatial_scale_conv)

        # Add the keypoint branch
        assert not cfg.MODEL.KEYPOINTS_ON, 'keypoints not implemented, otherwise need to extend here'

        assert not model.train, 'not implemented, should add losses here'

    optim.build_data_parallel_model(model, _single_gpu_build_func)
    return model

def generalized_rcnn_with_feature_extraction(model):
    """NB: This model type was only tested for Mask R-CNN
    (end-to-end joint training).
    """
    return build_generic_detection_model_for_feature_extraction(
        model,
        model_builder.get_func(cfg.MODEL.CONV_BODY),
        model_builder.get_func(cfg.FAST_RCNN.ROI_BOX_HEAD),
        add_roi_mask_head_func=model_builder.get_func(cfg.MRCNN.ROI_MASK_HEAD),
        add_roi_keypoint_head_func=model_builder.get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD),
        freeze_conv_body=cfg.TRAIN.FREEZE_CONV_BODY
    )

def create(model_type_func, train=False, gpu_id=0):
    """Generic model creation function that dispatches to specific model
    building functions.

    By default, this function will generate a data parallel model configured to
    run on cfg.NUM_GPUS devices. However, you can restrict it to build a model
    targeted to a specific GPU by specifying gpu_id. This is used by
    optimizer.build_data_parallel_model() during test time.
    """
    model = DetectionModelHelper(
        name=model_type_func,
        train=train,
        num_classes=cfg.MODEL.NUM_CLASSES,
        init_params=train
    )
    model.only_build_forward_pass = False
    model.target_gpu_id = gpu_id
    assert cfg.MODEL.TYPE == 'generalized_rcnn_with_feature_extraction'
    return generalized_rcnn_with_feature_extraction(model)


def add_inference_inputs(model, instanciate_head_also):
    """Create network input blobs used for inference.
    Note: adapted from model_builder.add_inference_inputs"""

    def create_input_blobs_for_net(net_def):
        for op in net_def.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)

    create_input_blobs_for_net(model.net.Proto())
    if instanciate_head_also:
        create_input_blobs_for_net(model.faster_rnn_head.Proto())
        if cfg.MODEL.MASK_ON:
            create_input_blobs_for_net(model.mask_net.Proto())
        assert not cfg.MODEL.KEYPOINTS_ON, 'keypoints not implemented, otherwise need to extend here'



def initialize_model_from_cfg(multi_gpu = False, instanciate_head_also=False):
    model = create(cfg.MODEL.TYPE, train=False)
    nu.initialize_from_weights_file(
        model, cfg.TEST.WEIGHTS, broadcast=multi_gpu)
    add_inference_inputs(model, instanciate_head_also)
    workspace.CreateNet(model.backbone)
    if instanciate_head_also:
        workspace.CreateNet(model.faster_rnn_head)
        if cfg.MODEL.MASK_ON:
            workspace.CreateNet(model.mask_net)
        assert not cfg.MODEL.KEYPOINTS_ON, 'keypoints not implemented, otherwise need to extend here'
    return model
