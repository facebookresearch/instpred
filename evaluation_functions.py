# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.autograd import Variable
from logging import getLogger
logger = getLogger()
import sys
import os
import pycocotools.mask as mask_util
import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval
from MaskRcnnHeadEndToEnd import MaskRcnnHead
from autoregressive_training import (
    format_variable_length_multiscale_sequence,
    prepareMultiscaleForForwardOnGpu,
    reshapeMultiscaleTargetsForCriterion,
    resetValProgressMultiscale)

# ------------------------------------------------------------------------------
# Helper functions
def squeeze_feature_maps(features):
    for k, v in features.iteritems():
        features[k] = v.squeeze(0)
        assert features[k].dim() == 3, 'No batch mode allowed in this evaluation script'
    return features


def complete_with_zero_features(ffpn_levels, outputs, gpu_id = 0):
    if ffpn_levels == 4:
        return outputs
    else:
        sz = outputs[u'fpn_res5_2_sum'].size()
        outputs[u'fpn_res2_2_sum'] = Variable(torch.zeros(sz[0], sz[1], sz[2]*8, sz[3]*8).cuda(gpu_id), volatile = True)
        if ffpn_levels == 3:
            return outputs
        else:
            outputs[u'fpn_res3_3_sum'] = Variable(torch.zeros(sz[0], sz[1], sz[2]*4, sz[3]*4).cuda(gpu_id), volatile = True)
            if ffpn_levels == 2:
                return outputs
            else:
                outputs[u'fpn_res4_5_sum'] = Variable(torch.zeros(sz[0], sz[1], sz[2]*2, sz[3]*2).cuda(gpu_id), volatile = True)
                return outputs

def get_instance_segmentation_results(config, i, outputs, mask_rcnn_head, json_classes, accumulate = True, image_path = None):
    assert config['batchsize'] == 1 # run_head is only configured for single inputs, not batches
    outputs = complete_with_zero_features(config['FfpnLevels'], outputs)
    outputs = squeeze_feature_maps(outputs)
    seq_boxes, seq_segms = mask_rcnn_head.run(i, outputs, accumulate = accumulate, image_path = image_path)
    # transform the obtained masks into predicted images
    results = []
    if not seq_segms is None:
        for j in range(1, len(seq_segms)):
            segms = seq_segms[j]
            boxes = seq_boxes[j]
            if segms == []:
                continue
            masks = mask_util.decode(segms)
            clss = json_classes[j]
            clss_id = cityscapes_eval.name2label[clss].id
            for k in range(boxes.shape[0]):
                score = boxes[k, -1]
                mask = masks[:, :, k]
                results.append([clss_id, score, mask])
    return results


def evaluate_features_for_mask_prediction_sequence(config, i, outputs, targets, mask_rcnn_head, json_classes, seqIDs = None):
    nT = config['n_target_frames']
    assert len(outputs) == len(targets) == nT
    for t in range(nT):
        evaluate_features_for_mask_prediction(config, i, t,  outputs[t], targets[t],
            mask_rcnn_head, json_classes,
            seqIDs = seqIDs, threshold = config['display_thresh'])


def save_segmentation_result(output_dir, maskrcnn_to_mocity, class_names, im_name, single_frame_results):
    import cv2
    results_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    basename = os.path.splitext(os.path.basename(im_name))[0]
    txtname = os.path.join(output_dir, basename + 'pred.txt')
    logger.info('Saving result in {}'.format(txtname))
    with open(txtname, 'w') as fid_txt:
        if len(single_frame_results) > 0:
            lct = 0
            kct = single_frame_results[0][0]
            for rslt in single_frame_results:
                clss_id, score, mask = rslt
                if clss_id != kct:
                    kct = clss_id
                    lct = 0
                clss_id_mo = maskrcnn_to_mocity[int(clss_id)]
                clss = class_names[clss_id_mo]
                pngname = os.path.join(
                    'results',
                    basename + '_' + clss + '_{}.png'.format(lct))
                lct += 1
                assert not os.path.exists(os.path.join(output_dir, pngname))
                # write txt
                fid_txt.write('{} {} {}\n'.format(pngname, clss_id, score))
                # save mask
                cv2.imwrite(os.path.join(output_dir, pngname), mask * 255)


def evaluate_features_for_mask_prediction(config, i, t, outputs, targets, mask_rcnn_head, json_classes, seqIDs = None, threshold = 0.5):
    import scipy.misc
    assert config['batchsize'] == 1
    t_output_dir = os.path.join(config['save'], 't+%d' % (t+1))
    if not os.path.exists(t_output_dir):
        os.mkdir(t_output_dir)
    # forward the features in the head of mask rcnn
    im_path = seqIDs[0][config['n_input_frames']+t]
    results = get_instance_segmentation_results(config, i, outputs, mask_rcnn_head, json_classes, image_path = im_path, accumulate = (t == config['n_target_frames']-1))
    save_segmentation_result(t_output_dir, config['maskrcnn_to_mocity'], config['classes'], seqIDs[0][-1], results) # we save under the sequence name, not under the actual frame name, to be coherent with previous methods...



def evaluate(config, model, val_loader):
    model.eval()
    stats = {}
    totalValLoss, ctValIt = resetValProgressMultiscale(config, val_loader, stats)
    im_list = val_loader.data_source.dataset.dataset.im_list
    coco_cityscapes_dataset = val_loader.data_source.dataset.dataset.dataset
    cityscapes_dataset = val_loader.data_source.dataset.dataset
    run_head_config = {
        'save' : config['save'],
        'nb_features' : config['nb_features'],
        'split': b'test' if config['test_set'] else b'val'
    }
    mask_rcnn_head = MaskRcnnHead(run_head_config, im_list)

    labels, classes = cityscapes_dataset.get_classes(task = 'instance_segmentation')
    maskrcnn_to_mocity = [label.trainId for label in labels]
    classes_to_id = {}
    for k, v in enumerate(classes):
        classes_to_id[v] = k
    config['classes_to_id'] = classes_to_id
    config['maskrcnn_to_mocity'] = maskrcnn_to_mocity
    config['classes'] = classes
    json_classes = coco_cityscapes_dataset.classes
    # Loop over validation set sample
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets, seqIDs = data
            inputs, targets = prepareMultiscaleForForwardOnGpu(inputs, targets, **{'volatile' : True, 'gpu_id' : config['gpu_id'], 'nb_scales': config['nb_scales']})

            targets = reshapeMultiscaleTargetsForCriterion(targets, config['n_target_frames'], config['nb_features'], config['nb_scales'])

            outputs = model(inputs)
            outputs = format_variable_length_multiscale_sequence(outputs, config['FfpnLevels'], config['n_target_frames'], config['nb_scales'])

            full_scale_outputs = [{k : f[-1] for k,f in out.items() }  for out in outputs]
            full_scale_targets = [{k : f[-1] for k,f in tar.items() }  for tar in targets]

            evaluate_features_for_mask_prediction_sequence(config, i, full_scale_outputs, full_scale_targets, mask_rcnn_head, json_classes, seqIDs)

            del inputs, targets, outputs, full_scale_outputs, full_scale_targets

    mask_rcnn_head.save_annotated_frame_results(config, output_dir=config['save'])
