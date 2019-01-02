# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

""" This script takes a path to a result directory (validation) and outputs:
    - the confusion matrix SEGM,
    - the confusion matrix GT, (checked for copy baseline nT = 3)
    - the instance segm evaluation GT (checked for copy baseline nT = 3)
"""
# ------------------------------------------------------------------------------
# Imports
from mypython.logger import create_logger
import os
import torch
import numpy as np
from SemanticSegmentationMeter import SemanticSegmentationMeter, reshape_spatial_map

from cityscapesDatasetAndFeatures import CityscapesDatasetAndFeatures

from mytorch.printing import bcolors
from PIL import Image
from operator import itemgetter

from detectron.core.config import cfg
from detectron.datasets.dataset_catalog import get_raw_dir

# ------------------------------------------------------------------------------
# Options
conf_threshold = 0.5 # threshold on confidence for conversion to semantic segm

from environment_configuration import ROOT_SAVE_HEAVY_RESULTS
# Inputs
root_dir = b'%s/%s' % (ROOT_SAVE_HEAVY_RESULTS, os.environ['PREDICTIONS_PATH'])

# Outputs
output_dir = b'%s/%s' % (ROOT_SAVE_HEAVY_RESULTS, os.environ['EVALUATION_RESULTS'])
# using these variables because of the cityscapes evaluation script

# Target semantic segmentations
target_segm_dir = 'precomputed/groundtruth/val/maskrcnn_target_semantic_segmentation/ffpn_levels_4_thresh0.5'

# ------------------------------------------------------------------------------
# make directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logger = create_logger(os.path.join(output_dir, 'eval_precomputed_results.log'))
logger.info('============ Initialized logger ============')
logger.info('root dir : %s' % root_dir)
logger.info('output dir : %s' % output_dir)

# Load dataset for a few useful functions
cityscapes_val = CityscapesDatasetAndFeatures(
    split = b'val',
    frame_ss = None, # note : not used, but value needed
    nSeq = 1,
    features = ['RGBs'],
    savedir = output_dir,
    loaded_model = '' # To avoid having it loaded
)

def convert_instance_result_to_semantic_segmentation(root, filename, classes_to_id, maskrcnn_to_mocity, threshold=0):
    logger.debug('Convert '+ filename + ' to semantic segmentation.')
    # Initialize the semantic segmentation result
    S = np.zeros((1024, 2048), dtype = 'uint8')
    S.fill(classes_to_id['background'])
    # Load all the instances in a list of lists path / class / confidence
    instances = []
    with open(os.path.join(root, filename)) as f:
        for line in f:
            instances.append(line.strip().split(' '))
    # If any instances were detected and segmented
    if instances != []:
        # Sort the instances in ascending confidence
        instances = sorted(instances, key=itemgetter(2))
        # double check that the intances are sorted in ascending confidence
        m = instances[0][2]
        for line in instances:
            assert m <= line[2]
            m = line[2]
        # Loop over the instances
        for inst in instances:
            # Threshold the score :
            if float(inst[2]) >= threshold:
                # Load the mask
                mask = np.array(Image.open(os.path.join(root, inst[0])))
                # Convert the instance's class to moving objects cityscapes label k
                k = maskrcnn_to_mocity[int(inst[1])]
                # Replace in the result with k whereever mask == 1
                try :
                    assert (np.all( np.unique(mask) == np.array([0, 255]))
                        or  np.all( np.unique(mask) == np.array([0]))
                        or  np.all( np.unique(mask) == np.array([255])))
                except Exception:
                    # For Camille's masks
                    if mask.ndim == 3:
                        assert np.all(np.equal(mask[:,:,0], mask[:,:,1]))
                        assert np.all(np.equal(mask[:,:,0], mask[:,:,2]))
                        mask = mask[:,:,0]
                    mask = (mask > 0) * 255
                    assert (np.all( np.unique(mask) == np.array([0, 255]))
                        or  np.all( np.unique(mask) == np.array([0]))
                        or  np.all( np.unique(mask) == np.array([255])))

                S[mask == 255] = k
    return S

def load_groundtruth(gt_root, filename, split='val'):
    city = filename.split('_')[0]
    gtname = '_'.join(filename.split('_')[:3] + ['gtFine_labelIds.png'])
    filepath = os.path.join(gt_root, 'gtFine', split, city, gtname)
    assert os.path.exists(filepath),\
        'Ground truth file does not exist at path %r' % filepath
    gt = np.array(Image.open(filepath))
    return gt

def convert_gt_semantic_segmentation_to_mocity(gt, city_to_mocity):
    for k, v in enumerate(city_to_mocity):
        gt[gt == k] = v
    return gt

def evaluate_instance_segmentations(
        json_dataset_name, inp_results_dir, out_results_dir):
    if cfg.CLUSTER.ON_CLUSTER:
        # On the cluster avoid saving these files in the job directory
        inp_results_dir = '/tmp'
    res_file = os.path.join(
        inp_results_dir, 'segmentations_' + json_dataset_name + '_results')
    res_file += '.json'

    results_dir = os.path.join(inp_results_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    os.environ['CITYSCAPES_RESULTS'] = inp_results_dir
    os.environ['CITYSCAPES_DATASET'] = get_raw_dir(json_dataset_name)
    os.environ['OUTPUTS_DIR'] = out_results_dir
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval
    logger.info('Evaluating...')
    cityscapes_eval.main()

def evaluate_instances_converted_to_semantic_segmentation(dataset, root_dir, target_segm_dir, output_dir, split='val'):
    import scipy.misc
    logger.info('Performance computed using the dataset groundtruth')
    labels, classes = dataset.get_classes(task = 'instance_segmentation')
    maskrcnn_to_mocity = [label.trainId for label in labels]
    city_to_mocity = maskrcnn_to_mocity
    classes_to_id = {}
    for k, v in enumerate(classes):
        classes_to_id[v] = k
    C = len(classes)

    groundtruth_root = get_raw_dir(dataset.dataset.name)

    results = os.listdir(root_dir)
    title = bcolors.CYAN + 'Confusion matrix %s' + bcolors.END
    conf_gt = SemanticSegmentationMeter(classes, final_untagged = True, title = title % 'gt')
    conf_segm = SemanticSegmentationMeter(classes, final_untagged = True, title = title % 'segm')
    for res in results:
        if '.txt' in res:
            # Transform the instance segmentation results to semantic segmentation
            segm = convert_instance_result_to_semantic_segmentation(
                root_dir, res, classes_to_id, maskrcnn_to_mocity, threshold=conf_threshold)
            if split == 'test':
                segm_submission = convert_instance_result_to_semantic_segmentation(
                    output_dir, res, classes_to_id, maskrcnn_to_mocity, threshold=conf_threshold)
            frameID = '_'.join(res.split('_')[0:3])
            save_output = '_'.join((frameID, 'prediction'))
            filename = os.path.join(output_dir, 'outputs_semantic_segmentation', '%s.png' % save_output)
            if not os.path.exists(os.path.dirname(filename)):os.mkdir(os.path.dirname(filename))
            scipy.misc.imsave(filename, segm)
            logger.info('Saved output semantic segm to %s' % filename)

            if split == 'test':
                filenamesub = os.path.join(output_dir, 'semantic_segmentation_for_submission', '%s.png' % save_output)
                if not os.path.exists(os.path.dirname(filenamesub)):os.mkdir(os.path.dirname(filenamesub))
                scipy.misc.imsave(filenamesub, segm_submission)
                print('Saved output semantic segm for submission to %s' % filenamesub)
            else:
                gt = load_groundtruth(groundtruth_root, res)
                gt = convert_gt_semantic_segmentation_to_mocity(gt, city_to_mocity)

                save_gt = '_'.join((frameID, 'gtconvertedmocity'))
                gt_fn = os.path.join(output_dir, 'gt_semantic_segmentation_converted_to_mocity_labels', '%s.png' % save_gt)
                if not os.path.exists(os.path.dirname(gt_fn)):os.mkdir(os.path.dirname(gt_fn))
                scipy.misc.imsave(gt_fn, gt)
                logger.info('Saved gt semantic segm to %s' % gt_fn)


                save_target = '_'.join((frameID, u'maskrcnntargetsemanticsegmentation'))
                target_segm_filename = os.path.join(target_segm_dir , '%s.png' % save_target)
                loaded_target_sem_segm = scipy.misc.imread(target_segm_filename)

                conf_gt.add(torch.from_numpy(segm), torch.from_numpy(gt), reshape_spatial_map)
                conf_segm.add(torch.from_numpy(segm), torch.from_numpy(loaded_target_sem_segm), reshape_spatial_map)

    # logger.info('Confusion matrix GT')
    logger.info(conf_gt)
    logger.info('mIOU over moving objects only : %2.2f' % (conf_gt.value(
        perf_type = 'iou', selected_classes = classes[1:9]) * 100))
    logger.info('where classes are : %r' % classes[1:9])
    _, per_class_iou_mo =  conf_gt.value(selected_classes = classes[1:9])
    logger.info('Per class iou for moving objects : ')
    logger.info(per_class_iou_mo)

    # logger.info('Confusion matrix segm')
    logger.info(conf_segm)
    logger.info('mIOU over moving objects only : %2.2f' % (conf_segm.value(
        perf_type = 'iou', selected_classes = classes[1:9]) * 100))
    logger.info('where classes are : %r' % classes[1:9])
    _, per_class_iou_mo =  conf_segm.value(selected_classes = classes[1:9])
    logger.info('Per class iou for moving objects : ')
    logger.info(per_class_iou_mo)

evaluate_instance_segmentations(cityscapes_val.dataset.name, root_dir, output_dir)
evaluate_instances_converted_to_semantic_segmentation(cityscapes_val, root_dir, target_segm_dir, output_dir, split ='val')
