# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torchnet.meter import meter
import numpy as np
import time
import math
import logging

# ------------------------------------------------------------------------------
# Reshaping function required for add function below, dependent on the input/target's nature and dimensions
def reshape_spatial_map(input):
    if input.ndim == 3:
        # C x H x W
        output = np.argmax(input, 0)
    elif input.ndim == 2:
        # H x W
        output = input
    else:
        logger.error('Wrong dimension for input : %d' % input.dim())
        exit()
    return output.flatten()

#-------------------------------------------------------------------------------
# SemanticSegmentationMeter definition
class SemanticSegmentationMeter(meter.Meter):
    """
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    """

    def __init__(self, classes, normalized=False, final_untagged = False, title=None):
        """
        Args:
            k (int): number of classes in the classification problem
            normalized (boolean): Determines whether or not the confusion matrix
                is normalized or not
            final_untagged : Determines whether or not the final category corresponds
            to 'untagged' label (frequent in semantic segmentation datasets) - category
            for which the performance is not considered.
        """
        super(SemanticSegmentationMeter, self).__init__()
        self.k = len(classes)
        self.conf = np.ndarray((self.k, self.k), dtype=np.int32)
        self.normalized = normalized
        self.classes = classes
        self.final_untagged = final_untagged
        self.title = title
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add_flattened(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (numpy array): Should be a N-array of predicted categories between 0 and K-1
                obtained from the model for N examples and K classes.
            target (numpy array): Should be a N-array of predicted categories between 0 and K-1
                obtained from the model for N examples and K classes.
        """

        assert predicted.ndim == target.ndim == 1, \
            'predicted and target dimension should be 1'
        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        k_input = self.k - (1 if self.final_untagged else 0)
        assert (predicted.max() < k_input) and (predicted.min() >= 0), \
            'predicted values are not between 0 and k_input-1'
        assert (target.max() < self.k) and (target.min() >= 0), \
            'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        if self.final_untagged:
            assert np.all(np.equal(conf[:, self.k-1], 0)), \
                'Prediction is not supposed to fall in untagged class'
            # We do not consider the predictions for the untagged samples
            conf[self.k-1,:].fill(0)

        self.conf += conf
        return conf # For per frame performance measures

    def add(self, predicted, target, flatten_for_conf):
        """
        Reshapes the inputs predicted and target according to the flatten_for_conf function.
        We prefer the leave the definition of this function to the user, so that there
        is no ambiguity as to the dimensions of the inputs.
        Once this function has been defined, updating the confusion matrix remains a one liner.
        Args:
            predicted (tensor)
            target (tensor)
        """
        predicted = predicted.cpu().squeeze().numpy()
        target = target.cpu().squeeze().numpy()

        predicted = flatten_for_conf(predicted)
        target = flatten_for_conf(target)
        self.add_flattened(predicted, target)

    def update_values(self, per_class_only=False, aggregated_only=False):
        with np.errstate(invalid='ignore'):
            total_accurate = 0
            k_tagged = self.k - (1 if self.final_untagged else 0)
            # Per class values
            per_class_precision = np.empty((self.k))
            per_class_iou = np.empty((self.k))
            for t in range(self.k):
                per_class_precision[t] = np.float64(self.conf[t][t]) / np.float64(self.conf[t, :].sum())
                per_class_iou[t] = np.float64(self.conf[t][t]) / np.float64(self.conf[t, :].sum() + self.conf[:, t].sum() - self.conf[t][t])
                total_accurate += self.conf[t][t]
            if per_class_only:
                return per_class_precision, per_class_iou
            else:
                # Aggregated values
                accuracy = np.float64(total_accurate) / np.float64(self.conf.sum())
                n_classes_valid_for_precision, mean_precision = 0, 0
                n_classes_valid_for_iou, mean_iou = 0, 0
                for t in range(k_tagged):
                    if not np.isnan(per_class_precision[t]):
                        mean_precision += per_class_precision[t]
                        n_classes_valid_for_precision += 1
                    if not np.isnan(per_class_precision[t]) and not np.isnan(per_class_iou[t]):
                        mean_iou += per_class_iou[t]
                        n_classes_valid_for_iou += 1
                mean_precision /= np.float64(n_classes_valid_for_precision)
                mean_iou /= np.float64(n_classes_valid_for_iou)
                if aggregated_only:
                    return accuracy, mean_precision, mean_iou
                else:
                    return per_class_precision, per_class_iou, \
                        accuracy, mean_precision, mean_iou

    def average_over_classes(self, values, classes_to_average_over):
        mean_value = 0
        n_classes_for_mean = 0
        for i, c in enumerate(self.classes):
            if c in classes_to_average_over and not np.isnan(values[i]):
                mean_value += values[i]
                n_classes_for_mean += 1
        with np.errstate(invalid='ignore'):
            mean_value = np.float64(mean_value)/n_classes_for_mean
        return mean_value


    def extract_values_for_classes(self, values, classes_to_extract):
        rslt = np.empty((len(classes_to_extract)))
        for i, c in enumerate(classes_to_extract):
            assert c in self.classes, \
                '%s is not part of the classes !' % c
            j = self.classes.index(c)
            rslt[i] = values[j]
        return rslt


    def value(self, perf_type=None, selected_classes = None):
        """
        Returns:
            Confusion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        with np.errstate(invalid='ignore'):
            if perf_type is None:
                if selected_classes is None:
                    per_class_precision, per_class_iou = \
                            self.update_values(per_class_only=True)
                    if self.normalized:
                        conf = self.conf.astype(np.float32)
                        return conf / conf.sum(1).clip(min=1e-12)[:, None], per_class_precision, per_class_iou
                    else:
                        return self.conf, per_class_precision, per_class_iou
                else:
                    per_class_precision, per_class_iou = \
                        self.update_values(per_class_only=True)
                    per_class_precision = self.extract_values_for_classes(per_class_precision, selected_classes)
                    per_class_iou = self.extract_values_for_classes(per_class_iou, selected_classes)
                    return per_class_precision, per_class_iou
            else: # return aggregated performance
                if selected_classes is None:
                    accuracy, mean_precision, mean_iou = self.update_values(aggregated_only=True)
                    if perf_type == 'pp':
                        return accuracy
                    elif perf_type == 'pc':
                        return mean_precision
                    elif perf_type == 'iou':
                        return mean_iou
                    elif perf_type == 'all':
                        return accuracy, mean_precision, mean_iou
                    else:
                        logger.error('Unknown aggregated performance type : %r' % perf_type)
                        exit()
                else:
                    per_class_precision, per_class_iou = \
                        self.update_values(per_class_only=True)
                    if perf_type == 'pc':
                        return self.average_over_classes(per_class_precision, selected_classes)
                    elif perf_type == 'iou':
                        return self.average_over_classes(per_class_iou, selected_classes)
                    elif perf_type == 'all':
                        mpcp_oc = self.average_over_classes(per_class_precision, selected_classes)
                        miou_oc = self.average_over_classes(per_class_iou, selected_classes)
                        return mpcp_oc, miou_oc

    def __str__(self):
        per_class_precision, per_class_iou, \
            accuracy, mean_precision, mean_iou = self.update_values()
        s = 'Confusion matrix' if self.title is None else self.title
        s += ':\n['
        maxCnt = np.max(self.conf)
        with np.errstate(invalid='ignore'):
            nDigits = max(8, 1 + math.ceil(np.log10(np.float64(maxCnt))))
        for t in range(self.k):
            class_precision = '%2.3f' % (per_class_precision[t] * 100)
            if t == 0:
                s += '['
            else:
                s += ' ['
            for p in range(self.k):
                s += ('%'+'%d'%nDigits+'d') % self.conf[t][p]
            if t == self.k:
                s += ']]  ' + class_precision + '% \t[class: ' + self.classes[t] + ']\n'
            else:
                s += ']   '+ class_precision + '% \t[class: ' + self.classes[t] + ']\n'

        s += ' + global accuracy: %2.1f %% \n' % (accuracy * 100)
        s += ' + mean class precision: %2.1f %% \n' % (mean_precision * 100)
        s += ' + mean intersection over union: %2.1f %% \n' % (mean_iou * 100)
        return s
