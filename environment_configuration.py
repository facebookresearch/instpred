# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ------------------------------------------------------------------------------
# Dataset related
# Precomputed P5 input and target features should be stored here (for the command, see README.md)
PRECOMPUTED_FEATURES_DIR = 'precomputed/P5_fr3_features'

# ------------------------------------------------------------------------------
# Experiments related
# root where all experimental "heavy" results will be saved, eg. predictions, checkpoints, etc.
ROOT_SAVE_HEAVY_RESULTS='results'
# root where all experimental "light" results will be saved, eg. logs and params
ROOT_LOGS='logs'

# ------------------------------------------------------------------------------
# Job related:
import os
# Note: this is helpful if you wish to launch the job on a cluster, to save
# results in a directory whose name begins with the job id.
CLUSTER_JOB_ID = ''
