# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from logging import getLogger
import torch
import shutil

logger = getLogger()
CHECKPOINT_tempfile = 'checkpoint.temp'

# For now keep periodic saving on its own and use this only for checkpointing
# The problem with periodic saving would be that we would save multiple times the options for instance,
# which we want to avoid.
def save_checkpoint(state, is_best, savedir, filename='checkpoint.pth.tar'):
    ''' Save the model to a temporary file first,
    then copy it to filename, in case the signal interrupts
    the torch.save() process.
    '''
    torch.save(state, os.path.join(savedir, CHECKPOINT_tempfile))
    if os.path.isfile(os.path.join(savedir, CHECKPOINT_tempfile)):
        os.rename(os.path.join(savedir, CHECKPOINT_tempfile),
            os.path.join(savedir, filename))
    if is_best:
        shutil.copyfile(os.path.join(savedir, filename),
            os.path.join(savedir, 'model_best.pth.tar'))
    logger.info("Checkpoint saved inside %s." % savedir)
