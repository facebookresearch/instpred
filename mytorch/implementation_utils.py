# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import os
import subprocess
# Copied and adapted from FaderNetworks
def get_dump_directory(rootdir, job_id=None):
    """
    Create a directory to store the experiment.
    """
    if job_id is None:
        # create a random name for the experiment
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_name = ''.join(random.choice(chars) for _ in range(10))
            dump_path = os.path.join(rootdir, exp_name)
            if not os.path.isdir(dump_path):
                break
    else:
        exp_name = job_id
        dump_path = os.path.join(rootdir, exp_name)
    return dump_path + '/', exp_name

def get_nb_parameters(model):
    params = model.parameters()
    s = 0
    for p in params :
        # Compute nb of parameters in layer and add it to the cumulative nb of params
        sz = p.size() ; m = 1
        for k in sz : m *= k
        s += m
    return s
