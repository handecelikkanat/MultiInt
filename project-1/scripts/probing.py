# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import pickle
import numpy as np
import scipy
import scipy.misc
import scipy.special

from sys import argv

scipy.misc.comb = scipy.special.comb

# If you're running on Miikka's system, you need to point to the
# correct installation of sklearn.
try:
    sys.path.insert(0, '/home/mpsilfve/.local/lib/python3.5/site-packages')
except:
    pass

# Set to FINAL, MAXPOOL or AVG.
MODE="MAXPOOL"

# Set PATHs. Hardcoded. Very ugly. Change as needed.
PATH_SENTEVAL = '/home/mpsilfve/src/SentEval/'
PATH_TO_DATA='/home/mpsilfve/src/SentEval/data/'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

def prepare(params, samples):
    pass

def batcher(params, batch):
    sentences = [params.bpe_translation[tuple(s)] for s in batch]

    if MODE == 'FINAL':
        assert(0)
#        return np.vstack([params.multiint[s]['encodings_final'].reshape(1,-1) for s in sentences])
    elif MODE == 'MAXPOOL':
        l = [params.multiint[s].reshape(1,-1) for s in sentences]
        return np.vstack([params.multiint[s].reshape(1,-1) for s in sentences])
#        return np.vstack([params.multiint[s]['encodings_maxpool'].reshape(1,-1) for s in sentences])
    elif MODE == 'AVG':
        assert(0)
#        return np.vstack([params.multiint[s]['encodings_avg'].reshape(1,-1) for s in sentences])
    else:
        assert(0)

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load model
    print("Loading model")
    model = pickle.load(open(argv[1],'br'))
    bpe_translation = pickle.load(open(argv[2],'br'))
    print("Loaded")
    
    params_senteval['multiint'] = model
    params_senteval['bpe_translation'] = bpe_translation

    se = senteval.engine.SE(params_senteval, batcher, prepare)
#    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',
#                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
#                      'OddManOut', 'CoordinationInversion']

    transfer_tasks = ['SubjNumber']
    results = se.eval(transfer_tasks)
    print(results)
