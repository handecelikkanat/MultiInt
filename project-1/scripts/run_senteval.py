#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals
#absolute_import,
import sys, getopt
import io
import numpy as np
import logging
import torch


PATH_TO_DATA = './data'

PATH_TO_SENTEVAL = '/wrk/vazquezc/DONOTREMOVE/git/SentEval'
#PATH_TO_INFERSENT='/home/local/vazquezj/git/InferSent'
PATH_TO_DATA = '/wrk/vazquezc/DONOTREMOVE/git/SentEval/data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

PATH_TO_ONMT='/wrk/vazquezc/DONOTREMOVE/git/OpenNMT-py-NI2_ALESSANDRO'
sys.path.insert(0, PATH_TO_ONMT)

import onmt.inputters as inputters
from available_models import run_sentevalUtil
from apply_bpe import BPE
import codecs

fields=None
model=None
codes = codecs.open("/wrk/vazquezc/DONOTREMOVE/git/OpenNMT-py-NI2_ALESSANDRO/available_models/someModels/codec10K-en.multi30k-alessandro.bpe", encoding='utf-8')
bpe = BPE(codes)

def invert_permutation(p):
    p = p.tolist()
    return torch.tensor([p.index(l) for l in range(len(p))])

# SentEval batcher
def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    sents=[]
    for sent in batch:
        #apply bpe
        str1 = ' '.join(sent)
        #print(str1)
        bpeversion=bpe.process_line(str1)
        sents.append(bpeversion)
        #print(bpeversion)

    print("###")
    #print(sents[1])
    #print(fields)
    data = run_sentevalUtil.build_dataset(fields, sents)
    print('-------------data.examples--------')
    print(data.examples)
    #print(data)
    data_iter = inputters.OrderedIterator(
        dataset=data, device="cuda",
        batch_size=len(batch), train=False, sort=False,
        sort_within_batch=True, shuffle=False)
    embeddings = []
    print('-------------data_iter--------')
    print(data_iter)
    print(data_iter.data())
    print(data_iter.data().examples)
    for batchz in data_iter:
        permutation = batchz.indices
        print(permutation)
        # (1) Run the encoder on the src.
        src = inputters.make_features(batchz, 'src', 'text')
        src_lengths = None
        _, src_lengths = batchz.src
        enc_states, memory_bank = model.encoders[model.encoder_ids["en"]](src, src_lengths)
        alphasZ, memory_bank = model.attention_bridge(memory_bank, src) # [r,bsz,nhid]
        memory_bank = memory_bank[:, invert_permutation(permutation), :]  # shape=[att_heads,batch_size,rnn_size]
    
        output = memory_bank.transpose(0, 1).contiguous()    
        #sentvec = output.view(output.size()[0], -1).detach()
        sentvec = torch.mean(output, 1).detach()#.data.cpu().numpy()
        print('sentvec size: ')
        print(sentvec.size())
        return sentvec
        #sentvec = output.unsqueeze(0)
        #print("sent")
        #print(sentvec.size())
        #embeddings.append(sentvec)
    
    #print(len(embeddings))
    #embeddings = np.vstack(embeddings)
    #return embeddings
    print("NOPE "+str(len(batch)))
    return None

# Set params for SentEval
#params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}
#params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
#params_senteval['classifier'] = {'nhid': 200, 'optim': 'adam', 'batch_size': 64,
#                                 'tenacity': 5, 'epoch_size': 4, 'dropout': 0.1}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


if __name__ == "__main__":
    argv=sys.argv[1:]
    opts, args = getopt.getopt(argv,"hi:o:t:s:",["ifile=","ofile=","tgtlang=","save_model="])
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    model_path = None
    transfer_tasks = None
    tgt=None
    for opt, arg in opts:
        print(opt, arg)
        if opt == '-h':
            print('test.py -i pathtomodel -o logidtsc or mlp')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            model_path = str(arg)
        elif opt in ("-o", "--ofile"):
            choose = int(arg)
            if choose == 0 :
                params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}   
                transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'WordContent',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16']

            else:
                params_senteval['classifier'] = {'nhid': 200, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4, 'dropout': 0.1}
                transfer_tasks = ['Length', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
        elif opt in ("-t", "--tgtlang"):
            tgt = str(arg)
        elif opt in ("-s", "--save_model"):
            save_model_path= str(arg)
      
    print('SentEval output saved on:'+save_model_path)


#    params_senteval['classifier'] = {'nhid': 200, 'optim': 'adam', 'batch_size': 64,
#                                 'tenacity': 5, 'epoch_size': 4, 'dropout': 0.1}

#    model_path = "/home/local/raganato/SentEval_OpenNMT/en-de.nil50.monolingual.model_step_188000.pt"

    print(model_path)


    torch.cuda.set_device(0)
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model = checkpoint['whole_model']
    device = torch.device("cuda")
    #device = torch.device("cpu")
    model.to(device)

    model.eval()

    fields = inputters.inputter.load_fields_from_vocab(
        {'src': model.src_vocabs["en"],
         'tgt': model.tgt_vocabs[tgt]})

    se = senteval.engine.SE(params_senteval, batcher)#, prepare)
#    transfer_tasks = ['SNLI', 'CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 
#                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 
#                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

#    transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
#                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
#                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16']
#    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']


    results = se.eval(transfer_tasks)
    np.save(save_model_path, results) 
    print(results)

