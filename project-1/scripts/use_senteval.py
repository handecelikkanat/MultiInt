
'''
in Joerg's: 
          source ~/.barshrc 
          source activate senteval
RUNNING WITH: 
python available_models/use_senteval.py \
          -model available_models/someModels/Many2EN-EN2Many.mono.model_step_82000.pt \
          -src_lang en \
          -src fake \
          -tgt_lang de \
          -tgt fake \
          -gpu 0 

python available_models/use_senteval.py \
          -train_from available_models/someModels/Many2EN-EN2Many.mono.model_step_82000.pt \
          -gpuid 0 \
          -use_attention_bridge True \
          -src_tgt FAKE \
          -data FAKE
'''

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import io
import numpy as np
import logging

import csv
# import onmt_utils

# Set PATHs
PATH_TO_SENTEVAL = '/home/local/vazquezj/git/SentEval'
#PATH_TO_INFERSENT='/home/local/vazquezj/git/InferSent'
PATH_TO_DATA = '/home/local/vazquezj/git/SentEval/data'

#####TAITO:
PATH_TO_SENTEVAL = '/wrk/vazquezc/DONOTREMOVE/git/SentEval'
#PATH_TO_INFERSENT='/home/local/vazquezj/git/InferSent'
PATH_TO_DATA = '/wrk/vazquezc/DONOTREMOVE/git/SentEval/data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import torch
import numpy as np

PATH_TO_ONMT='/wrk/vazquezc/DONOTREMOVE/git/OpenNMT-py-NI2_ALESSANDRO'
sys.path.insert(0, PATH_TO_ONMT)

#sys.path.insert(0, PATH_TO_INFERSENT)
#import data

def invert_permutation(p):
    p = p.tolist()
    return torch.tensor([p.index(l) for l in range(len(p))])

# SentEval prepare and batcher
# I THINK I DON'T NEED THE prepare FUNCTION...
#def prepare(params, samples):
#    import ipdb
#    ipdb.set_trace()
#    return


#################################################
#   STEP1: import the trained model
##################################################

#import ipdb; ipdb.set_trace(context=5)
PATH_TO_ONMT='/home/local/vazquezj/git/OpenNMT-py-NI2_ALESSANDRO'
sys.path.insert(0, PATH_TO_ONMT)

import onmt # this complains if loss.py lines 20--33 are not commented

import argparse
parser = argparse.ArgumentParser(description='train.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
onmt.opts.add_md_help_argument(parser)
onmt.opts.train_opts(parser)
opt = parser.parse_args()
torch.cuda.set_device(0)
checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
model_opt = checkpoint['opt']
#src_tgt = model_opt.src_tgt
model = checkpoint['whole_model']
import onmt.inputters as inputters


if opt.train_from.find('en-') > -1:
  lpair = opt.train_from[opt.train_from.find('en-'):opt.train_from.find('en-')+5]
  ind = model_opt.src_tgt.index(lpair)
elif opt.train_from.find('-en') > -1:
  lpair = opt.train_from[opt.train_from.find('-en')-2:opt.train_from.find('-en')+3]
  ind = model_opt.src_tgt.index(lpair)
else:
  lpair = model_opt.src_tgt[0]
  ind = model_opt.src_tgt.index(opt.src_tgt[0])
  #ind = model_opt.src_tgt.index('en-de')


fields = inputters.inputter.load_fields_from_vocab(
            {'src': model.src_vocabs[model_opt.src_tgt[ind].split('-')[0]],
             'tgt': model.tgt_vocabs[model_opt.src_tgt[ind].split('-')[1]]}) 

if len(opt.gpuid) > 0:
  cur_device = "cuda"
  device = torch.device("cuda")
  model.to(device)
else:
  cur_device = "cpu"

print("current device: ",cur_device)



def batcher(params, batch, key='EN'):
    #import ipdb; ipdb.set_trace(context=5)
    #batch_size = len(batch)
    '''#----------------------------------------------------------
    # Onmt patch:
    #----------------------------------------------------------'''
    #    generate a temporal textfile to pass to Onmt modules
    #print(batch)
    batch = [sent if sent != [] else ['.'] for sent in batch]
    #print('------------again----------------')
    #print(batch)
    batchfile = "/wrk/vazquezc/current-batch."+lpair+str(sentevalseed)+".tmp"
    with open(batchfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=" ", quotechar='|')
        writer.writerows(batch)
    #    pass batch textfile -> builds features
    # batch is already lowercased, tokenized and normalized

    # ---------- this is for XNLI evaluation: ------------------------
    langpair=lpair
    index=ind
    if key.find('EN') > -1:
      lang = 'en'
    elif key.find('DE') > -1:
      lang = 'de'
      if opt.train_from.find('en-') == -1:
        langpair ='de-en'
        #index = model_opt.src_tgt.index(langpair)
    elif key.find('FR') > -1:
      lang = 'fr'
      if opt.train_from.find('en-') == -1:
        langpair ='fr-en'
        #index = model_opt.src_tgt.index(langpair)
    else:
      lang = 'en' # only for multi30k data, else change!
    
    # ------------------ end of xnli ----------

    #  apply BPE to batch        
    BPE_APPLY="/wrk/vazquezc/DONOTREMOVE/git/subword-nmt/subword_nmt/apply_bpe.py"
    enBPECODES="/wrk/vazquezc/DONOTREMOVE/git/lingeval97/codec10K-"+lang+".multi30k-alessandro.bpe" #multi30k
    #enBPECODES="~/git/OpenNMT-py-NI2_ALESSANDRO/available_models/europarlModels/codecs32K.en" #europarl
    OUTP=batchfile
    cmd="python "+BPE_APPLY+" -c "+enBPECODES+" -i "+OUTP+" -o "+OUTP+".bpe"
    os.system(cmd)
    os.remove(batchfile)
    batchfile=OUTP+".bpe"

    data = inputters.build_dataset(fields=fields,
                             data_type='text',
                             src_path=batchfile,
                             tgt_path=None,
                             src_dir='',
                             sample_rate='16000',
                             window_size=0.2,
                             window_stride=0.01,
                             window='hamming',
                             use_filter_pred=False)
    #    generate iterator (of size 1) over the dataset
    bsize=len(batch)
    #print(bsize, params.batch_size)
    #print(data.examples)
    #print('###############')
    data_iter = inputters.OrderedIterator(
        dataset=data, device=cur_device,
        batch_size=bsize, train=False, sort=False,
        sort_within_batch=True, shuffle=False)
    #    pass the batch information through the encoder
    #print('-------------data_iter--------')
    #print(data_iter)
    #print(data_iter.data())
    #print(data_iter.data().examples)

    #print('#######out_here######')

    for BATCH in data_iter:
        #print('here_inside')
        permutation = BATCH.indices
        src = inputters.make_features(BATCH, side='src', data_type="text")
        src_lengths = None
        _, src_lengths = BATCH.src
        enc_states, memory_bank = model.encoders[index](src, src_lengths)
        enc_final, memory_bank = model.attention_bridge(memory_bank, src)

    #print('out_again')
    memory_bank = memory_bank[:,invert_permutation(permutation),:] #shape=[att_heads,batch_size,rnn_size]
    #----------------------------------------------------------
    
    #import ipdb; ipdb.set_trace()
    output = memory_bank.transpose(0, 1).contiguous() #shape=[batch_size, att_heads,rnn_size]
    #output = output.view(output.size()[0], -1).detach()
    output = torch.mean(output, 1).detach()

    # make sure embeddings has 1 flattened M matrix per row.
    #memory_bank = memory_bank.transpose(0, 1).contiguous() #shape=[batch_size,att_heads,rnn_size]
    #embeddings = [mat.transpose(0,1).flatten().detach() for mat in memory_bank]
    #embeddings = np.vstack(embeddings)
    os.remove(batchfile)
    #return embeddings
    return output.cpu().numpy()

#####################
#   call SentEval
#####################

# Set params for SentEval
import random
#random.seed(1234)
sentevalseed = random.randint(1111, 9999)
print('using seed', sentevalseed)
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'seed': sentevalseed}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher)#, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    # '''DO NOT WORK: CR,'MPQA', '''
    # # override: only test the ones that work
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',                       
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',                       
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 
    #                   'OddManOut', 'CoordinationInversion']
    #self.list_tasks = ['SNLI', 'ImageCaptionRetrieval']
    #transfer_tasks = ['STS12', 'STS13']
    choose=0
    if choose == 0 :
      params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                     'tenacity': 5, 'epoch_size': 4}   
      transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                        'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'WordContent',
                        'STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    else:
      params_senteval['classifier'] = {'nhid': 200, 'optim': 'adam', 'batch_size': 64,
                                     'tenacity': 5, 'epoch_size': 4, 'dropout': 0.1}
      transfer_tasks = ['Length', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    
    #transfer_tasks = ['XNLI']
    #transfer_tasks = ['CR', 'MPQA']

    transfer_tasks = ['MR', 'SUBJ', 'SST2', 'SST5', 'TREC','MRPC', 'SNLI',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',                       
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 
                      'OddManOut', 'CoordinationInversion',
                      'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'CR', 'MPQA']
    transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents',                
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    np.save(opt.save_model, results) 
    print(results)

# ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment', 'STSBenchmark', 'SNLI', 'ImageCaptionRetrieval', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

# source .bashrc
# source activate senteval
# cd git/OpenNMT-py-NI2_ALESSANDRO/
# python available_models/use_senteval.py -train_from available_models/someModels/en-cs.nil.model_step_15000.pt  -src_tgt en-cs     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_en-cs.nil.model_step_15000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/en-fr.nil.model_step_14000.pt  -src_tgt en-fr     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_en-fr.nil.model_step_14000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/en-de.nil.model_step_20000.pt    -src_tgt en-de     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_en-de.nil.model_step_20000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/Many2EN-EN2Many.mono.model_step_82000.pt  -src_tgt en-de     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_Many2EN-EN2Many.mono.model_step_82000.OUT
# python available_models/use_senteval.py -train_from available_models/someModels/multi-TO-multi.mono.model_step_91000.pt  -src_tgt en-de     -gpuid 0    -use_attention_bridge True    -data FAKE1~  > ~/Documents/attBridge/senteval_multi-TO-multi.mono.model_step_91000.OUT


