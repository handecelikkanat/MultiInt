(Ale’s current project, also Edinburgh people) Analyzing Transformer Heads. Pruning (Edinburgh) vs. growing (Ale) transformer heads to find an optimal number of non-redundant heads. Experiments on massive data on new CSC cluster.

### TODO
* train a system with 6 layers and 1 ah of 64 dim (in training).  Then, re-train the system keeping those ah fixed and adding a new one.
* training pre-trained word vector on the training data and use all them as fixed input layer (in training)


### Benchmark data
* Train: *europarl, news-commentary, commoncrawl, rapid, paracrawl*  (allfiltered, shuffled, bpe35k from our wmt submission) 11555682 sentences (probably i will scale down to 2.5 M and re-run the training)
* Dev: *newstest2013*
* Test: *newstest2014*

## Training commands
python  $opennmt/train.py -data $trainData -save_model $savemodel -layers 1 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 1  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0


python  $opennmt/train.py -data $trainData -save_model $savemodel -layers 1 -rnn_size 512 -word_vec_size 512 -train_steps 200000  -dropout 0.1 -batch_size 256 -optim adam -learning_rate 0.001 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0

python  $opennmt/train.py -data $trainData -save_model $savemodel -layers 1 -rnn_size 64 -word_vec_size 64 -transformer_ff 2048 -heads 1  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0

## Scores

Testing last checkpoint (200k training steps).

BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.2.11



| Model                  | number of parameters     | BLEU newstest2014 |
| ---                    | ---                      |---                |
| 1-layer LSTM           | 57860738                 |       21.71            |
| 6-layer TR 8AH tot 512 (default)    |   95963778         |         training          |
| 6-layer TR 1AH 64 we 512    |   79424514         |         training          |
| 1-layer TR 1AH 64 we 512    |   56425154         |         training          |
| 1-layer TR 1AH 64 we 512 (init fasttext)   |   56425154         |         training          |
| 1-layer TR 1AH tot 512 dim  | 59181698            |       18.47            |
| 1-layer TR 8AH tot 512 dim  | 59181698            |       21.96            |
| 1-layer TR 1AH tot 64 dim and we 64  |  7087810            |       10.47            |
