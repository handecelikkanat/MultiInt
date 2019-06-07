(Ale’s current project, also Edinburgh people) Analyzing Transformer Heads. Pruning (Edinburgh) vs. growing (Ale) transformer heads to find an optimal number of non-redundant heads. Experiments on massive data on new CSC cluster.

### TODO
* retrain 3rd ah 1 layer
* retrain 2nd ah 6 layer

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

from previous AH model (He distribution (and generator to the previous one))

| Model                  | number of parameters     | BLEU newstest2014 |
| ---                    | ---                      |---                |
| 1-layer LSTM           | 57860738                 |       21.71            |
| 6-layer TR 8AH tot 512 (default)    |   95963778         |    25.91               |
| 6-layer TR 1AH 64 we 512    |   79424514         |          24.04         |
| 6-layer TR 2AH 64 we 512   |   46709378   (35077888 frozen)      |          training         |
| 1-layer TR 8AH tot 512 dim  | 59181698            |       21.96            |
| 1-layer TR 1AH tot 512 dim  | 59181698            |       18.47            |
| 1-layer TR 1AH 64 we 512   |   56425154         |         16.74          |
| 1-layer TR 2AH 64 we 512 (freeze only encoder ah)   |   22824514  (33994432 frozen)       |         50k steps 18.11 100k steps 18.42   150k steps 18.44 still training       |
| 1-layer TR 2AH 64 we 512 |   22726018  (34092928 frozen)       |         training       |






| Model                  | number of parameters     | BLEU newstest2014 |
| ---                    | ---                      |---                |
| 1-layer TR 2AH 64 we 512 from previous 1 AH (identity matrix (and generator to the previous one)) |   22824514  (33994432 frozen)       |     150k steps   3.91          |
| 1-layer TR 2AH 64 we 512 from previous 1 AH (xavier (as the default setting) (and generator to the previous one)) |   22824514  (33994432 frozen)       |      150k steps   9.44          |
| 1-layer TR 2AH 64 we 512 from previous 1 AH (identity matrix and generator to xavier) |   22824514  (33994432 frozen)       |         100k steps 3.77          |
| 1-layer TR 1AH 64 we 512 (init fasttext)   |   56425154         |       16.79            |
| 1-layer TR 1AH 64 we 512 (init and fixed fasttext)   |   56425154 - we        |         9.76          |
| 1-layer TR 1AH tot 512 dim  | 59181698            |       18.47            |
| 1-layer TR 8AH tot 512 dim  | 59181698            |       21.96            |
| 1-layer TR 1AH tot 64 dim and we 64  |  7087810            |       10.47            |
