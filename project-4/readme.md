(Ale’s current project, also Edinburgh people) Analyzing Transformer Heads. Pruning (Edinburgh) vs. growing (Ale) transformer heads to find an optimal number of non-redundant heads. Experiments on massive data on new CSC cluster.

### TODO
* inspect 6-layer TR 1AH 64 we 512 

### Benchmark data
* Train: *europarl, news-commentary, commoncrawl, rapid, paracrawl*  (allfiltered, shuffled, bpe35k from our wmt submission) 11555682 sentences (probably i will scale down to 2.5 M and re-run the training)
* Dev: *newstest2013*
* Test: *newstest2014*

## Training commands
python  $opennmt/train.py -data $trainData -save_model $savemodel -layers 1 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 1  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0


python  $opennmt/train.py -data $trainData -save_model $savemodel -layers 1 -rnn_size 512 -word_vec_size 512 -train_steps 200000  -dropout 0.1 -batch_size 256 -optim adam -learning_rate 0.001 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0

python  $opennmt/train.py -data $trainData -save_model $savemodel -layers 1 -rnn_size 64 -word_vec_size 64 -transformer_ff 2048 -heads 1  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0


attn = batch, ah, seq, seq <- individual attention value

context_original = batch, ah, seq, dim <- individual attention vector

visualization tool: https://github.com/jessevig/bertviz

## Scores

Testing last checkpoint (200k training steps).

BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.2.11


learning which AH turn off:


| Model                  | number of parameters     | BLEU newstest2014 |
| ---                    | ---                      |---                |
| 6-layer TR 8AH tot 512 (default)    |   95963778         |    50k 23.62 100k 24.83 150k 26.12 200k 25.91               |
| 6-layer TR 8AH tot 512 (DET)    |   102316728   (it opens only first and last layer)      |    50k 23.84  100k 25.20  150k 25.53   200k  26.24       |
| 2-layer TR 8AH tot 512 (default)    |   66538114         |   50k 22.75 100k 23.92 150k 24.52  200k  24.68         |
| 3-layer TR 8AH tot 512 (default)    |   73894530         |    50k 23.36 100k 24.93 150k 25.06  200k 25.46         |
| 2-layer enc 6-layer dec TR 8AH tot 512 (default)    |   83354242         |  50k 23.67 100k 24.52 150k 25.52 200k  25.93 |


| Model  Europarl               | number of parameters     | BLEU newstest2014 |
| ---                    | ---                      |---                |
| 6-layer enc 6-layer dec TR 8AH tot 512 (default)    |   84150790        | 50k  18.11 100k 19.04 150k 200k |
| 6-layer enc (DET off2) 6-layer dec TR 8AH tot 512   |   84206140    | 50k  100k  150k  200k |
| 6-layer enc 1-layer dec TR 8AH tot 512   |    63130630        | 50k 16.84 100k 17.39 150k 17.98 200k 17.93 |
| 6-layer enc (DET off2) 1-layer dec TR 8AH tot 512   |   63185980    | 50k 15.94 100k 16.87 150k 17.12 200k 16.98 |
| 6-layer enc (DET off1) 1-layer dec TR 8AH tot 512   |   69483580    | 50k 15.86 100k 16.78 150k 16.82 200k 16.56 |
| 2-layer enc 1-layer dec TR 8AH tot 512   |    50521094        | 50k 15.33 100k 15.86 150k 15.82 200k 16.04 |







from previous AH model (He distribution (and generator to the previous one))

| Model                  | number of parameters     | BLEU newstest2014 |
| ---                    | ---                      |---                |
| 1-layer LSTM           | 57860738                 |       21.71            |
| 6-layer TR 8AH tot 512 (default)    |   95963778         |    50k 23.62 100k 24.83 150k 26.12 200k 25.91               |
| 6-layer TR-BIG 16AH tot 1024 (default)    |   279972994         |   training               |
| 6-layer TR 8AH tot 512 (splitted ah)    |   95963778         |    26.00             |
| 6-layer TR 1AH masking 0    |   95963778         |          24.47         |
| 6-layer TR 1AH 64 we 512    |   79424514         |          24.04         |
| 6-layer TR 2AH 64 we 512   |   46709378   (35077888 frozen)      |          25.84         |
| 6-layer TR 3AH 64 we 512   |   47890178   (36259840 frozen)      |          26.45        |
| 6-layer TR 4AH 64 we 512   |   47298050   (39214720 frozen)      |          26.58        |
| 1-layer TR 8AH tot 512 dim  | 59181698            |       21.96            |
| 1-layer TR 1AH tot 512 dim  | 59181698            |       18.47            |
| 1-layer TR 1AH masking 0   |   59181698         |        16.90          |
| 1-layer TR 1AH 64 we 512   |   56425154         |         16.74          |
| 1-layer TR 2AH 64 we 512 |   22726018  (34092928 frozen)       |         18.30       |
| 1-layer TR 3AH 64 we 512 |   22922818  (34289920 frozen)       |         19.77      |
| 1-layer TR 4AH 64 we 512 |   22824130  (34782400 frozen)       |        20.12      |
| 1-layer TR 5AH 64 we 512 |   22922434 (35077888 frozen)    |      20.46        |


&nbsp;
&nbsp;
&nbsp;


other experiments:


&nbsp;
&nbsp;
&nbsp;


| Model                  | number of parameters     | BLEU newstest2014 |
| ---                    | ---                      |---                |
| 1-layer TR 2AH 64 we 512 (freeze only encoder ah)   |   22824514  (33994432 frozen)       |         50k steps 18.11 100k steps 18.42   150k steps 18.44 200k steps 18.77      |
| 1-layer TR 2AH 64 we 512 from previous 1 AH (identity matrix (and generator to the previous one)) |   22824514  (33994432 frozen)       |     150k steps   3.91          |
| 1-layer TR 2AH 64 we 512 from previous 1 AH (xavier (as the default setting) (and generator to the previous one)) |   22824514  (33994432 frozen)       |      150k steps   9.44          |
| 1-layer TR 2AH 64 we 512 from previous 1 AH (identity matrix and generator to xavier) |   22824514  (33994432 frozen)       |         100k steps 3.77          |
| 1-layer TR 1AH 64 we 512 (init fasttext)   |   56425154         |       16.79            |
| 1-layer TR 1AH 64 we 512 (init and fixed fasttext)   |   56425154 - we        |         9.76          |
| 1-layer TR 1AH tot 512 dim  | 59181698            |       18.47            |
| 1-layer TR 8AH tot 512 dim  | 59181698            |       21.96            |
| 1-layer TR 1AH tot 64 dim and we 64  |  7087810            |       10.47            |
