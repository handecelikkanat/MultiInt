### Research Question
As we add more languages, do we get more abstract/informed representations? As a way of explicit measuring, does our model get better in probing tasks? This is one of the main hypotheses of FoTran. (We will start from this. Dataset and evaluation details below.)

### TODO
* train 1-layer 1-head transformer on English-Finnish
* try to access attention weights and visualize them
* apply SentEval probing tasks, specifically from Conneau et al. What you can cram into a single bleeping vector?

### Benchmark data
* WMT English-Finnish data. Which ones in specific?

## Training commands
python  $opennmt/train.py -data $trainData -save_model $savemodel -layers 1 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 1  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0

