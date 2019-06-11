ONMT=/home/mpsilfve/src/OpenNMT-py-0.9.0
configfile_path=/home/mpsilfve/src/MultiInt/config
db=short
specs=filtered
src_lang=en
tgt_lang=fi
data_path=/home/mpsilfve/src/MultiInt/data
onmt_data_path=/home/mpsilfve/src/MultiInt/opennmtdata
savemodel=/home/mpsilfve/src/MultiInt/models

python3 $ONMT/train.py -data $onmt_data_path/data -save_model $savemodel/transformer.model -layers 1 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 1  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 200000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 25000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0
