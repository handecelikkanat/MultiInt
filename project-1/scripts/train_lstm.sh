# Requires $MULTIINT_PATH, $ONMT_PATH defined

configfile_path=$MULTIINT_PATH/config
db=short
specs=filtered
src_lang=en
tgt_lang=fi
data_path=$MULTIINT_PATH/data
onmt_data_path=$MULTIINT_PATH/opennmtdata
savemodel=$MULTIINT_PATH/models

#export CUDA_VISIBLE_DEVICES=1

python  $ONMT_PATH/train.py -data $onmt_data_path/data -save_model $savemodel/lstm.model -layers 1 -rnn_size 512 -word_vec_size 512 -train_steps 200000  -dropout 0.1 -batch_size 256 -optim adam -learning_rate 0.001 -valid_steps 25000 -save_checkpoint_steps 25000 -report_every 50 -gpu_ranks 0
