ONMT=/home/mpsilfve/src/OpenNMT-py-0.9.0
configfile_path=/home/mpsilfve/src/MultiInt/project-1/config
db=short
specs=filtered
src_lang=en
tgt_lang=fi
data_path=/home/mpsilfve/src/MultiInt/project-1/data
onmt_data_path=/home/mpsilfve/src/MultiInt/project-1/opennmtdata
savemodel=/home/mpsilfve/src/MultiInt/project-1/models

#export CUDA_VISIBLE_DEVICES=1

python  $ONMT/train.py -data $onmt_data_path/data-lstm -save_model $savemodel/lstm.model -layers 1 -rnn_size 512 -word_vec_size 512 -train_steps 200000  -dropout 0.1 -batch_size 256 -optim adam -learning_rate 0.001 -valid_steps 25000 -save_checkpoint_steps 25000 -report_every 50 -gpu_ranks 0
