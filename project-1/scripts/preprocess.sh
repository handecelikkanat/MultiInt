ONMT=/home/mpsilfve/src/OpenNMT-py-0.9.0
configfile_path=/home/mpsilfve/src/MultiInt/config
db=short
specs=filtered
src_lang=en
tgt_lang=fi
data_path=/home/mpsilfve/src/MultiInt/data
onmt_data_path=/home/mpsilfve/src/MultiInt/opennmtdata

python3 ${ONMT}/preprocess.py -save_config ${configfile_path}/preprocess_config.yml\
       -train_src ${data_path}/${db}.en-fi.${specs}.${src_lang} \
       -train_tgt ${data_path}/${db}.en-fi.${specs}.${tgt_lang} \
       -valid_src ${data_path}/dev.en-fi.${specs}.${src_lang}   \
       -valid_tgt ${data_path}/dev.en-fi.${specs}.${tgt_lang}   \
       -save_data ${onmt_data_path}/data \
       -data_type text      \
       -share_vocab 'true'  \
       -shard_size 1000000  \
       -src_seq_length 600 \
               -tgt_seq_length 600 

python3 ${ONMT}/preprocess.py -config ${configfile_path}/preprocess_config.yml
