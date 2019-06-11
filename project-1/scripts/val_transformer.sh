ONMT=/home/mpsilfve/src/OpenNMT-py-0.9.0
configfile_path=/home/mpsilfve/src/MultiInt/project-1/config
db=short
specs=filtered
src_lang=en
tgt_lang=fi
data_path=/home/mpsilfve/src/MultiInt/project-1/data
results_path=/home/mpsilfve/src/MultiInt/project-1/results
onmt_data_path=/home/mpsilfve/src/MultiInt/project-1/opennmtdata
savemodel=/home/mpsilfve/src/MultiInt/project-1/models

echo "python3 $ONMT/translate.py -src $data_path/dev.en-fi.filtered.en -output $results_path/dev.en-fi.filtered.en.sys -model $savemodel/transformer.model_step_200000.pt"
python3 $ONMT/translate.py -src $data_path/dev.en-fi.filtered.en -output $results_path/dev.en-fi.filtered.en.sys -model $savemodel/transformer.model_step_200000.pt
