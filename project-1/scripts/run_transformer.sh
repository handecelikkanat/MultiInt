BASE=/Users/celikkan/Github/MultiInt/project-1/
ONMT=$BASE/opennmtsrc

model_file=$BASE/models/transformer.model_step_200000.pt
data_name=test.en-fi.filtered.en
data_file=$BASE/data/$data_name
output_file=$BASE/outputs/$data_name.tran

python3 $ONMT/translate.py -model $model_file -src $data_file -output $output_file -replace_unk -verbose

