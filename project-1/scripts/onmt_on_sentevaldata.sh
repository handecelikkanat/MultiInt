BASE=/Users/celikkan/Github/MultiInt/project-1/
ONMT=$BASE/opennmtsrc

MODEL_FILE=$BASE/models/transformer.model_step_200000.pt

SENTEVAL_TASKS=("bigram_shift" "coordination_inversion" "obj_number" "odd_man_out" "past_present" 
                "sentence_length" "subj_number" "top_constituents" "tree_depth" "word_content")

for TASK in ${SENTEVAL_TASKS[*]}; do
    BPE_FILE=$BASE/data/sample_bpe/$TASK.txt.bpe
    OUTPUT_FILE=$BASE/outputs/$TASK.tran
    REPRS_FILE=$BASE/outputs/$TASK.tran
    python3 $ONMT/translate.py -model $MODEL_FILE -src $BPE_FILE -output $OUTPUT_FILE -representations_file $REPRS_FILE -replace_unk -verbose
done
