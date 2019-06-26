BASE=$HOME/Github/MultiInt/project-1/
ONMT=$BASE/opennmtsrc
SCRIPTS_DIR=$BASE/scripts

MODEL_FILE=$BASE/models/transformer.model_step_200000.pt

#SENTEVAL_TASKS=("bigram_shift" "coordination_inversion" "obj_number" "odd_man_out" "past_present" 
#                "sentence_length" "subj_number" "top_constituents" "tree_depth" "word_content")

SENTEVAL_TASKS=("subj_number")

for TASK in ${SENTEVAL_TASKS[*]}; do
    BPE_FILE=$BASE/data/sample_bpe/$TASK.txt.bpe
    OUTPUT_FILE=$BASE/outputs/$TASK.tran
    REPRS_FILE=$BASE/outputs/$TASK.reprs.pickle
    
    cut -f4 $BPE_FILE > bpe_temp
    python3 $ONMT/translate.py -model $MODEL_FILE -src bpe_temp -output $OUTPUT_FILE -representations_file $REPRS_FILE.list -replace_unk -verbose -gpu 0

    # put back the correct keys from the bpe file:
    # python3 $SCRIPTS_DIR/list2dict.py -list_file $REPRS_FILE.list -bpe_file $BPE_FILE -dict_file $REPRS_FILE
    
done
