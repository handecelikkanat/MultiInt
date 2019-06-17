BASE=/home/celikkan/Github/MultiInt/project-1/
ONMT=$BASE/opennmtsrc

MODEL_FILE=$BASE/models/transformer.model_step_200000.pt

SENTEVAL_TASKS=("bigram_shift" "coordination_inversion" "obj_number" "odd_man_out" "past_present" 
                "sentence_length" "subj_number" "top_constituents" "tree_depth" "word_content")

for TASK in ${SENTEVAL_TASKS[*]}; do
    BPE_FILE=$BASE/data/sample_bpe/$TASK.txt.bpe
    OUTPUT_FILE=$BASE/outputs/$TASK.tran
    REPRS_FILE=$BASE/outputs/$TASK.reprs.pickle
    
    cut -f4 $BPE_FILE > bpe_temp
    python3 $ONMT/translate.py -model $MODEL_FILE -src bpe_temp -output $OUTPUT_FILE -representations_file $REPRS_FILE -replace_unk -verbose -gpu 0
    
#parse the correct sentence from BPE_FILE=$BASE/data/sample_bpe/$TASK.txt.bpe and change with keys.
    # it will be a list of dictionaries.
done
