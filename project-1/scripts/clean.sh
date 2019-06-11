MOSES=/home/mpsilfve/src/mosesdecoder-RELEASE-4.0/
TOKENIZER=$MOSES/scripts/tokenizer
RECASER=$MOSES/scripts/recaser
tgt_lang=$1

sed -E 's/(@@ )|(@@ ?$)//g' |
perl ${TOKENIZER}/replace-unicode-punctuation.perl |
perl ${TOKENIZER}/remove-non-printing-char.perl |
perl ${TOKENIZER}/normalize-punctuation.perl -l ${tgt_lang} |
perl ${RECASER}/detruecase.perl |
perl ${TOKENIZER}/detokenizer.perl -l ${tgt_lang} |
sed 's/\([0-9]\) - \([0-9|a-z]\)/\1-\2/g' |
sed 's# / #/#g' 
