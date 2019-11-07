
### Pickled attention score from encoder

the sentences come from the test part of the English PUD dataset from the UD project. The *.conllu* is the original format containing the UD annotations. The scores are generated from the *.bpe* file, so some words may be splitted by the bpe model.

the *HM.pdf* is the heatmap visualization of the attention scores from different models on one example. It is just to have an idea on what to expect. 

The *loadPickleAH.py* is a simple script to load the pickled self-attention scores from the model. The format of the pickled data is quite easy, it is a list of numpy arrays. Each index on the list correspond to the sentence in the same position on the *.bpe* file, so index 0 on the list is the first sentence on the text file. 
The shape of the numpy array is (attention heads, sentence length (bpe), sentence length (bpe)) 

10K sentences for sentence length 10,20,30,40,50 (each sentence contains bpe words long max 3 bpe units)
[bpe.tar.gz](https://helsinkifi-my.sharepoint.com/:u:/g/personal/raganato_ad_helsinki_fi/Ed32eN5HN8xOh45nGYLcneABdRcMhTpRnPwglQMCSzASzw?e=1N7Gfs) (9.24 GB)

10K sentences for sentence length 10,20,30,40,50 (each sentence contains full form words, the pickle with 50 sentence length contains only 3536 sentences)
[nobpe.tar.gz](https://helsinkifi-my.sharepoint.com/:u:/g/personal/raganato_ad_helsinki_fi/EYUdO-NQKORNjM2hKAvEHSgBUsJd2VWbCZHsCmWKb9GAOg?e=ssIs9Z) (6.53 GB)

