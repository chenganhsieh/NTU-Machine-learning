#!/usr/bin/python
wget "https://www.dropbox.com/s/s4ks3s3s7486609/rnn_Acc0.7626666666666667.pkl?dl=1" -O rnn_Acc0.7626666666666667.pkl
wget "https://www.dropbox.com/s/hup5p4w8pedujte/word2vec2.model?dl=1" -O word2vec2.model
python3 testdata.py $1 $2 $3