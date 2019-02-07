# Hierarchical Attention Networks

Pytorch/Tensorflow implementation of [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).  
Model has a hierarchical structure that mirrors the hierarchical structure of documents, and consist of word-level encoder/attention layer, sentence-level encoder/attention layer.

<img src="https://github.com/SSinyu/Hierarchical-Attention-Network/blob/master/img/HAN_model.png" height="400">

## Requirements

- Pytorch of Tensorflow, nltk, NumPy, pandas, matplotlib

## Data

- Sample_text.zip (Sample_text.csv)
- Data consist of 100,000 reviews and stars.

class|text|
----|----|
4|"It was a great experience. They helped us ... "|
5|"Amazing service to use for removing junk ..."|
1|"Good little cafe in Matthews. I tried the ..."|
 ... | ... |

## Use

1. Download sample_text.zip an unzip
2. run `python prep.py` for text preprocessing
3. run `python pytorch_main.py` or `python tf_main.py` for traning (check argument)
4. run `python pytorch_main.py --mode='test' --test_iters=***`  
or `python tf_main.py --mode='test' --test_iters=***` for test

## Result

<img src="https://github.com/SSinyu/Hierarchical-Attention-Network/blob/master/img/train_eval_loss.png" height="500">
