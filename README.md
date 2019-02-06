# Hierarchical Attention Networks

Pytorch implementation of [Hierarchical Attention Networks for Document Classification].(https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

<img src="https://github.com/SSinyu/Hierarchical-Attention-Network/blob/master/img/HAN_model.png" height="400">

## Requirements

- Pytorch, nltk, NumPy, pandas, matplotlib

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

```
python prep.py  # for text preprocessing 

python main.py  # for training

python main.py --mode='test' --test_iters=*** # for model test
```

## Result

<img src="https://github.com/SSinyu/Hierarchical-Attention-Network/blob/master/img/train_eval_loss.png" height="400">
