# Parallel Machine Translation using Transformer
# # French to English

Number of training examples: **29000**;
Number of validation examples: **1014**;
Number of testing examples: **1000**.

Requirements:
```
python=3.8

pip install torch==2.0.1 torchtext==0.5.0 spacy==3.7.4 nltk==3.5

python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_lg
```


Hyperparameters for ```basic.py```:

| Variable | Value |
| --- | --- |
| src_vocab | `6469` | 
| trg_vocab | `5893` |
| embed_dim | `2048` |
| nhead | `4` |
| num_encoder_layers | `2` |
| num_decoder_layers | `2` |
| dropout | `0.1` |
| dim_feedforward | `2048` |
| learning_rate | `1e-4` |
| BATCH | `4` |
| N_EPOCHS | `10` |

