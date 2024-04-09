# deepspeed --num_nodes=2 --deepspeed --include localhost:0,1 model_deepspeed.py
# deepspeed --num_gpus=2 model_deepspeed.py
# you can use only GPUs 0 and 1:
# deepspeed --include localhost:0,1

import deepspeed
from deepspeed.accelerator import get_accelerator

import torch
import torch.nn as nn
from torchtext import data
import time
import math
import os

from nltk.translate.bleu_score import corpus_bleu
import fr_core_news_lg
import en_core_web_lg

import wandb
wandb.init(project="paral-project")

local_rank = int(os.getenv("LOCAL_RANK", "0"))
print('local_rank:', local_rank)
device = (torch.device(get_accelerator().device_name(), local_rank) if (local_rank > -1)
              and get_accelerator().is_available() else torch.device("cpu"))
print(device)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, embed_dim, nhead, num_encoder_layers,
                 dropout, num_decoder_layers, dim_feedforward, maximum_sentence_len=200):
        super(Transformer, self).__init__()

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward

        self.token_embedding_encoder = nn.Embedding(num_embeddings=self.src_vocab, embedding_dim=self.embed_dim)
        self.token_embedding_decoder = nn.Embedding(num_embeddings=self.trg_vocab, embedding_dim=self.embed_dim)
        self.position_embedding_encoder = nn.Embedding(num_embeddings=maximum_sentence_len, embedding_dim=self.embed_dim)
        self.position_embedding_decoder = nn.Embedding(num_embeddings=maximum_sentence_len, embedding_dim=self.embed_dim)

        # Encoder-Decoder Transformer
        self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=self.nhead,
                                          num_encoder_layers=self.num_encoder_layers,
                                          dropout=self.dropout, num_decoder_layers=self.num_decoder_layers,
                                          dim_feedforward=self.dim_feedforward)

        self.decoder = nn.Linear(self.embed_dim, self.trg_vocab)
        self.drop_layer = nn.Dropout()

    def forward(self, src, tgr):
        src_seq_len, batch_size = src.shape
        targ_seq_len, _ = tgr.shape
        src_position = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, batch_size).to(device))
        targ_position = (torch.arange(0, targ_seq_len).unsqueeze(1).expand(targ_seq_len, batch_size).to(device))
        embed_src = self.drop_layer((self.token_embedding_encoder(src) + self.position_embedding_encoder(src_position)))
        embed_tgr = self.drop_layer((self.token_embedding_decoder(tgr) + self.position_embedding_decoder(targ_position)))

        mask_src = (src == SRC.vocab.stoi[SRC.pad_token]).transpose(0, 1).to(device)
        mask_targ = self.transformer.generate_square_subsequent_mask(targ_seq_len).to(device)

        output = self.transformer(embed_src, embed_tgr, src_key_padding_mask=mask_src, tgt_mask=mask_targ)

        output = self.decoder(output)

        return output


def tokenize_fr(text):
    """
    Tokenizes French text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def train_func(model, iterator):
    model.train()
    epoch_loss = 0
    se = []
    start_time = time.time()
    
    for i, batch in enumerate(iterator):
        src = batch.SRC.to(device)
        trg = batch.TRG.to(device)
        
        loss = model(src, trg[:-1, :])
        model.backward(loss.mean())
        model.step()

        epoch_loss += loss.mean().item()
        se.append(loss.mean().item())
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = len(iterator) / elapsed_time  # Throughput calculation
    se = torch.std(torch.tensor(se)) / math.sqrt(len(iterator))
    se = se.item()
    
    return epoch_loss / len(iterator), throughput, se


def inference_func(model, file_name, max_trg_len = 64, ep=0):
    '''
    Function for translation inference

    Input: 
    model: translation model;
    file_name: the directoy of test file that the first column is target reference, and the second column is source language;
    trg_vocab: Target torchtext Field
    attention: the model returns attention weights or not.
    max_trg_len: the maximal length of translation text (optinal), default = 64

    Output:
    Corpus BLEU score.
    '''
    test = data.TabularDataset(
      path=file_name, # the root directory where the data lies
      format='tsv',
      skip_header=True, # if your tsv file has a header, make sure to pass this to ensure it doesn't get proceesed as data!
      fields=[('TRG', TRG), ('SRC', SRC)])

    test_iter = data.Iterator(
      dataset = test, # we pass in the datasets we want the iterator to draw data from
      sort = False, 
      batch_size=1,
      sort_key=None,
      shuffle=False,
      sort_within_batch=False,
      device = device,
      train=False
    )
    model.eval()
    all_gold_trg_tokids = []
    all_translated_trg_tokids = []

    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            src = batch.SRC.to(device)
            trg = batch.TRG.to(device)

            outputs = [TRG.vocab.stoi["<sos>"]]
            for i in range(max_trg_len):
                trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
                output = model(src, trg_tensor)
                output = torch.softmax(output, dim=-1)
                topv, topi = output[-1,0,:].topk(1)
                cur_decoded_token = topi.squeeze().detach()  # detach from history as input
                outputs.append(cur_decoded_token.item())

                if cur_decoded_token.item() == TRG.vocab.stoi["<eos>"]:
                    break
            all_translated_trg_tokids.append(outputs[1:-1])
            all_gold_trg_tokids.append([trg[idx, 0].item() for idx in range(1, trg.size(0)-1)])
    
    # convert token ids to token strs
    all_gold_text = []
    all_translated_text = []
    for i in range(len(all_gold_trg_tokids)): 
        all_gold_text.append([[TRG.vocab.itos[idx] for idx in all_gold_trg_tokids[i]]])
        all_translated_text.append([TRG.vocab.itos[idx] for idx in all_translated_trg_tokids[i]])
        
    corpus_bleu_score = corpus_bleu(all_gold_text, all_translated_text)
    
    if ep == 0:
        print('all_gold_text:\n', all_gold_text[0], '\n')
        print('all_translated_text:\n', all_translated_text[0], '\n')

    return corpus_bleu_score


def evaluate(model, iterator, ep=0):
    model.eval()
    epoch_loss = 0
    
#     print(f'#{local_rank} - I am ready to eval!')
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.SRC.to(device)
            trg = batch.TRG.to(device)
            loss = model(src, trg[:-1, :])
            epoch_loss += loss.mean().item()
        
    bleu = inference_func(model, "data/val_eng_fre.tsv", 64, ep)
    
    return epoch_loss / len(iterator) , bleu



# set the pseudo-random generator
manual_seed = 77
torch.manual_seed(manual_seed)
n_gpu = torch.cuda.device_count()
print('n_gpu:', n_gpu)
if n_gpu > 0:
    torch.cuda.manual_seed(manual_seed)


spacy_fr = fr_core_news_lg.load() #fr_core_news_sm
spacy_en = en_core_web_lg.load() #en_core_web_sm
print(f"\n##### Data pipelines   'fr_core_news_lg' and 'en_core_web_lg' loaded.\n")

SRC = data.Field(tokenize = tokenize_fr,
            # init_token = '<sos>', # since initial encoder hidden state is always set to zero, the network can figure out that the time step is 0 and this token is optional
            eos_token = '<eos>',
            lower = True)
TRG = data.Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train, val, test = data.TabularDataset.splits(
    path='data/', train='train_eng_fre.tsv',validation='val_eng_fre.tsv', test='test_eng_fre.tsv',
    format='tsv', skip_header=True, fields=[('TRG', TRG), ('SRC', SRC)])

# print(f"Number of training examples: {len(train.examples)}")
# print(f"Number of validation examples: {len(val.examples)}")
# print(f"Number of testing examples: {len(test.examples)}")

# print(vars(train.examples[0]))

TRG.build_vocab(train, min_freq=2)
SRC.build_vocab(train, min_freq=2)

# print(f"Unique tokens in source (fr) vocabulary: {len(SRC.vocab)}")
# print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

# hyperparameters
src_vocab = len(SRC.vocab)
trg_vocab = len(TRG.vocab)
embed_dim = 2048 #512
nhead = 4
num_encoder_layers = 2
dropout = 0.1 
num_decoder_layers = 2
dim_feedforward = 2048
learning_rate = 1e-4
BATCH = 64 # 64
N_EPOCHS = 10 # 10
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
print('TRG_PAD_IDX', TRG_PAD_IDX)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), # we pass in the datasets we want the iterator to draw data from
    batch_sizes=(BATCH, 256, 256), device = device,
    sort_key=lambda x: len(x.SRC), # the BucketIterator needs to be told what function it should use to group the data.
    sort_within_batch=False)

# DeepSpeed configuration
config = {
#     "train_micro_batch_size_per_gpu": BATCH,
    "train_batch_size": BATCH,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": learning_rate
        }
    },
    "steps_per_print":1000,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": learning_rate,
            "warmup_num_steps": 1000
        }
    }
}

model = Transformer(src_vocab, trg_vocab, embed_dim, nhead, num_encoder_layers, dropout, num_decoder_layers, dim_feedforward).to(device)
# Initialize DeepSpeed
model, _, _, _ = deepspeed.initialize(model=model,
                                      model_parameters=model.parameters(),
                                      config=config)

print(f'\n##### Training started:\n')
print('src_vocab', src_vocab)
print('trg_vocab', trg_vocab)
print('embed_dim', embed_dim)
print('nhead', nhead)
print('num_encoder_layers', num_encoder_layers)
print('num_decoder_layers', num_decoder_layers)
print('dropout', dropout)
print('dim_feedforward', dim_feedforward)
print('learning_rate', learning_rate)
print('BATCH', BATCH)
print('N_EPOCHS', N_EPOCHS)

print(f"#{local_rank}: Start training...\n")
wandb.run.name = "Paral_"+str(BATCH)+"b_"+str(N_EPOCHS)+"ep_AdamW_"+str(local_rank)

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, throughput, se = train_func(model, train_iter)
    valid_loss, bleu = evaluate(model, val_iter, epoch)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    
#     torch.cuda.empty_cache()
    wandb.log({"valid_bleu": bleu, "train_loss": train_loss, "valid_loss": valid_loss, 'epoch_mins':epoch_mins, 'epoch_secs':epoch_secs, "throughput": throughput, "se": se})
    print(f'Epoch: [{epoch+1}/{N_EPOCHS}] | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val.  Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f'\t Val. BLEU: {bleu:.3f}')


bleu = inference_func(model, "data/test_eng_fre.tsv", 64, 1)
print(f'\t Test BLEU: {bleu:.3f}')
wandb.log({"test_bleu": bleu})
wandb.finish()
print("\nTon modèle est très bon!")
print("C'est la fin -_0")


