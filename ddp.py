# CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 ddp.py


import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtext import data
import time
import math
import os

from nltk.translate.bleu_score import corpus_bleu
import fr_core_news_lg
import en_core_web_lg

is_wandb = False

if is_wandb:
    import wandb
    wandb.init(project="paral-project")


# DDP
WORLD_SIZE = torch.cuda.device_count()
local_rank = int(os.getenv("LOCAL_RANK", "0"))
dist.init_process_group("nccl", rank=local_rank, world_size=WORLD_SIZE)
device = torch.device("cuda:"+str(local_rank) if torch.cuda.is_available() else "cpu")
## DDP

print('local_rank:', local_rank)
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


def train_func(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    se = []
    start_time = time.time()
    
    for i, batch in enumerate(iterator):
        
        src = batch.SRC.to(device)
        trg = batch.TRG.to(device)
        optimizer.zero_grad()
        
        output = model(src, trg[:-1, :]) # for target, provide targ seq len-1 tokens for each sentence
        output = output.reshape(-1, output.shape[2])
        target = trg[1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
#         epoch_loss += loss.item()
        se.append(loss.item())
    
    epoch_loss = sum(se)
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


def evaluate(model, iterator, criterion, ep=0, is_test=False):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.SRC.to(device)
            trg = batch.TRG.to(device)
            output = model(src, trg[:-1, :]) #turn off teacher forcing
            output_dim = output.shape[-1]
            
            output = output.reshape(-1, output.shape[2])
            target = trg[1:].reshape(-1)

            loss = criterion(output, target)
            epoch_loss += loss.item()
        
    if is_test: bleu = inference_func(model, "data/test_eng_fre.tsv", 64, ep)
    else: bleu = inference_func(model, "data/val_eng_fre.tsv", 64, ep)
    
    return epoch_loss / len(iterator) , bleu


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

TRG.build_vocab(train, min_freq=2)
SRC.build_vocab(train, min_freq=2)

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

model = Transformer(src_vocab, trg_vocab, embed_dim, nhead, num_encoder_layers, dropout, num_decoder_layers, dim_feedforward).to(device)


# set the optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# create the loss function
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
# print('<pad> token index: ', TRG_PAD_IDX)
## we will ignore the pad token in true target set
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


model = DDP(model, device_ids=[local_rank], output_device=local_rank)


total_params = sum(p.numel() for p in model.parameters())

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
print(f"#{local_rank}| Number of parameters: {total_params}")

if is_wandb:
    wandb.run.name = "paral_"+str(BATCH)+"b_"+str(N_EPOCHS)+"ep_AdamW_"+str(local_rank)
    wandb.log({"num_param": total_params})

print(f"#{local_rank}| Start training...\n")

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, throughput, se = train_func(model, train_iter, optimizer, criterion)
    valid_loss, bleu = evaluate(model, val_iter, criterion, epoch)
    end_time = time.time()
    epoch_secs = end_time - start_time

    if is_wandb: wandb.log({"valid_bleu": bleu, "train_loss": train_loss, "valid_loss": valid_loss, 'epoch_secs':epoch_secs, "throughput": throughput, "se": se})
    print(f'#{local_rank}|Epoch: [{epoch+1}/{N_EPOCHS}] | Time: {epoch_secs:.3f}s')
    print(f'\t Train Loss: {train_loss:.3f} | Throughput: {throughput:7.3f}')
    print(f'\t Val.  Loss: {valid_loss:.3f} |         SE: {se:7.3f}')
    print(f'\t Val. BLEU: {bleu:.3f}')

# DDP
dist.destroy_process_group()
## DDP

test_loss, bleu = evaluate(model, test_iter, 1)
print(f'\t Test Loss: {test_loss:.3f}\t Test BLEU: {bleu:.3f}')
if is_wandb:
    wandb.log({"test_bleu": bleu, "test_loss": test_loss})
    wandb.finish()
print("\nTon modèle est très bon!")
print("C'est la fin -_0")

