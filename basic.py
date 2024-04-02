import torch
import torch.nn as nn
from torch import optim
from torchtext import data
# from torchtext.legacy import data
import time
import math
import os

from nltk.translate.bleu_score import corpus_bleu
#import fr_core_news_sm
#import en_core_web_sm
import fr_core_news_lg
import en_core_web_lg

import wandb
wandb.init(project="paral-project")


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, embed_dim, nhead, num_encoder_layers,
                 dropout, num_decoder_layers, dim_feedforward, maximum_sentence_len=200):
        super(Transformer, self).__init__()

        # get initial hyper-parameters
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward

        # add embedding layers
        self.token_embedding_encoder = nn.Embedding(num_embeddings=self.src_vocab, embedding_dim=self.embed_dim)
        self.token_embedding_decoder = nn.Embedding(num_embeddings=self.trg_vocab, embedding_dim=self.embed_dim)
        self.position_embedding_encoder = nn.Embedding(num_embeddings=maximum_sentence_len, embedding_dim=self.embed_dim)
        self.position_embedding_decoder = nn.Embedding(num_embeddings=maximum_sentence_len, embedding_dim=self.embed_dim)

        # Encoder-Decoder Transformer
        self.transformer = nn.Transformer(d_model=self.embed_dim, nhead=self.nhead,
                                          num_encoder_layers=self.num_encoder_layers,
                                          dropout=self.dropout, num_decoder_layers=self.num_decoder_layers,
                                          dim_feedforward=self.dim_feedforward)

        # output layer to predict next token
        self.decoder = nn.Linear(self.embed_dim, self.trg_vocab)
        self.drop_layer = nn.Dropout()

    def forward(self, src, tgr):
        # read shapes
        # src = src_seq_len x batch_size
        # tgr = targ_seq_len x batch_size
        src_seq_len, batch_size = src.shape
        targ_seq_len, _ = tgr.shape

        # create position input for encoder
        # src_position = src_seq_len x batch_size
        src_position = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, batch_size).to(device))

        # create position input for decoder
        # src_position = targ_seq_len x batch_size
        targ_position = (torch.arange(0, targ_seq_len).unsqueeze(1).expand(targ_seq_len, batch_size).to(device))

        # input embedding by merging token embedding with position embedding
        # embed_src = src_seq_len x batch_size x d_model
        # embed_src = targ_seq_len x batch_size x d_model
        embed_src = self.drop_layer((self.token_embedding_encoder(src) + self.position_embedding_encoder(src_position)))
        embed_tgr = self.drop_layer((self.token_embedding_decoder(tgr) + self.position_embedding_decoder(targ_position)))

        # create mask for source
        # mask_src = batch_size x src_seq_len
        mask_src = (src == SRC.vocab.stoi[SRC.pad_token]).transpose(0, 1).to(device)

        # create mask for target
        # mask_targ = targ_seq_len x targ_seq_len
        mask_targ = self.transformer.generate_square_subsequent_mask(targ_seq_len).to(device)

        # feed via transformer
        # output = targ_seq_len x batch_size x d_model
        output = self.transformer(embed_src, embed_tgr, src_key_padding_mask=mask_src, tgt_mask=mask_targ)

        # transform the output to match no of. tokens in target vocab
        # output = targ_seq_len x batch_size x targ_vocab_size
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


def train_func(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.SRC.to(device)
        trg = batch.TRG.to(device)
        optimizer.zero_grad()
        
        #output = [targ seq len-1, batch size, output dim]
        output = model(src, trg[:-1, :]) # for target, provide targ seq len-1 tokens for each sentence
        output = output.reshape(-1, output.shape[2])
        target = trg[1:].reshape(-1)

        # loss function works only 2d logits, 1d targets
        # so flatten the trg, output tensors. Ignore the <sos> token
        # target shape should be [(targ seq len - 1) * batch_size]
        # output shape should be [(targ seq len - 1) * batch_size, output_dim]
        loss = criterion(output, target)
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def inference(model, file_name, src_vocab, trg_vocab, attention = False, max_trg_len = 64, ep=0):
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
      fields=[('TRG', trg_vocab), ('SRC', src_vocab)])

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
            #src = [src len, batch size]
            #trg = [trg len, batch size]
            src = batch.SRC.to(device)
            trg = batch.TRG.to(device)
            batch_size = trg.shape[1]

            outputs = [trg_vocab.vocab.stoi["<sos>"]]
            for i in range(max_trg_len):
                trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
                output = model(src, trg_tensor)
                topv, topi = output[-1,0,:].topk(1)
                cur_decoded_token = topi.squeeze().detach()  # detach from history as input
                outputs.append(cur_decoded_token.item())

                if cur_decoded_token.item() == trg_vocab.vocab.stoi["<eos>"]:
                    break
            all_translated_trg_tokids.append(outputs[1:-1])
            all_gold_trg_tokids.append([ trg[idx, 0].item() for idx in range(1, trg.size(0)-1)])
    
    # convert token ids to token strs
    all_gold_text = []
    all_translated_text = []
    for i in range(len(all_gold_trg_tokids)): 
        all_gold_text.append([[trg_vocab.vocab.itos[idx] for idx in all_gold_trg_tokids[i]]])
        all_translated_text.append([trg_vocab.vocab.itos[idx] for idx in all_translated_trg_tokids[i]])
        
    corpus_bleu_score = corpus_bleu(all_gold_text, all_translated_text)
    
    if ep == 0 or ep == N_EPOCHS-1:
        print('Epoch', ep+1)
        print('all_gold_text:\n', all_gold_text[0], '\n')
        print('all_translated_text:\n', all_translated_text[0], '\n')

    return corpus_bleu_score


def evaluate(model, iterator, criterion, ep=0):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            src = batch.SRC.to(device)
            trg = batch.TRG.to(device)
            output = model(src, trg[:-1, :]) #turn off teacher forcing
            output_dim = output.shape[-1]

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            output = output.reshape(-1, output.shape[2])
            target = trg[1:].reshape(-1)

            loss = criterion(output, target)
            epoch_loss += loss.item()
            #break
        
    bleu = inference(model, "data/val_eng_fre.tsv", SRC, TRG, False, 64, ep)

    return epoch_loss / len(iterator) , bleu



# set the pseudo-random generator
manual_seed = 77
torch.manual_seed(manual_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
n_gpu = torch.cuda.device_count()
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

print(f"Number of training examples: {len(train.examples)}")
print(f"Number of validation examples: {len(val.examples)}")
print(f"Number of testing examples: {len(test.examples)}")

# print(vars(train.examples[0]))

TRG.build_vocab(train, min_freq=2)
SRC.build_vocab(train, min_freq=2)

print(f"Unique tokens in source (fr) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

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
BATCH = 64 # 32
N_EPOCHS = 2 # 15

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), # we pass in the datasets we want the iterator to draw data from
    batch_sizes=(BATCH, 256, 256), device = device,
    sort_key=lambda x: len(x.SRC), # the BucketIterator needs to be told what function it should use to group the data.
    sort_within_batch=False)


# model instance
model = Transformer(src_vocab, trg_vocab, embed_dim, nhead, num_encoder_layers, dropout, num_decoder_layers, dim_feedforward).to(device)

# set the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# create the loss function
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
print('<pad> token index: ', TRG_PAD_IDX)
## we will ignore the pad token in true target set
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


# initial best valid loss
best_valid_loss = float('inf')

path = "best"
path_2 = "checkpoints"

if not os.path.exists(path):
   os.makedirs(path)
   print(f"The new directory '{path}' is created!")

if not os.path.exists(path_2):
   os.makedirs(path_2)
   print(f"The new directory '{path_2}' is created!")

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


for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train_func(model, train_iter, optimizer, criterion)
    valid_loss, bleu = evaluate(model, val_iter, criterion, epoch)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # Create checkpoint at end of each epoch
    state_dict_model = model.state_dict() 
    state = {
        'epoch': epoch,
        'state_dict': state_dict_model,
        'optimizer': optimizer.state_dict()
        }
    torch.save(state, "checkpoints/model_"+str(N_EPOCHS)+"ep_"+str(BATCH)+"b_"+str(epoch+1)+".pt")
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best/model_"+str(N_EPOCHS)+"ep_"+str(BATCH)+"b.pth")
        print(f"\nEpoch {epoch+1}\nbest_valid_loss:\n {best_valid_loss:.3f}\n")
    
    wandb.log({"valid_bleu": bleu, "train_loss": train_loss, "valid_loss": valid_loss, 'epoch_mins':epoch_mins, 'epoch_secs':epoch_secs})
    print(f'Epoch: [{epoch+1}/{N_EPOCHS}] | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val.  Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f'\t Val. BLEU: {bleu:.3f}')


wandb.log({"best_valid_loss": best_valid_loss})
wandb.finish()
print("\nTon modèle est très bon!")
print("Et tu intelligence est vraiment étonnante!")
print("C'est la fin -_0")


