import code
import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm
import time

from vanilla_seq2seq import *
from attention_seq2seq import *
from dataloader import *

parser = argparse.ArgumentParser(description="Seq2Seq Training")
parser.add_argument('--attention', '-a', default=False, action="store_true", help='Train Attention model')
parser.add_argument('--non-copy', '-n', default=False, action='store_true', help='Train on non-copy dataset')
parser.add_argument('--resume', '-r', default = 0, help='Training resumption. Pass the epoch number from which to resume')
args = parser.parse_args()

model_type = ''

if args.attention:
    print('Activating Attention Models')
    Encoder = AttnEncoder
    Decoder = AttnDecoder
    S2SModel = AttnSeq2Seq
    model_type += 'attention_s2s'
else:
    print('Activating Vanilla Models')
    Encoder = S2SEncoder
    Decoder = S2SDecoder
    S2SModel = VanillaSeq2Seq
    model_type += 'vanilla_s2s'


# Training hyperparameters
num_epochs = 100
learning_rate = 0.001
#batch_size = 64
batch_size = 32

# f = open('../../data/CPY_dataset.pkl', 'rb')
# data = pickle.load(f)
# f.close()
# data

data = pd.read_pickle('../../data/CPY_dataset_new.pkl')

pseudo_voc = Vocabulary()
if args.non_copy:
    pseudo_voc.build_vocabulary(data, 'pseudo_token')
    model_type += '_noncopy'
else:
    pseudo_voc.build_vocabulary(data, 'pseudo_gen_seq')
    model_type += '_copy'
    

code_voc = Vocabulary()
if args.non_copy:
    code_voc.build_vocabulary(data, 'code_token_aug')
else:
    code_voc.build_vocabulary(data, 'code_gen_seq_aug')

# Model hyperparameters
load_model = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(pseudo_voc) 
input_size_decoder = len(code_voc)
output_size = len(code_voc)
encoder_embedding_size = 100 #300
decoder_embedding_size = 100 #300
hidden_size = 256 #1024  # Needs to be the same for both RNN's
num_layers = 1 # 2
enc_dropout = 0.5
dec_dropout = 0.5

print('Pseudo Vocab', input_size_encoder)
print('Code Voc', input_size_decoder)
print('Device', device)

# print('Pseudo Vocab', pseudo_voc.itos)
# print('\n\n\n')
# print('Code Vocab', code_voc.itos)

for key in pseudo_voc.itos:
    if pseudo_voc.stoi[pseudo_voc.itos[key]] != key:
        print('ERROR')

for key in code_voc.itos:
    if code_voc.stoi[code_voc.itos[key]] != key:
        print('ERROR')

print('PASSED')

writer = SummaryWriter(f"runs/{model_type}") # CHANGE BASED ON CASE
step = 0

if args.non_copy:
    train_dataset = TrainDataset(data, 'pseudo_token', 'code_token_aug')
else:
    train_dataset = TrainDataset(data, 'pseudo_gen_seq', 'code_gen_seq_aug') 

train_loader = get_train_loader(train_dataset, batch_size)

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = S2SModel(encoder_net, decoder_net, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = code_voc.stoi["[PAD]"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Loading checkpoint
if args.resume:
    start_epoch = args.resume

    print(f'Loading checkpoint: {model_type}/{args.resume}.tar')
    resume_checkpoint = torch.load(f'./checkpoints/{model_type}/{args.resume}.tar') # CHANGE BASED ON CASE
    model.load_state_dict(resume_checkpoint['state_dict'])
    optimizer.load_state_dict(resume_checkpoint['optimizer'])
    step = resume_checkpoint['global_step']
    start_epoch = resume_checkpoint['epoch']

    print('Starting from step', step)
else:
    start_epoch = 0


# Main Training loop
for epoch in range(start_epoch, num_epochs):
    print(f"\n\n[Epoch {epoch} / {num_epochs}] : {time.strftime('%Y-%m-%d %H:%M')} ")

    checkpoint = {"state_dict": model.state_dict(), 
                  "optimizer": optimizer.state_dict(),
                  "global_step": step,
                  "epoch": epoch
                }

    if not os.path.exists(f'./checkpoints/{model_type}'):
        os.makedirs(f'./checkpoints/{model_type}')

    torch.save(checkpoint, f'./checkpoints/{model_type}/{epoch}.tar') # CHANGE BASED ON CASE

    model.eval()

    #test_pseudo = "set [CPY] to [CPY]"
    #test_to_indices = [pseudo_voc.stoi[token] for token in test_pseudo.split()] 
    #sentence_tensor = torch.LongTensor(test_to_indices).unsqueeze(1).to(device)
    #with torch.no_grad():
    #    hidden, cell = model.encoder(sentence_tensor)

    #outputs = [pseudo_voc.stoi["[START]"]]
    #stop_condition = False
    #while not stop_condition:
    #    previous_word = torch.LongTensor([outputs[-1]]).to(device)

    #   with torch.no_grad():
    #        output, hidden, cell = model.decoder(previous_word, hidden, cell)
    #        best_guess = output.argmax(1).item()

    #    outputs.append(best_guess)

        # Model predicts it's the end of the sentence
    #    if output.argmax(1).item() == code_voc.stoi["[STOP]"] or len(outputs) > 50:
    #        break

    #code_test = [code_voc.itos[index] for index in outputs]
    #print(f"Translated example sentence: \n {code_test}")
    #print('\n\n\n')

    
    model.train()
    running_loss = 0.0

    # Training 
    for batch_idx, batch in enumerate(tqdm(train_loader, unit='batch')):
        # Get input and targets and get to cuda
        inp_data = batch[0].to(dtype=torch.int64, device=device)
        target = batch[1].to(dtype=torch.int64, device=device)

        # print('Input Data Size:', inp_data.size())
        # print('Target Data Size:', target.size())

        # Forward prop
        output = model(inp_data, target, output_size)

        # print(output.size())

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping

        # Before reshape: (max_seq_len, BATCH, output_vocab_size) # output_vocab_size is softmax prob for each token in vocab
        output = output[1:].reshape(-1, output.shape[2]) # Slice 1: to remove START 
        #After reshape it is (max_seq_len * BATCH, output_vocab_size) 

        # Before reshape: (max_seq_len, BATCH) # No third dimension as targets are class indices here NOT ONE-HOT VECTORS
        target = target[1:].reshape(-1) # Slice 1: to remove START. 
        # After reshape (max_seq_len * BATCH)

        optimizer.zero_grad()
        loss = criterion(output, target)
        
        running_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Clip to avoid exploding gradient issues
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    writer.add_scalar("Epoch loss", running_loss/len(train_loader), global_step = epoch) 
    running_loss = 0.0


torch.save(model.state_dict(), './checkpoints/{model_type}/attention_model.pth') #CHANGE BASED ON CASE
