import code
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm

from vanilla_seq2seq import *
from attention_seq2seq import *
from dataloader import *

parser = argparse.ArgumentParser(description="Seq2Seq Training")
parser.add_argument('--attention', '-a', default=False, action="store_true", help='Train Attention model')
args = parser.parse_args()

if args.attention:
    print('Activating Attention Models')
    Encoder = AttnEncoder
    Decoder = AttnDecoder
    S2SModel = AttnSeq2Seq
else:
    print('Activating Vanilla Models')
    Encoder = S2SEncoder
    Decoder = S2SDecoder
    S2SModel = VanillaSeq2Seq


# Training hyperparameters
num_epochs = 1 #100
learning_rate = 0.001
batch_size = 32 # 64


f = open('../../data/CPY_dataset.pkl', 'rb')
data = pickle.load(f)
f.close()
data

pseudo_voc = Vocabulary()
pseudo_voc.build_vocabulary(data, 'pseudo_gen_seq')

code_voc = Vocabulary()
code_voc.build_vocabulary(data, 'code_gen_seq_aug')

# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(pseudo_voc) # +5 # TODO +5 is a hack. +5 because in dataloader stoi it is indexed with +5
input_size_decoder = len(code_voc) # +5
output_size = len(code_voc) # +5
encoder_embedding_size = 50 # 300
decoder_embedding_size = 50 # 300
hidden_size = 100 #1024  # Needs to be the same for both RNN's
num_layers = 1 # 2
enc_dropout = 0.5
dec_dropout = 0.5

print('Pseudo Vocab', input_size_encoder)
print('Code Voc', input_size_decoder)

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

writer = SummaryWriter(f"runs/loss_plot")
step = 0

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


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(), 
                  "optimizer": optimizer.state_dict()
                }

    torch.save(checkpoint, f'./checkpoints/lstm_seq2seq/{epoch}.tar') #TODO Change path

    # model.eval()

    # translated_sentence = translate_sentence(
    #     model, sentence, german, english, device, max_length=50
    # )

    # print(f"Translated example sentence: \n {translated_sentence}")

    model.train()


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

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Clip to avoid exploding gradient issues
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

