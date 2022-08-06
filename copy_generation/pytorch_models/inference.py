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

# f = open('../../data/CPY_dataset.pkl', 'rb')
# data = pickle.load(f)
# f.close()
# data

MAXLEN = 74 # Got from experimentation.ipynb
eval_data = pd.read_pickle('../../data/CPY_dataset_eval_tree_copy.pkl')
train_data = pd.read_pickle('../../data/CPY_dataset_new.pkl')

pseudo_full_voc = Vocabulary()
pseudo_copy_voc = Vocabulary()
pseudo_full_voc_eval = Vocabulary()
pseudo_copy_voc_eval = Vocabulary()

pseudo_full_voc.build_vocabulary(train_data, 'pseudo_token')
pseudo_copy_voc.build_vocabulary(train_data, 'pseudo_gen_seq')

pseudo_full_voc_eval.build_vocabulary(eval_data, 'pseudo_token')
pseudo_copy_voc_eval.build_vocabulary(eval_data, 'pseudo_gen_seq')

if args.non_copy:
    # pseudo_voc.build_vocabulary(data, 'pseudo_token')
    pseudo_voc_size = len(pseudo_full_voc)
    model_type += '_noncopy'
else:
    # pseudo_voc.build_vocabulary(data, 'pseudo_gen_seq')
    pseudo_voc_size = len(pseudo_copy_voc)
    model_type += '_copy'
    

code_voc = Vocabulary()
if args.non_copy:
    code_voc.build_vocabulary(train_data, 'code_token_aug')
    # code_voc.build_vocabulary(train_data, 'truth_code_token_aug')
    # code_voc.build_vocabulary(train_data, 'truth_code_token_aug')
else:
    code_voc.build_vocabulary(train_data, 'code_gen_seq_aug')
    # code_voc.build_vocabulary(train_data, 'truth_code_gen_seq_aug')
    # code_voc.build_vocabulary(train_data, 'truth_code_gen_seq_aug')

# Model hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size_encoder = pseudo_voc_size
input_size_decoder = len(code_voc)
output_size = len(code_voc)

if args.attention:
    encoder_embedding_size = 100 
    decoder_embedding_size = 100 
    hidden_size = 256
else:
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024  

num_layers = 1 # 2
enc_dropout = 0.5
dec_dropout = 0.5

print('Pseudo Vocab', input_size_encoder)
print('Code Voc', input_size_decoder)
print('Device', device)

print('Hyperparams', encoder_embedding_size, decoder_embedding_size , hidden_size)

# print('Pseudo Vocab', pseudo_voc.itos)
# print('\n\n\n')
# print('Code Vocab', code_voc.itos)

# for key in pseudo_full_voc.itos:
#     if pseudo_full_voc.stoi[pseudo_full_voc.itos[key]] != key:
#         print('ERROR')

# for key in code_voc.itos:
#     if code_voc.stoi[code_voc.itos[key]] != key:
#         print('ERROR')

# print('PASSED')

step = 0

if args.non_copy:
    test_dataset = TestDataset(eval_data, 'pseudo_token', pseudo_full_voc)
else:
    test_dataset = TestDataset(eval_data, 'pseudo_gen_seq', pseudo_copy_voc) 

print(len(test_dataset.source_vocab.stoi))

test_loader = get_test_loader(test_dataset)
print('No. of samples', len(test_loader))

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

pad_idx = code_voc.stoi["[PAD]"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Loading checkpoint
print(f'Loading checkpoint: {model_type}/99.tar')
resume_checkpoint = torch.load(f'./checkpoints/{model_type}/99.tar') # CHANGE BASED ON CASE
model.load_state_dict(resume_checkpoint['state_dict'])

model.eval()

for batch_idx, batch in enumerate(tqdm(test_loader, unit='batch')):
    print(batch)
#     inp_data = batch.to(dtype=torch.int64, device=device)

#     with torch.no_grad():
#         hidden, cell = model.encoder(inp_data)

#     outputs = [pseudo_voc.stoi["[START]"]]
#     stop_condition = False
#     while not stop_condition:
#         previous_word = torch.LongTensor([outputs[-1]]).to(device)

#         with torch.no_grad():
#             output, hidden, cell = model.decoder(previous_word, hidden, cell)
#             best_guess = output.argmax(1).item()

#         outputs.append(best_guess)

#         # Model predicts it's the end of the sentence
#         if output.argmax(1).item() == code_voc.stoi["[STOP]"] or len(outputs) > 50:
#             break

#     code_test = [code_voc.itos[index] for index in outputs]



# test_pseudo = "set l to m"
# # test_pseudo = "input [CPY] and [CPY]"
# test_to_indices = [pseudo_voc.stoi[token] for token in test_pseudo.split()] 
# sentence_tensor = torch.LongTensor(test_to_indices).unsqueeze(1).to(device)
# with torch.no_grad():
#     hidden, cell = model.encoder(sentence_tensor)

# outputs = [pseudo_voc.stoi["[START]"]]
# stop_condition = False
# while not stop_condition:
#     previous_word = torch.LongTensor([outputs[-1]]).to(device)

#     with torch.no_grad():
#         output, hidden, cell = model.decoder(previous_word, hidden, cell)
#         best_guess = output.argmax(1).item()

#     outputs.append(best_guess)

#     # Model predicts it's the end of the sentence
#     if output.argmax(1).item() == code_voc.stoi["[STOP]"] or len(outputs) > 50:
#         break

# code_test = [code_voc.itos[index] for index in outputs]
# print(f"Translated example sentence: \n {code_test}")
# print('\n\n\n')
