import code
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm
import time

from vanilla_transformer import Transformer
from dataloader_vanilla_trans import *

parser = argparse.ArgumentParser(description="Vanilla transformer inference")
parser.add_argument('--non-copy', '-n', default=False, action='store_true', help='Train on non-copy dataset')
parser.add_argument('--checkpoint', '-c', default=99, help='Checkpoint to test with')
args = parser.parse_args()

model_type = 'vanilla_transformer'

# Read CPY dataset
train_data = pd.read_pickle('../../data/CPY_dataset_new.pkl')

MAXLEN = 74 # Got from experimentation.ipynb
eval_data = pd.read_pickle('../../data/CPY_dataset_eval_tree_copy.pkl')

pseudo_voc = Vocabulary('pseudocode')

pseudo_full_voc_train = Vocabulary('train pseudocode')
pseudo_copy_voc_train = Vocabulary('train pseudo with cpy')
pseudo_full_voc_train.build_vocabulary(train_data, 'pseudo_token')
pseudo_copy_voc_train.build_vocabulary(train_data, 'pseudo_gen_seq')

if args.non_copy:
    # pseudo_full_voc_eval.build_vocabulary(eval_data, 'pseudo_token')
    pseudo_voc.build_vocabulary(eval_data, 'pseudo_token')
    pseudo_voc_size = len(pseudo_full_voc_train)
    model_type += '_noncopy'
else:
    # pseudo_full_voc_eval.build_vocabulary(eval_data, 'pseudo_gen_seq')
    pseudo_voc.build_vocabulary(eval_data, 'pseudo_gen_seq')
    pseudo_voc_size = len(pseudo_copy_voc_train)
    model_type += '_copy'


code_voc = Vocabulary('code')
if args.non_copy:
    code_voc.build_vocabulary(train_data, 'code_token_aug')
else:
    code_voc.build_vocabulary(train_data, 'code_gen_seq_aug')

# Model hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
src_vocab_size = pseudo_voc_size
trg_vocab_size = len(code_voc)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 87 # From experimentation.ipynb or get_max_len from generate.ipynb
forward_expansion = 4
src_pad_idx = code_voc.stoi[PAD_TOKEN]



print('Pseudo Vocab', len(pseudo_voc))
print('Code Voc', len(code_voc))
print('Device', device)
print('Hyperparams', embedding_size, forward_expansion, num_encoder_layers)

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

writer = SummaryWriter(f"runs/{model_type}") # CHANGE BASED ON CASE
step = 0

if args.non_copy:
    test_dataset = TestDataset(eval_data, 'pseudo_token', pseudo_voc)
else:
    test_dataset = TestDataset(eval_data, 'pseudo_gen_seq', pseudo_voc) 

print(len(test_dataset.source_vocab.stoi))

test_loader = get_test_loader(test_dataset)
print('No. of samples', len(test_loader))

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)


pad_idx = code_voc.stoi[PAD_TOKEN]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Loading checkpoint
print(f'Loading checkpoint: {model_type}/{args.checkpoint}.tar')
resume_checkpoint = torch.load(f'./checkpoints/{model_type}/{args.checkpoint}.tar', map_location=device) # CHANGE BASED ON CASE
model.load_state_dict(resume_checkpoint['state_dict'])


model.eval()

final_code = []
outputs = []

for batch_idx, batch in enumerate(tqdm(test_loader, unit='lines')):
    inp_data = batch.permute(1,0).to(dtype=torch.int64, device=device) # Permute because model expects 1 column with all words indexes

    if not args.non_copy:
        unks = torch.where(inp_data == UNKNOWN_INDEX)[0]
        if len(unks) > 0:
            inp_data[unks] = CPY_INDEX

    prev_output = [START_INDEX]
    output_replaced = []

    copy_seq = eval_data['dt_copy_seq'][batch_idx]
    actual_pseudo = eval_data['pseudo_token'][batch_idx]

    cpy_indexes = np.where(copy_seq == 1)[0]
    cpy_cnt = 0

    # unks = torch.where(inp_data == pseudo_voc.stoi['[UNK]'])[0]
    # if len(unks) > 0:
    #     inp_data[unks] = pseudo_voc.stoi['[CPY]']
    

    for _ in range(MAXLEN):

        with torch.no_grad():
            # Forward prop

            # Pad the target
            prev_output_tensor = torch.LongTensor(prev_output).unsqueeze(1).to(device)
            # print(prev_output_tensor)
            output = model(inp_data, prev_output_tensor)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            best_guess = output[-1,:].argmax().item()

        if not args.non_copy and best_guess == CPY_INDEX:
                if cpy_cnt < len(cpy_indexes): 
                    index = cpy_indexes[cpy_cnt]
                    pseudo_token = actual_pseudo[index]

                    token_index = pseudo_copy_voc_train.stoi[pseudo_token] 
                    output_replaced.append(-token_index)
                    prev_output.append(best_guess)
                    cpy_cnt += 1
                else: # If more CPY tags generated do not add anything
                    pass
        else:
            output_replaced.append(best_guess)
            prev_output.append(best_guess)

        if best_guess == STOP_INDEX:
            break
        


    if args.non_copy:
        string_outputs = [code_voc.itos[index] for index in prev_output]

    else:
        ### Convert to string
        string_outputs = []

        for token in output_replaced:
            if token >= 0:
                string_outputs.append(code_voc.itos[token])
            else:
                string_outputs.append(pseudo_copy_voc_train.itos[-token])

    outputs.append(prev_output[1:-1])
    final_code.append(string_outputs[1:-1])


print('Finished generating')
print('Output:', outputs[0])


# # test_pseudo = "set l to m"
# # test_pseudo = "input [CPY] and [CPY]"
# test_pseudo = "[CPY]"
# test_to_indices = [pseudo_copy_voc_train.stoi[token] for token in test_pseudo.split()] 
# sentence_tensor = torch.LongTensor(test_to_indices).unsqueeze(1).to(device)
# print(sentence_tensor.size())
# with torch.no_grad():
#     hidden, cell = model.encoder(sentence_tensor)

# outputs = [code_voc.stoi["[START]"]]
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

eval_data['outputs'] = outputs
eval_data['final_code'] = final_code

pd.to_pickle(eval_data, f'./preds/{model_type}_{args.checkpoint}.pkl')
