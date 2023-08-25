import code
import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm
import time

from models.vanilla_seq2seq import *
from models.attention_seq2seq import *
from dataloaders.dataloader import *

parser = argparse.ArgumentParser(description="Seq2Seq Training")
parser.add_argument('--attention', '-a', default=False, action="store_true", help='Train Attention model')
parser.add_argument('--non-copy', '-n', default=False, action='store_true', help='Train on non-copy dataset')
parser.add_argument('--checkpoint', '-c', default = 99, help='Training resumption. Pass the epoch number from which to resume')
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
train_data = pd.read_pickle('../../data/CPY_dataset.pkl')

pseudo_full_voc_train = Vocabulary('train pseudocode')
pseudo_copy_voc_train = Vocabulary('train pseudo with cpy')
pseudo_full_voc_eval = Vocabulary('eval pseudocode')
pseudo_copy_voc_eval = Vocabulary('eval pseudo with cpy')

pseudo_full_voc_train.build_vocabulary(train_data, 'pseudo_token')
pseudo_copy_voc_train.build_vocabulary(train_data, 'pseudo_gen_seq')

pseudo_full_voc_eval.build_vocabulary(eval_data, 'pseudo_token')
pseudo_copy_voc_eval.build_vocabulary(eval_data, 'pseudo_gen_seq')

if args.non_copy:
    # pseudo_voc.build_vocabulary(data, 'pseudo_token')
    pseudo_voc_size = len(pseudo_full_voc_train)
    model_type += '_noncopy'
else:
    # pseudo_voc.build_vocabulary(data, 'pseudo_gen_seq')
    pseudo_voc_size = len(pseudo_copy_voc_train)
    model_type += '_copy'
    

code_voc = Vocabulary('code')
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
    test_dataset = TestDataset(eval_data, 'pseudo_token', pseudo_full_voc_train)
else:
    test_dataset = TestDataset(eval_data, 'pseudo_gen_seq', pseudo_copy_voc_train) 

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
print(f'Loading checkpoint: {model_type}/{args.checkpoint}.tar')
resume_checkpoint = torch.load(f'./checkpoints/{model_type}/{args.checkpoint}.tar', map_location=device) # CHANGE BASED ON CASE
model.load_state_dict(resume_checkpoint['state_dict'])

if args.non_copy:
    pseudo_voc = pseudo_full_voc_train
else:
    pseudo_voc = pseudo_copy_voc_train

model.eval()

final_seqs = []
code_gen_seqs = []

for batch_idx, batch in enumerate(tqdm(test_loader, unit='lines')):
    inp_data = batch.permute(1,0).to(dtype=torch.int64, device=device) # Permute because model expects 1 column with all words indexes

    # TODO replace UNK tags if any with CPY tags
    # unks = torch.where(inp_data == pseudo_copy_voc_train.stoi['[UNK]'])[0]
    # if len(unks) > 0:
    #     inp_data[unks] = pseudo_copy_voc_train.stoi['[CPY]']

    unks = torch.where(inp_data == pseudo_voc.stoi['[UNK]'])[0]
    if len(unks) > 0:
        inp_data[unks] = pseudo_voc.stoi['[CPY]']
    

    with torch.no_grad():
        if args.attention:
            encoder_states, hidden, cell = model.encoder(inp_data)
        else:
            hidden, cell = model.encoder(inp_data)

    outputs = [code_voc.stoi["[START]"]] # Outputs including combine
    gen_seq = [code_voc.stoi["[START]"]] # Pure gen seq without combine, used for feeding previous word

    stop_condition = False

    copy_seq = eval_data['dt_copy_seq'][batch_idx]
    actual_pseudo = eval_data['pseudo_token'][batch_idx]
    # print(copy_seq, actual_pseudo)
    # print(actual_pseudo)

    cpy_indexes = np.where(copy_seq == 1)[0]
    # print(cpy_indexes)
    cpy_cnt = 0

    for _ in range(MAXLEN):
        previous_word = torch.LongTensor([gen_seq[-1]]).to(device)

        with torch.no_grad():
            if args.attention:
                output, hidden, cell = model.decoder(previous_word, encoder_states, hidden, cell)
            else:
                output, hidden, cell = model.decoder(previous_word, hidden, cell)

            best_guess = output.argmax(1).item()

        
        if not args.non_copy:
            if best_guess == code_voc.stoi['[CPY]']: #CPY tag generated
                if cpy_cnt < len(cpy_indexes): # If CPYs present in pseudo (Less than that generated)
                    index = cpy_indexes[cpy_cnt] # index in pseudo string
                    pseudo_token = actual_pseudo[index] 

                    token_index = pseudo_full_voc_eval.stoi[pseudo_token] 
                    outputs.append(-token_index)

                    cpy_cnt += 1
                
                else: # If more CPY tags generated do not add anything
                    pass
            else:
                outputs.append(best_guess)

        else:
            outputs.append(best_guess)

        gen_seq.append(best_guess)

        # Model predicts it's the end of the sentence
        # if output.argmax(1).item() == code_voc.stoi["[STOP]"] or len(outputs) > 50:
        if output.argmax(1).item() == code_voc.stoi["[STOP]"]:
            break

    gen_seq_conv = [code_voc.itos[index] for index in gen_seq]

    if args.non_copy:
        string_outputs = gen_seq_conv

    else:
        ### Convert to string
        string_outputs = []

        for token in outputs:
            if token >= 0:
                string_outputs.append(code_voc.itos[token])
            else:
                string_outputs.append(pseudo_full_voc_eval.itos[-token])

    final_seqs.append(string_outputs[1:-1])
    code_gen_seqs.append(gen_seq_conv[1:-1])


eval_data['code_gen_seq'] = code_gen_seqs
eval_data['final_code'] = final_seqs


pd.to_pickle(eval_data, f'./preds/{model_type}_{args.checkpoint}.pkl')

    # print('Pseudo', actual_pseudo)
    # print('Pseudo copy', eval_data['pseudo_gen_seq'][batch_idx])
    # print('Output copy', code_test)
    # print('Outputs', string_outputs)
    # print()


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
