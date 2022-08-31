import code
import argparse
import os

import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast
import pickle
from tqdm import tqdm
import time

from vanilla_seq2seq import *
from attention_seq2seq import *
from transformer_dataloader import *

parser = argparse.ArgumentParser(description="Seq2Seq Training")
parser.add_argument('--non-copy', '-n', default=False, action='store_true', help='Train on non-copy dataset')
parser.add_argument('--checkpoint', '-c', default = 9, help='Training resumption. Pass the epoch number from which to resume')
args = parser.parse_args()

model_type = 'transformer_fixed_tokenizer'

# Read CPY dataset
data = pd.read_pickle('../../data/CPY_dataset_new.pkl')

MAXLEN = 74 # Got from experimentation.ipynb
eval_data = pd.read_pickle('../../data/CPY_dataset_eval_tree_copy.pkl')

# pseudo_full_voc = None
# pseudo_copy_voc = None

# try:
#     # Read vocabulary from pickle file
#     with open('../../data/vocabs/pseudo_full.pkl', 'rb') as f:
#         pseudo_full_voc = pickle.load(f)

#     with open('../../data/vocabs/pseudo_copy.pkl', 'rb') as f:
#         pseudo_copy_voc = pickle.load(f)

# except FileNotFoundError:
#     eval_data = pd.read_pickle('../../data/CPY_dataset_eval_tree_copy.pkl')
#     train_data = pd.read_pickle('../../data/CPY_dataset_new.pkl')

#     pseudo_full_voc = Vocabulary('train pseudocode')
#     pseudo_copy_voc = Vocabulary('train pseudo with cpy')

#     pseudo_full_voc.build_vocabulary(train_data, 'pseudo_token')
#     pseudo_copy_voc.build_vocabulary(train_data, 'pseudo_gen_seq')


    
#     # Save vocabularies to pickle
#     if not os.path.exists('../../data/vocabs'):
#         os.makedirs('../../data/vocabs')

#     with open('../../data/vocabs/pseudo_full.pkl', 'wb') as f:
#         pickle.dump(pseudo_full_voc, f)

#     with open('../../data/vocabs/pseudo_copy.pkl', 'wb') as f:
#         pickle.dump(pseudo_copy_voc, f)

# if args.non_copy:
#     pseudo_voc_size = len(pseudo_full_voc)
#     model_type += '_noncopy'
# else:
#     pseudo_voc_size = len(pseudo_copy_voc)
#     model_type += '_copy'

# code_voc = None

# if args.non_copy:
#     try:
#         # Read vocabulary from pickle file
#         with open('../../data/vocabs/code_voc_full.pkl', 'rb') as f:
#             code_voc = pickle.load(f)

#     except FileNotFoundError:
#         code_voc = Vocabulary('code_full')
#         code_voc.build_vocabulary(train_data, 'code_token_aug')

#         # Save vocabularies to pickle
#         with open('../../data/vocabs/code_voc_full.pkl', 'wb') as f:
#             pickle.dump(code_voc, f)

# else:
#     try:
#         # Read vocabulary from pickle file
#         with open('../../data/vocabs/code_voc_copy.pkl', 'rb') as f:
#             code_voc = pickle.load(f)
#     except FileNotFoundError:
#         code_voc = Vocabulary('code_copy')
#         code_voc.build_vocabulary(train_data, 'code_gen_seq_aug')

#         # Save vocabularies to pickle
#         with open('../../data/vocabs/code_voc_copy.pkl', 'wb') as f:
#             pickle.dump(code_voc, f)

# Model hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# input_size_encoder = pseudo_voc_size
# input_size_decoder = len(code_voc)
# output_size = len(code_voc)

batch_size = 8
weight_decay = 0.01


step = 0

if args.non_copy:
    tokenizer_name = './codet5-small_noncopy'
    tokenizer = RobertaTokenizerFast.from_pretrained(f"./models/{tokenizer_name}/")
    test_dataset = TestDataset(eval_data, 'pseudo_token', tokenizer)
    
else:
    tokenizer_name = './codet5-small_copy'
    tokenizer = RobertaTokenizerFast.from_pretrained(f"./models/{tokenizer_name}/")
    test_dataset = TestDataset(eval_data, 'pseudo_gen_seq', tokenizer) 


model_checkpoint = "./Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
model.resize_token_embeddings(len(tokenizer))
model.to(device)


test_loader = get_test_loader(test_dataset)
print('No. of samples', len(test_loader))

# pad_idx = code_voc.stoi["[PAD]"]
# criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Loading checkpoint
print(f'Loading checkpoint: {model_type}/{args.checkpoint}.tar')
resume_checkpoint = torch.load(f'./checkpoints/{model_type}/{args.checkpoint}.tar', map_location=device) # CHANGE BASED ON CASE
model.load_state_dict(resume_checkpoint['state_dict'])

# if args.non_copy:
#     pseudo_voc = pseudo_full_voc
# else:
#     pseudo_voc = pseudo_copy_voc

model.eval()

final_seqs = []
code_gen_seqs = []

for batch_idx, batch in enumerate(tqdm(test_loader, unit='batch')):
        # Get input and targets and get to cuda
        inp_data = batch.to(dtype=torch.int64, device=device)
        # print(inp_data)

        unks = torch.where(inp_data == tokenizer.encode('[UNK]')[0])
        print('unks', unks)

        if len(unks) > 0 and not args.non_copy:
            inp_data[unks] = tokenizer.encode('[CPY]')[0]

        outputs = model.generate(input_ids=inp_data)
        print(outputs)
        
        # gen_seq = [code_voc.stoi["[START]"]] 
    

        # # Get the actual tokens that were converted to CPY
        # copy_seq = eval_data['dt_copy_seq'][batch_idx]
        # actual_pseudo = eval_data['pseudo_token'][batch_idx]
        # cpy_indexes = np.where(copy_seq == 1)[0]

        # # Number of CPY tags encountered
        # cpy_cnt = 0

        # for i, token_idx in enumerate(outputs):
            
        #     if not args.non_copy:
        #         if token_idx == tokenizer('[CPY]').input_ids[0]:
        #             actual_token_index = cpy_indexes[cpy_cnt]
        #             pseudo_token = actual_pseudo[actual_token_index]

        #             outputs[i] = pseudo_token


        #             cpy_cnt += 1

        #         else:
        #             gen_seq.append(token_idx)

        #     else:
        #         gen_seq.append(token_idx)


                
                    


# for batch_idx, batch in enumerate(tqdm(test_loader, unit='lines')):
#     inp_data = batch.permute(1,0).to(dtype=torch.int64, device=device) # Permute because model expects 1 column with all words indexes

#     # TODO replace UNK tags if any with CPY tags
#     # unks = torch.where(inp_data == pseudo_copy_voc_train.stoi['[UNK]'])[0]
#     # if len(unks) > 0:
#     #     inp_data[unks] = pseudo_copy_voc_train.stoi['[CPY]']

#     unks = torch.where(inp_data == pseudo_voc.stoi['[UNK]'])[0]
#     if len(unks) > 0:
#         inp_data[unks] = pseudo_voc.stoi['[CPY]']
    

#     with torch.no_grad():
#         if args.attention:
#             encoder_states, hidden, cell = model.encoder(inp_data)
#         else:
#             hidden, cell = model.encoder(inp_data)

#     outputs = [code_voc.stoi["[START]"]] # Outputs including combine
#     gen_seq = [code_voc.stoi["[START]"]] # Pure gen seq without combine, used for feeding previous word

#     stop_condition = False

#     copy_seq = eval_data['dt_copy_seq'][batch_idx]
#     actual_pseudo = eval_data['pseudo_token'][batch_idx]
#     # print(copy_seq, actual_pseudo)
#     # print(actual_pseudo)

#     cpy_indexes = np.where(copy_seq == 1)[0]
#     # print(cpy_indexes)
#     cpy_cnt = 0

#     for _ in range(MAXLEN):
#         previous_word = torch.LongTensor([gen_seq[-1]]).to(device)

#         with torch.no_grad():
#             if args.attention:
#                 output, hidden, cell = model.decoder(previous_word, encoder_states, hidden, cell)
#             else:
#                 output, hidden, cell = model.decoder(previous_word, hidden, cell)

#             best_guess = output.argmax(1).item()

        
#         if not args.non_copy:
#             if best_guess == code_voc.stoi['[CPY]']: #CPY tag generated
#                 if cpy_cnt < len(cpy_indexes): # If CPYs present in pseudo (Less than that generated)
#                     index = cpy_indexes[cpy_cnt] # index in pseudo string
#                     pseudo_token = actual_pseudo[index] 

#                     token_index = pseudo_full_voc_eval.stoi[pseudo_token] 
#                     outputs.append(-token_index)

#                     cpy_cnt += 1
                
#                 else: # If more CPY tags generated do not add anything
#                     pass
#             else:
#                 outputs.append(best_guess)

#         else:
#             outputs.append(best_guess)

#         gen_seq.append(best_guess)

#         # Model predicts it's the end of the sentence
#         # if output.argmax(1).item() == code_voc.stoi["[STOP]"] or len(outputs) > 50:
#         if output.argmax(1).item() == code_voc.stoi["[STOP]"]:
#             break

#     gen_seq_conv = [code_voc.itos[index] for index in gen_seq]

#     if args.non_copy:
#         string_outputs = gen_seq_conv

#     else:
#         ### Convert to string
#         string_outputs = []

#         for token in outputs:
#             if token >= 0:
#                 string_outputs.append(code_voc.itos[token])
#             else:
#                 string_outputs.append(pseudo_full_voc_eval.itos[-token])

#     final_seqs.append(string_outputs[1:-1])
#     code_gen_seqs.append(gen_seq_conv[1:-1])


# eval_data['code_gen_seq'] = code_gen_seqs
# eval_data['final_code'] = final_seqs


# pd.to_pickle(eval_data, f'./preds/{model_type}_{args.checkpoint}.pkl')

   
