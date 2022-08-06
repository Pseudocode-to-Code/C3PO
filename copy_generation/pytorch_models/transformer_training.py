import argparse
import time
import os
from xml.dom import ValidationErr
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AdamW
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from dataloader import Vocabulary, TrainDataset, get_train_loader

parser = argparse.ArgumentParser(description="Transformer Training")
parser.add_argument('--non-copy', '-n', default=False, action='store_true', help='Train on non-copy dataset')
parser.add_argument('--resume', '-r', default = 0, help='Training resumption. Pass the epoch number from which to resume')
args = parser.parse_args()

model_type = 'transformer'

# Read CPY dataset
data = pd.read_pickle('../../data/CPY_dataset_new.pkl')

# Create pseudocode and code vocabularies
pseudo_voc = Vocabulary('pseudocode')
code_voc = Vocabulary('code')
source_col = ''
target_col = ''

if args.non_copy:
    source_col = 'pseudo_token'
    target_col = 'code_token_aug'
    pseudo_voc.build_vocabulary(data, source_col)
    code_voc.build_vocabulary(data, target_col)
    model_type += '_noncopy'
else:
    source_col = 'pseudo_gen_seq'
    target_col = 'code_gen_seq_aug'
    pseudo_voc.build_vocabulary(data, source_col)
    code_voc.build_vocabulary(data, target_col)
    model_type += '_copy'



# Model hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
num_epochs = 2
learning_rate = 0.001
batch_size = 128
weight_decay = 0.01
save_total_limit = num_epochs

for key, value in pseudo_voc.itos.items():
    if pseudo_voc.stoi[value] != key:
        raise ValidationErr('Pseudocode vocabulary error')

for key, value in code_voc.itos.items():
    if code_voc.stoi[value] != key:
        raise ValidationErr('Code vocabulary error')

print('Finished building vocabularies')


writer = SummaryWriter(f"runs/{model_type}") # CHANGE BASED ON CASE
step = 0

# Use the t5-small pretrained transformer

model_checkpoint = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.to(device)

# model_name = model_checkpoint.split("/")[-1]


# args = Seq2SeqTrainingArguments(
#     f"{model_name}-finetuned-pseudo-code",
#     evaluation_strategy="epoch",
#     learning_rate=learning_rate,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     weight_decay=weight_decay,
#     save_total_limit=save_total_limit,
#     num_train_epochs=num_epochs,
#     predict_with_generate=True,
#     push_to_hub=False,
# )

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


train_dataset = TrainDataset(data, source_col, target_col)
train_loader = get_train_loader(train_dataset, batch_size)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"\n\n[Epoch {epoch} / {num_epochs}] : {time.strftime('%Y-%m-%d %H:%M')} ")
    
    checkpoint = {"state_dict": model.state_dict(), 
                  "optimizer": optimizer.state_dict(),
                  "global_step": step,
                  "epoch": epoch
                }

    if not os.path.exists(f'./checkpoints/{model_type}'):
        os.makedirs(f'./checkpoints/{model_type}')

    torch.save(checkpoint, f'./checkpoints/{model_type}/{epoch}.tar')    

    for batch_idx, batch in enumerate(tqdm(train_loader, unit='batch')):

        optimizer.zero_grad()

        # Get input and targets and get to cuda
        inp_data = batch[0].to(dtype=torch.int64, device=device)
        target = batch[1].to(dtype=torch.int64, device=device)

        outputs = model(inp_data, labels=target)

        loss = outputs[0]
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

        if step % 100 == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss}')


torch.save(model.state_dict(), f'./checkpoints/{model_type}/complete_model.pth') #CHANGE BASED ON CASE
