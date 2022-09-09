import code
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
import pandas as pd
import argparse 
import os
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description="Seq2Seq Training")
parser.add_argument('--non-copy', '-n', default=False, action='store_true', help='Train on non-copy dataset')
parser.add_argument('--resume', '-r', default = 0, help='Training resumption. Pass the epoch number from which to resume')
args = parser.parse_args()

class Transformer(nn.Module):
    def __init__(
        self,
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
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

if __name__ == "__main__":

    data = pd.read_pickle('../../data/CPY_dataset_new.pkl')

    model_type = 'vanilla_transformer'

    pseudo_voc = Vocabulary('pseudocode')
    if args.non_copy:
        pseudo_voc.build_vocabulary(data, 'pseudo_token')
        model_type += '_noncopy'
    else:
        pseudo_voc.build_vocabulary(data, 'pseudo_gen_seq')
        model_type += '_copy'
        

    code_voc = Vocabulary('code')
    if args.non_copy:
        code_voc.build_vocabulary(data, 'code_token_aug')
    else:
        code_voc.build_vocabulary(data, 'code_gen_seq_aug')


    # We're ready to define everything we need for training our Seq2Seq model
    load_model = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training hyperparameters
    num_epochs = 100
    learning_rate = 0.001 #3e-4
    batch_size = 32

    # Model hyperparameters
    src_vocab_size = len(pseudo_voc)
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
    print('Hyperparams', batch_size, embedding_size, forward_expansion, num_encoder_layers)

    # Tensorboard to get nice loss plot
    writer = SummaryWriter(f"runs/{model_type}") # CHANGE BASED ON CASE
    step = 0

    if args.non_copy:
        train_dataset = TrainDataset(data, 'pseudo_token', 'code_token_aug')
    else:
        train_dataset = TrainDataset(data, 'pseudo_gen_seq', 'code_gen_seq_aug') 

    print(len(train_dataset.source_vocab.stoi))

    train_loader = get_train_loader(train_dataset, batch_size)

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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.1, patience=10, verbose=True
    # )

    pad_idx = code_voc.stoi[PAD_TOKEN]
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



    # Main training loop
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
        # Test


        model.train()
        running_loss = 0.0
        losses = []

        for batch_idx, batch in enumerate(tqdm(train_loader, unit='batch')):
            # Get input and targets and get to cuda
            inp_data = batch[0].to(dtype=torch.int64, device=device)
            target = batch[1].to(dtype=torch.int64, device=device)

            # Forward prop
            output = model(inp_data, target[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = criterion(output, target)
            running_loss += loss.item()

            # losses.append(loss.item())

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        # Per sample loss - divide total batch average loss by number of batches
        writer.add_scalar("Epoch loss", running_loss/len(train_loader), global_step = epoch) 
        running_loss = 0.0

        # mean_loss = sum(losses) / len(losses)
        # scheduler.step(mean_loss)

    torch.save(model.state_dict(), f'./checkpoints/{model_type}/attention_model.pth') #CHANGE BASED ON CASE