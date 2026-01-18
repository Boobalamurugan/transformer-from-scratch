import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

from pathlib import Path
import os

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
# import warnings

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config, latest_weights_file_path




def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.projection(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["target_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not tokenizer_path.exists():
        ## CREATE DIRECTORY SAFELY
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer=trainer
        )

        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(config["datasource"],f'{config["lang_src"]}-{config["lang_target"]}',split='train')

    ## build tokenzier
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config,ds_raw,config['lang_target'])

    ## 90% for training , 10% vaildation
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw)- train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_target,config["lang_src"],config["lang_target"],config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_target,config["lang_src"],config["lang_target"],config["seq_len"])

    max_len_src = 0
    max_len_target =0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_src.encode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_target = max(max_len_target,len(target_ids))

    print(f"Max length of source sentence:{max_len_src}")
    print(f"Max length of target sentence:{max_len_target}")

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_target

def get_model(config, vocob_src_len,vocab_target_len):

    model = build_transformer(vocob_src_len,vocab_target_len,config['seq_len'],config['seq_len'],config["d_model"])
    return model

def train_model(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device:{device}")

    Path(config['model_dir']).mkdir(parents=True,exist_ok=True)

    train_dataloader,val_dataloader,tokenizer_src,tokenizer_target = get_ds(config)
    model = get_model(config,len(tokenizer_src.get_vocab()),len(tokenizer_target.get_vocab()))
    model = model.to(device)

    ## tensorboard setup
    tb_writer = SummaryWriter(log_dir=config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    print(model_filename)
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_target.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch,config['num_epochs']):
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dataloader,desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device) ## (batch_size,seq_len)
            decoder_input = batch['decoder_input'].to(device) ## (batch_size,seq_len)

            encoder_mask = batch['encoder_mask'].to(device) ## (batch_size,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) ## (batch_size,1,seq_len,seq_len)

            ## run the tensor through the model
            encoder_op = model.encode(encoder_input,encoder_mask) ## (batch_size,seq_len,d_model)
            decoder_op = model.decode(encoder_op,encoder_mask,decoder_input,decoder_mask) ## (batch_size,seq_len,d_model)
            logits = model.projection(decoder_op) ## (batch_size,seq_len,vocab_size)

            label = batch['label'].to(device) ## (batch_size,seq_len)
            
            ## (B,seq_len,vocab_size) -> (B*seq_len,vocab_size)
            loss = loss_fn(logits.view(-1,tokenizer_target.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({'loss':loss.item()})

            ## log the loss to tensorboard
            tb_writer.add_scalar('Training Loss',loss.item(),global_step)

            ## backpropagation
            loss.backward()

            ## update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_target, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, tb_writer)
        
        ## save the model weights after every epoch
        weights_file_path = get_weights_file_path(config,str(epoch))
        torch.save(
            {
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'global_step':global_step
            },weights_file_path
        )

if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
