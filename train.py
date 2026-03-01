import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config, latest_weights_file_path


# ---------------------------------------------------------------------------
# Greedy decoding (used for validation)
# ---------------------------------------------------------------------------

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt,
                  max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input  = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out          = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob         = model.projection(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt,
                   max_len, device, print_msg, global_step, writer,
                   num_examples=2):
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask  = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask,
                tokenizer_src, tokenizer_tgt, max_len, device
            )

            source_text    = batch["src_text"][0]
            target_text    = batch["target_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_msg('-' * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-' * console_width)
                break

    if writer:
        cer  = CharErrorRate()(predicted, expected)
        wer  = WordErrorRate()(predicted, expected)
        bleu = BLEUScore()(predicted, expected)
        writer.add_scalar('validation/cer',  cer,  global_step)
        writer.add_scalar('validation/wer',  wer,  global_step)
        writer.add_scalar('validation/bleu', bleu, global_step)
        writer.flush()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if not tokenizer_path.exists():
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


# ---------------------------------------------------------------------------
# Dataset + DataLoader
# ---------------------------------------------------------------------------

def get_ds(config):
    ds_raw = load_dataset(
        config["datasource"],
        f'{config["lang_src"]}-{config["lang_target"]}',
        split='train',
    )

    tokenizer_src    = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_target'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size   = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_target,
        config["lang_src"], config["lang_target"], config["seq_len"]
    )
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_target,
        config["lang_src"], config["lang_target"], config["seq_len"]
    )

    max_len_src = max_len_tgt = 0
    for item in ds_raw:
        s = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        t = tokenizer_target.encode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src, len(s))
        max_len_tgt = max(max_len_tgt, len(t))
    print(f"Max source length: {max_len_src}")
    print(f"Max target length: {max_len_tgt}")

    num_workers = config.get('num_workers', 4)
    use_persistent = num_workers > 0

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(config, vocab_src_len, vocab_target_len):
    return build_transformer(
        vocab_src_len, vocab_target_len,
        config['seq_len'], config['seq_len'],
        d_model=config['d_model'],
        N=config.get('N', 6),
        h=config.get('h', 8),
        dropout=config.get('dropout', 0.1),
        d_ff=config.get('d_ff', 2048),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    if torch.cuda.is_available():
        # Enable TF32 on A100/H100 for matmul and cuDNN (free ~10 % speed-up)
        torch.backends.cudnn.benchmark          = True
        torch.backends.cuda.matmul.allow_tf32   = True
        torch.backends.cudnn.allow_tf32         = True
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    Path(config['model_dir']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_target = get_ds(config)

    # ---- Build model -------------------------------------------------------
    model = get_model(
        config,
        len(tokenizer_src.get_vocab()),
        len(tokenizer_target.get_vocab()),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {total_params / 1e6:.1f} M")

    # ---- Mixed precision ---------------------------------------------------
    use_amp  = config.get('use_amp', True) and torch.cuda.is_available()
    bf16_ok  = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    # GradScaler is needed only with FP16 (BF16 has sufficient dynamic range)
    use_scaler = use_amp and not bf16_ok
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    except TypeError:                           # PyTorch < 2.4 API
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    if use_amp:
        print(f"AMP dtype  : {amp_dtype}  (scaler={'on' if use_scaler else 'off'})")

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )

    # ---- LR schedule: linear warmup → cosine decay to 10 % of peak ---------
    accum_steps          = config.get('gradient_accumulation_steps', 1)
    warmup_steps         = config.get('warmup_steps', 4000)
    total_training_steps = max(
        1, (config['num_epochs'] * len(train_dataloader)) // accum_steps
    )

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_training_steps - warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Checkpoint resume -------------------------------------------------
    initial_epoch = 0
    global_step   = 0
    preload       = config['preload']

    if preload == 'latest':
        model_filename = latest_weights_file_path(config)
    elif preload:
        model_filename = get_weights_file_path(config, preload)
    else:
        model_filename = None

    print(f"Checkpoint : {model_filename}")

    if model_filename:
        print(f"Preloading {model_filename} ...")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if 'scaler_state_dict' in state:
            scaler.load_state_dict(state['scaler_state_dict'])
    else:
        print('No checkpoint found — starting from scratch.')

    # ---- torch.compile (PyTorch 2.0+, skip on Colab if unsupported) --------
    if hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile() ...")
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile() skipped: {exc}")

    # ---- Loss --------------------------------------------------------------
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_target.token_to_id('[PAD]'),
        label_smoothing=0.1,
    ).to(device)

    # ---- Tensorboard -------------------------------------------------------
    tb_writer = SummaryWriter(log_dir=config['experiment_name'])

    # ========================================================================
    # Training loop
    # ========================================================================
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{config['num_epochs']}",
        )

        for batch_idx, batch in enumerate(batch_iterator):
            # Non-blocking transfer — works with pin_memory=True
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask  = batch['encoder_mask'].to(device, non_blocking=True)
            decoder_mask  = batch['decoder_mask'].to(device, non_blocking=True)
            label         = batch['label'].to(device, non_blocking=True)

            # Forward pass under autocast
            with torch.autocast(
                device_type='cuda' if torch.cuda.is_available() else 'cpu',
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                encoder_op = model.encode(encoder_input, encoder_mask)
                decoder_op = model.decode(encoder_op, encoder_mask, decoder_input, decoder_mask)
                logits     = model.projection(decoder_op)
                loss       = loss_fn(
                    logits.view(-1, tokenizer_target.get_vocab_size()),
                    label.view(-1),
                )

            # Backward — scale loss for gradient accumulation
            scaler.scale(loss / accum_steps).backward()

            # Optimizer step every accum_steps batches (or at end of epoch)
            is_last_batch = (batch_idx + 1) == len(train_dataloader)
            if (batch_idx + 1) % accum_steps == 0 or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                tb_writer.add_scalar('train/loss', loss.item(), global_step)
                tb_writer.add_scalar(
                    'train/lr', optimizer.param_groups[0]['lr'], global_step
                )

            batch_iterator.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr':   f'{optimizer.param_groups[0]["lr"]:.2e}',
            })

        # Validation at end of each epoch
        run_validation(
            model, val_dataloader,
            tokenizer_src, tokenizer_target,
            config['seq_len'], device,
            lambda msg: batch_iterator.write(msg),
            global_step, tb_writer,
        )

        # Save checkpoint — extract underlying module if compiled
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'global_step':          global_step,
        }, get_weights_file_path(config, str(epoch)))


if __name__ == "__main__":
    config = get_config()
    train_model(config)
