from pathlib import Path


def get_config():
    return {
        # ------------------------------------------------------------------ #
        # Data                                                                 #
        # ------------------------------------------------------------------ #
        'batch_size': 64,           # per-GPU batch; effective = batch × accum_steps
        'num_workers': 8,           # DataLoader workers (set 2 on Colab free tier)
        'seq_len': 350,
        'lang_src': 'en',
        'lang_target': 'ta',
        'datasource': 'Helsinki-NLP/opus-100',
        'tokenizer_file': 'tokenizer_{0}.json',

        # ------------------------------------------------------------------ #
        # Model architecture                                                   #
        # ------------------------------------------------------------------ #
        'd_model': 512,
        'N': 6,             # encoder/decoder layers  (paper "base" = 6)
        'h': 8,             # attention heads          (paper "base" = 8)
        'd_ff': 2048,       # feedforward hidden dim
        'dropout': 0.1,

        # ------------------------------------------------------------------ #
        # Training                                                             #
        # ------------------------------------------------------------------ #
        'num_epochs': 30,
        'lr': 1e-4,                         # peak LR after warmup
        'warmup_steps': 4000,               # linear warmup length
        'gradient_accumulation_steps': 4,   # effective batch = 64 × 4 = 256
        'use_amp': True,                    # BF16 on A100/H100, FP16 otherwise

        # ------------------------------------------------------------------ #
        # Checkpointing / logging                                              #
        # ------------------------------------------------------------------ #
        'model_dir': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,                    # None | 'latest' | '<epoch_str>'
        'experiment_name': 'runs/tmodel',
    }


def get_weights_file_path(config, epoch: str) -> str:
    model_folder = config['model_dir']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = config['model_dir']
    weights_files = list(Path(model_folder).glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        return None
    weights_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(weights_files[-1])
