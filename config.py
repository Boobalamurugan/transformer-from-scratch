from pathlib import Path

def get_config():
    return {
        # ------------------------------------------------------------------ #
        # Data                                                               #
        # ------------------------------------------------------------------ #
        'batch_size': 512,          # increased to utilize H100 properly
        'num_workers': 8,           # increase to 12 if CPU allows
        'seq_len': 256,
        'lang_src': 'en',
        'lang_target': 'ta',
        'datasource': 'Helsinki-NLP/opus-100',
        'tokenizer_file': 'tokenizer_{0}.json',

        # ------------------------------------------------------------------ #
        # Model architecture                                                 #
        # ------------------------------------------------------------------ #
        'd_model': 512,
        'N': 6,
        'h': 8,
        'd_ff': 2048,
        'dropout': 0.1,

        # ------------------------------------------------------------------ #
        # Training                                                           #
        # ------------------------------------------------------------------ #
        'num_epochs': 30,
        'lr': 3e-4,                        # scaled for larger batch
        'warmup_steps': 2000,              # slightly reduced
        'gradient_accumulation_steps': 1,  # IMPORTANT: remove accumulation
        'use_amp': True,                   # BF16 on H100

        # ------------------------------------------------------------------ #
        # Checkpointing / logging                                            #
        # ------------------------------------------------------------------ #
        'model_dir': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
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