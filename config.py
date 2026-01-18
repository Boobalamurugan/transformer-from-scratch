from pathlib import Path

def get_config():
    return {
        'batch_size':8,
        'num_epochs':20,
        'lr':10**-4,
        'seq_len': 350,
        'd_model':512,
        'lang_src':'en',
        'datasource':'Helsinki-NLP/opus-100',
        'lang_target':'ta',
        'model_dir':'weights',
        'model_basename':'tmodel_',
        'preload':None,
        'tokenizer_file':'tokenizer_{0}.json',
        'experiment_name':'runs/tmodel'
    }

def get_weights_file_path(config,epoch:str):
    model_folder = config['model_dir']
    model_base_filename = config['model_basename']
    model_filename = f"{model_base_filename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_dir']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])