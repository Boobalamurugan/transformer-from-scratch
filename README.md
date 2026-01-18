# Transformer from Scratch

A PyTorch implementation of the Transformer architecture for sequence-to-sequence translation tasks. This project builds a complete neural machine translation (NMT) system from scratch, implementing all core components of the Transformer model.

## Project Overview

This repository contains a full implementation of the Transformer architecture for English-Tamil translation, including:

- **Multi-head Self-Attention**: Core mechanism for learning relationships between tokens
- **Positional Encoding**: Encodes token positions in the sequence
- **Feed-Forward Networks**: Non-linear transformations in each layer
- **Encoder-Decoder Architecture**: Full seq2seq translation model
- **Tokenization**: WordLevel tokenizer with special tokens (SOS, EOS, PAD, UNK)
- **Training Pipeline**: Complete training loop with validation and checkpointing

## Project Structure

```
.
├── model.py              # Transformer architecture components
├── dataset.py            # Data loading and preprocessing
├── config.py             # Configuration management
├── train.py              # Training script
├── tokenizer_en.json     # English tokenizer
├── tokenizer_ta.json     # Tamil tokenizer
```

## Configuration

Key parameters in `config.py`:

- **batch_size**: 8 (samples per batch)
- **num_epochs**: 20 (training epochs)
- **lr**: 1e-4 (learning rate)
- **seq_len**: 128 (maximum sequence length)
- **d_model**: 256 (embedding dimension)
- **lang_src**: 'en' (source language - English)
- **lang_target**: 'ta' (target language - Tamil)
- **datasource**: 'Helsinki-NLP/opus-100' (dataset)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv sync
```

Required packages:
- torch
- tokenizers
- datasets
- tqdm
- tensorboard

## Usage

### Training

Start training the model:
```bash
python train.py
```

The training script will:
1. Load the OPUS100 English-Tamil dataset
2. Build/load tokenizers for both languages
3. Create train/validation splits (90/10)
4. Initialize the Transformer model
5. Train for the specified number of epochs
6. Save checkpoints after each epoch
7. Log metrics to TensorBoard

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir runs/tmodel
```

### Model Checkpoints

Trained model weights are saved in the `weights/` directory with names:
- `tmodel_0.pt` (epoch 0)
- `tmodel_1.pt` (epoch 1)
- etc.

## Model Architecture Details

### Transformer Components

1. **InputEmbedding**: Converts token IDs to embedding vectors
2. **PositionalEncoding**: Adds positional information to embeddings
3. **MultiHeadAttentionBlock**: Parallel attention heads for rich representations
4. **FeedForwardBlock**: Two linear layers with ReLU activation
5. **EncoderBlock**: Combines self-attention and feed-forward with residual connections
6. **DecoderBlock**: Combines self-attention, cross-attention, and feed-forward
7. **Encoder**: Stack of encoder blocks
8. **Decoder**: Stack of decoder blocks
9. **Transformer**: Complete encoder-decoder model with projection layer

### Key Hyperparameters

- **d_model**: 256 (embedding/hidden dimension)
- **num_heads**: 8 (number of attention heads)
- **d_ff**: 2048 (feed-forward hidden dimension)
- **num_layers**: 6 (encoder/decoder depth)
- **dropout**: 0.1 (regularization)

## Dataset

The project uses the **OPUS100** dataset from Hugging Face:
- Source: English (en)
- Target: Tamil (ta)
- Split: Training set

The dataset is automatically downloaded and preprocessed during the first training run.

## Features

- ✅ Complete Transformer implementation from scratch
- ✅ Bilingual tokenization (English-Tamil)
- ✅ Automatic dataset loading and preprocessing
- ✅ Mixed precision training support (CUDA)
- ✅ TensorBoard logging for monitoring
- ✅ Model checkpointing and resuming training
- ✅ Causal masking for decoder attention
- ✅ Padding masking for variable-length sequences

## Acknowledgements

This project is deeply inspired by the work from  
**hkproj / pytorch-transformer**.

The repository played a major role in my learning journey and helped me
understand the internal working of Transformers and Attention mechanisms.
This implementation is written independently from scratch, but the learning
and conceptual inspiration came from their excellent work.

Repository: https://github.com/hkproj/pytorch-transformer

## License

This project is provided for educational purposes.

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need" - Original Transformer paper
- Hugging Face Transformers Library Documentation
- PyTorch Documentation
