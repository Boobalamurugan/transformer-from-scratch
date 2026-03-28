# Transformer from Scratch 🚀

A robust, from-scratch PyTorch implementation of the Transformer architecture for sequence-to-sequence translation tasks. This project builds a complete neural machine translation (NMT) system capable of translating **English to Tamil**, implementing all core components of the "Attention Is All You Need" paper with modern improvements.

## 🌟 Key Features

- **Complete Transformer Architecture:** Multi-Head Attention, Feed-Forward Networks, Positional Encoding, and Encoder-Decoder blocks.
- **Modern "Pre-Norm" Residuals:** Uses LayerNormalization *before* the attention and feed-forward sublayers for significantly more stable training.
- **Multi-Dataset Scalability:** Seamlessly trains on small scale (`opus-100`) or massive scale (`samanantar`), combining them dynamically via CLI.
- **Mixed Precision (AMP):** Utilizes `bfloat16`/`float16` and Tensor Cores for massive speedups on modern GPUs.
- **Environment Profiles:** Instantly switch between Colab Free-Tier configurations and High-End VM Multi-GPU configurations.
- **Rich Inference Visualizations:** Extracts and plots attention heatmaps and token probabilities during inference.

---

## 🛠️ Installation

1. **Create and activate a virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
uv sync
```

---

## 🚀 Usage & Training

The training script (`train.py`) is highly configurable via CLI arguments, allowing you to seamlessly move from a free Colab environment to a production GPU cluster.

### 1. Choosing Your Dataset
You can dynamically load and merge datasets using the `--dataset` flag:
- `--dataset opus`: Trains on **Helsinki-NLP/opus-100** (~227,000 pairs). Great for quick tests. (Default)
- `--dataset samanantar`: Trains on **ai4bharat/samanantar** (~5.26 Million pairs). Massive state-of-the-art Tamil dataset.
- `--dataset both`: Merges and shuffles both datasets for a combined corpus of **~5.5 Million pairs!**

### 2. Choosing Your Environment
- **Colab Free Tier (T4 GPU)**
  ```bash
  python train.py --dataset opus --colab
  ```
  *(Applies memory optimizations: Batch Size 16, Grad Accumulation 8, 2 Workers)*

- **Production High-End GPU (L40S, A100, H100)**
  ```bash
  python train.py --dataset both --prod
  ```
  *(Removes bottlenecks: Batch Size 128, TF32 Math, 8 Workers, max throughput)*

---

## 💻 Hardware Recommendations

If you are renting a GPU instance (e.g., RunPod, Lambda Labs) to train on the massive 5.5 Million pair `samanantar` dataset, here is the cost-to-performance breakdown for this ~60M parameter model:

* **🥇 The Sweet Spot: NVIDIA L40S (48GB)**
  * **Cost:** ~₹102.00/hr ($1.20/hr)
  * **Performance:** The L40S boasts the Ada Lovelace architecture and blisteringly fast `bfloat16` Tensor Cores. It matches or beats the A100 for small/dense transformers but costs nearly half the price.
* **⚡ Maximum Speed: NVIDIA A100 (40GB/80GB)**
  * **Cost:** ~₹180.00/hr ($2.00/hr)
  * **Performance:** If you need to hit the absolute fastest training time possible and budget is not a strict constraint, the A100's extreme memory bandwidth will crush standard batches.
* **⚠️ Overkill Warning: NVIDIA H100 / H200**
  * Do not rent these GPUs for this model. They are optimized for massively parallel multi-billion parameter LLMs. A 60M parameter Transformer will bottleneck your CPU/Data-Loader long before it saturates an H100, meaning you will overpay drastically for no speed benefit.

*(Note: Training 30 epochs on the 5.5M merged dataset will take 18+ hours. We recommend dropping your `num_epochs` configuration to `2` while running `--dataset both`, which will still yield a smarter model in a fraction of the time).*

---

## 🏗️ Model Architecture Details

### Transformer Components
1. **InputEmbedding:** Converts token IDs to embeddings, scaled by `sqrt(d_model)`.
2. **PositionalEncoding:** Sine/Cosine positional information.
3. **MultiHeadAttentionBlock:** Scaled dot-product attention with strict padding/causal masking.
4. **FeedForwardBlock:** Linear (2048) -> ReLU -> Dropout -> Linear (512).
5. **Residual Connections:** Implements Pre-Normalization `x + Dropout(Sublayer(LayerNorm(x)))`.

### Hyperparameters (Configurable in `config.py`)
- **d_model:** 512
- **num_heads:** 8
- **d_ff:** 2048
- **num_layers:** 6 (Encoder/Decoder depth)
- **dropout:** 0.1
- **lr:** 1e-4 with Linear Warmup + Cosine Decay
- **optimizer:** AdamW (`betas=(0.9, 0.98)`, `eps=1e-9`)

---

## 📊 Monitoring & Inference

### TensorBoard
Track metrics (Loss, Learning Rate, BLEU, Character Error Rate, and Word Error Rate) in real-time:
```bash
tensorboard --logdir runs/tmodel
```

### Visual Inference
To translate text and visualize the model's inner workings (Cross-Attention Heatmaps, Encoder Maps, and Token Probabilities):
```bash
python inference.py --model_path latest --query "Hello, how are you?"
```

---

## 🙏 Acknowledgements
This project is deeply inspired by the work from **hkproj / pytorch-transformer**. Their educational repository was instrumental in understanding the mechanics of Attention. This implementation is written independently from scratch but draws conceptual inspiration from their design.

## 📝 License
Provided for educational purposes.
