import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path
from model import build_transformer
from dataset import causal_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EN→TA inference with attention visualization"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to .pt checkpoint, or 'latest' to auto-find newest checkpoint"
    )
    parser.add_argument(
        "--query", type=str, required=True,
        help="English sentence to translate"
    )
    parser.add_argument(
        "--save_dir", type=str, default="inference_output",
        help="Directory to save PNG plots (default: inference_output/)"
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Top-K tokens shown in generation probability heatmap (default: 10)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tokenizer loading (no dataset download required)
# ---------------------------------------------------------------------------

def load_tokenizers(config: dict) -> tuple:
    src_path = Path(config["tokenizer_file"].format(config["lang_src"]))
    tgt_path = Path(config["tokenizer_file"].format(config["lang_target"]))

    if not src_path.exists():
        raise FileNotFoundError(
            f"Source tokenizer not found at '{src_path}'. "
            "Run training first to generate tokenizer files."
        )
    if not tgt_path.exists():
        raise FileNotFoundError(
            f"Target tokenizer not found at '{tgt_path}'. "
            "Run training first to generate tokenizer files."
        )

    tokenizer_src = Tokenizer.from_file(str(src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tgt_path))
    return tokenizer_src, tokenizer_tgt


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, config: dict, tokenizer_src, tokenizer_tgt,
               device: torch.device):
    if model_path == "latest":
        resolved = latest_weights_file_path(config)
        if resolved is None:
            raise FileNotFoundError(
                f"No checkpoint files found in '{config['model_dir']}/' matching "
                f"'{config['model_basename']}*.pt'. Train the model first."
            )
        model_path = resolved

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: '{model_path}'")

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        N=config.get("N", 6),
        h=config.get("h", 8),
        dropout=config.get("dropout", 0.1),
        d_ff=config.get("d_ff", 2048),
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded  : {model_path}")
    print(f"Epoch   : {state.get('epoch', 'unknown')}")
    print(f"Step    : {state.get('global_step', 'unknown')}")
    return model


# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------

def preprocess_query(query: str, tokenizer_src, config: dict,
                     device: torch.device) -> tuple:
    seq_len = config["seq_len"]
    sos_id  = tokenizer_src.token_to_id("[SOS]")
    eos_id  = tokenizer_src.token_to_id("[EOS]")
    pad_id  = tokenizer_src.token_to_id("[PAD]")

    src_ids = tokenizer_src.encode(query).ids
    max_src = seq_len - 2
    if len(src_ids) > max_src:
        print(f"WARNING: Query is {len(src_ids)} tokens; truncating to {max_src}.")
        src_ids = src_ids[:max_src]

    num_pad = seq_len - len(src_ids) - 2
    encoder_input = torch.tensor(
        [sos_id] + src_ids + [eos_id] + [pad_id] * num_pad,
        dtype=torch.long
    ).unsqueeze(0).to(device)                            # (1, seq_len)

    encoder_mask = (encoder_input != pad_id).unsqueeze(1).to(device)  # (1, 1, seq_len)

    src_tokens = (
        ["[SOS]"]
        + [tokenizer_src.id_to_token(i) or "[UNK]" for i in src_ids]
        + ["[EOS]"]
    )
    return encoder_input, encoder_mask, src_tokens


# ---------------------------------------------------------------------------
# Greedy decode — same logic as train.py but also collects step probabilities
# ---------------------------------------------------------------------------

def greedy_decode_with_probs(model, source, source_mask, tokenizer_src,
                              tokenizer_tgt, max_len: int,
                              device: torch.device, top_k: int = 10) -> tuple:
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    step_probs = []  # list of np.ndarray (vocab_size,) — one per generated token

    with torch.no_grad():
        encoder_output = model.encode(source, source_mask)
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

        while True:
            if decoder_input.size(1) == max_len:
                break

            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            logits = model.projection(out[:, -1])           # (1, vocab_size)
            probs  = torch.softmax(logits, dim=-1)          # (1, vocab_size)
            step_probs.append(probs.squeeze(0).cpu().numpy())

            _, next_word = torch.max(probs, dim=1)
            decoder_input = torch.cat([
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
            ], dim=1)

            if next_word.item() == eos_idx:
                break

    return decoder_input.squeeze(0), step_probs             # (tgt_len,), list


# ---------------------------------------------------------------------------
# Attention map extraction — one full forward pass after decoding
# ---------------------------------------------------------------------------

def extract_attention_maps(model, encoder_input, encoder_mask,
                            decoded_ids: torch.Tensor, device: torch.device) -> dict:
    tgt_len = decoded_ids.size(0)
    decoder_input = decoded_ids.unsqueeze(0).to(device)      # (1, tgt_len)
    decoder_mask  = causal_mask(tgt_len).to(device)          # (1, 1, tgt_len, tgt_len)

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)
        model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

    N = len(model.encoder.layers)

    encoder_self  = [
        model.encoder.layers[i].self_attention_block.attention_scores
            .squeeze(0).cpu().numpy()                        # (h, src_len, src_len)
        for i in range(N)
    ]
    decoder_self  = [
        model.decoder.layers[i].self_attention_block.attention_scores
            .squeeze(0).cpu().numpy()                        # (h, tgt_len, tgt_len)
        for i in range(N)
    ]
    decoder_cross = [
        model.decoder.layers[i].cross_attention_block.attention_scores
            .squeeze(0).cpu().numpy()                        # (h, tgt_len, src_len)
        for i in range(N)
    ]

    return {
        "encoder_self":  encoder_self,
        "decoder_self":  decoder_self,
        "decoder_cross": decoder_cross,
    }


# ---------------------------------------------------------------------------
# Helper: token ID tensor → list of string labels
# ---------------------------------------------------------------------------

def get_token_labels(token_ids: torch.Tensor, tokenizer) -> list:
    return [tokenizer.id_to_token(tid) or "[UNK]"
            for tid in token_ids.cpu().tolist()]


# ---------------------------------------------------------------------------
# Plot 1 — Cross-attention averaged over all layers & heads
# ---------------------------------------------------------------------------

def plot_cross_attention_avg(attention_maps: dict, src_tokens: list,
                              tgt_tokens: list, save_dir: str,
                              active_src_len: int, active_tgt_len: int) -> None:
    stacked  = np.stack(attention_maps["decoder_cross"], axis=0)  # (N, h, tgt, src)
    avg_attn = stacked.mean(axis=(0, 1))                          # (tgt, src)
    avg_attn = avg_attn[:active_tgt_len, :active_src_len]

    disp_src = src_tokens[:active_src_len]
    disp_tgt = tgt_tokens[:active_tgt_len]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(avg_attn, cmap="Blues", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(disp_src)))
    ax.set_xticklabels(disp_src, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(disp_tgt)))
    ax.set_yticklabels(disp_tgt, fontsize=9)
    ax.set_xlabel("Source (English) Tokens")
    ax.set_ylabel("Generated (Tamil) Tokens")
    ax.set_title("Cross-Attention — Averaged Over All Layers & Heads")

    plt.tight_layout()
    path = os.path.join(save_dir, "cross_attention_avg.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2 — Per-head cross-attention (last decoder layer)
# ---------------------------------------------------------------------------

def plot_cross_attention_per_head(attention_maps: dict, src_tokens: list,
                                   tgt_tokens: list, save_dir: str,
                                   active_src_len: int, active_tgt_len: int) -> None:
    last_cross = attention_maps["decoder_cross"][-1]  # (h, tgt, src)
    h = last_cross.shape[0]

    ncols = 2
    nrows = (h + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.flatten()

    disp_src = src_tokens[:active_src_len]
    disp_tgt = tgt_tokens[:active_tgt_len]

    for head_idx in range(h):
        ax   = axes[head_idx]
        attn = last_cross[head_idx, :active_tgt_len, :active_src_len]

        im = ax.imshow(attn, cmap="Blues", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax)

        if active_src_len <= 30:
            ax.set_xticks(range(active_src_len))
            ax.set_xticklabels(disp_src, rotation=45, ha="right", fontsize=7)
        else:
            ax.set_xticks([])

        if active_tgt_len <= 30:
            ax.set_yticks(range(active_tgt_len))
            ax.set_yticklabels(disp_tgt, fontsize=7)
        else:
            ax.set_yticks([])

        ax.set_title(f"Head {head_idx + 1}")
        ax.set_xlabel("Source Tokens")
        ax.set_ylabel("Target Tokens")

    for idx in range(h, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Cross-Attention Per Head (Last Decoder Layer)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "cross_attention_per_head.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3 — Encoder self-attention (last layer, head-averaged)
# ---------------------------------------------------------------------------

def plot_encoder_self_attention(attention_maps: dict, src_tokens: list,
                                 save_dir: str, active_src_len: int) -> None:
    last_enc = attention_maps["encoder_self"][-1]              # (h, src, src)
    avg_attn = last_enc.mean(axis=0)[:active_src_len, :active_src_len]
    disp_tok = src_tokens[:active_src_len]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_attn, cmap="Greens", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(disp_tok)))
    ax.set_xticklabels(disp_tok, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(disp_tok)))
    ax.set_yticklabels(disp_tok, fontsize=9)
    ax.set_xlabel("Source Token (Key)")
    ax.set_ylabel("Source Token (Query)")
    ax.set_title("Encoder Self-Attention (Last Layer, Head-Averaged)")

    plt.tight_layout()
    path = os.path.join(save_dir, "encoder_self_attention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4 — Token generation probability heatmap
# ---------------------------------------------------------------------------

def plot_generation_probs(step_probs: list, tgt_tokens: list,
                           tokenizer_tgt, save_dir: str, top_k: int = 10) -> None:
    if not step_probs:
        print("  No generation steps to visualize.")
        return

    num_steps = len(step_probs)

    # Collect global top-K token IDs across all steps
    all_top_ids: set = set()
    for probs in step_probs:
        top_ids = np.argsort(probs)[-top_k:]
        all_top_ids.update(top_ids.tolist())
    all_top_ids = sorted(all_top_ids)

    # Build (num_top_tokens, num_steps) probability matrix
    prob_matrix = np.zeros((len(all_top_ids), num_steps), dtype=np.float32)
    for step_idx, probs in enumerate(step_probs):
        for row_idx, tid in enumerate(all_top_ids):
            prob_matrix[row_idx, step_idx] = probs[tid]

    y_labels = [tokenizer_tgt.id_to_token(tid) or f"[ID:{tid}]" for tid in all_top_ids]
    x_labels = [
        f"Step {i + 1}\n({tgt_tokens[i + 1] if i + 1 < len(tgt_tokens) else '?'})"
        for i in range(num_steps)
    ]

    fig_w = max(12, num_steps * 0.9)
    fig_h = max(6, len(all_top_ids) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(prob_matrix, cmap="Reds", aspect="auto",
                   interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Softmax Probability")

    ax.set_xticks(range(num_steps))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticks(range(len(all_top_ids)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Generation Step (chosen token shown below)")
    ax.set_ylabel("Token")
    ax.set_title(f"Token Generation Probabilities (Top-{top_k} Tokens Across All Steps)")

    plt.tight_layout()
    path = os.path.join(save_dir, "generation_probs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_results(query: str, translated_text: str,
                  decoded_ids: torch.Tensor, tokenizer_tgt) -> None:
    line = "=" * 80
    print(line)
    print("  INFERENCE RESULT")
    print(line)
    print(f"  Source (EN)  : {query}")
    print(f"  Translation  : {translated_text}")
    print(line)
    print("  Token-by-token output:")
    for step, tid in enumerate(decoded_ids.cpu().tolist()):
        token_str = tokenizer_tgt.id_to_token(tid) or "[UNK]"
        print(f"    Step {step + 1:>3}: {token_str:<25} (id={tid})")
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    print(f"Vocab   : src={tokenizer_src.get_vocab_size()}, "
          f"tgt={tokenizer_tgt.get_vocab_size()}")

    model = load_model(args.model_path, config, tokenizer_src, tokenizer_tgt, device)

    encoder_input, encoder_mask, src_tokens = preprocess_query(
        args.query, tokenizer_src, config, device
    )

    decoded_ids, step_probs = greedy_decode_with_probs(
        model, encoder_input, encoder_mask,
        tokenizer_src, tokenizer_tgt,
        config["seq_len"], device, top_k=args.top_k
    )

    translated_text = tokenizer_tgt.decode(
        decoded_ids.cpu().numpy().tolist(),
        skip_special_tokens=True
    )

    print_results(args.query, translated_text, decoded_ids, tokenizer_tgt)

    attention_maps = extract_attention_maps(
        model, encoder_input, encoder_mask, decoded_ids, device
    )

    active_src_len = int(encoder_mask.squeeze().sum().item())
    active_tgt_len = int(decoded_ids.size(0))
    tgt_tokens     = get_token_labels(decoded_ids, tokenizer_tgt)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\nGenerating visualizations → {args.save_dir}/")

    plot_cross_attention_avg(
        attention_maps, src_tokens, tgt_tokens,
        args.save_dir, active_src_len, active_tgt_len
    )
    plot_cross_attention_per_head(
        attention_maps, src_tokens, tgt_tokens,
        args.save_dir, active_src_len, active_tgt_len
    )
    plot_encoder_self_attention(
        attention_maps, src_tokens, args.save_dir, active_src_len
    )
    plot_generation_probs(
        step_probs, tgt_tokens, tokenizer_tgt, args.save_dir, top_k=args.top_k
    )

    print(f"\nDone. All plots saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
