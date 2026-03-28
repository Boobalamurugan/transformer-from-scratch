import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Causal mask function
# ---------------------------
def causal_mask(size):
    """Returns a causal mask for decoder (upper-triangular)."""
    mask = torch.triu(torch.ones(1, size, size, dtype=torch.bool), diagonal=1)
    return ~mask  # True where allowed, False where masked

# ---------------------------
# Dataset class
# ---------------------------
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.sos_token_src = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token_src = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token_src = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.long)

        self.sos_token_tgt = torch.tensor([tokenizer_target.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token_tgt = torch.tensor([tokenizer_target.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token_tgt = torch.tensor([tokenizer_target.token_to_id("[PAD]")], dtype=torch.long)

        # Pre-tokenize everything for speed
        self.encoded_pairs = []
        self.raw_pairs = []
        for item in ds:
            src_text = item['translation'][src_lang]
            tgt_text = item['translation'][target_lang]
            src_ids = tokenizer_src.encode(src_text).ids
            tgt_ids = tokenizer_target.encode(tgt_text).ids
            self.encoded_pairs.append((src_ids, tgt_ids))
            self.raw_pairs.append((src_text, tgt_text))

        # Precompute causal mask
        self.causal_mask = causal_mask(seq_len)

    def __len__(self):
        return len(self.encoded_pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.raw_pairs[idx]
        src_ids, tgt_ids = self.encoded_pairs[idx]

        # Truncate if longer than seq_len
        src_ids = src_ids[: self.seq_len - 2]
        tgt_ids = tgt_ids[: self.seq_len - 1]

        # Padding counts
        enc_pad = self.seq_len - len(src_ids) - 2
        dec_pad = self.seq_len - len(tgt_ids) - 1

        # Encoder input: [SOS] + src + [EOS] + PAD
        encoder_input = torch.cat([
            self.sos_token_src,
            torch.tensor(src_ids, dtype=torch.long),
            self.eos_token_src,
            torch.full((enc_pad,), self.pad_token_src.item(), dtype=torch.long)
        ])

        # Decoder input: [SOS] + tgt + PAD
        decoder_input = torch.cat([
            self.sos_token_tgt,
            torch.tensor(tgt_ids, dtype=torch.long),
            torch.full((dec_pad,), self.pad_token_tgt.item(), dtype=torch.long)
        ])

        # Labels: tgt + [EOS] + PAD
        label = torch.cat([
            torch.tensor(tgt_ids, dtype=torch.long),
            self.eos_token_tgt,
            torch.full((dec_pad,), self.pad_token_tgt.item(), dtype=torch.long)
        ])

        # Masks
        pad_id_enc = self.pad_token_src.item()
        pad_id_dec = self.pad_token_tgt.item()

        encoder_mask = (encoder_input != pad_id_enc).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)
        decoder_mask = (decoder_input != pad_id_dec).unsqueeze(0).unsqueeze(0) & self.causal_mask  # (1,1,seq_len) & (1,seq_len,seq_len) -> (1,seq_len,seq_len)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "target_text": tgt_text
        }

# ---------------------------
# DataLoader helper
# ---------------------------
def create_dataloader(dataset, batch_size=16, shuffle=True, num_workers=8):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

# ---------------------------
# Usage example
# ---------------------------
# Assuming you have:
# ds = list of dictionaries with 'translation': {'en': '...', 'ta': '...'}
# tokenizer_src = trained tokenizer for English
# tokenizer_target = trained tokenizer for Tamil

# seq_len = 350 (or smaller for faster training)
# batch_size = 16 (increase if GPU memory allows)
# num_workers = 8 (adjust depending on CPU cores)

# dataset = BilingualDataset(ds, tokenizer_src, tokenizer_target, 'en', 'ta', seq_len=350)
# dataloader = create_dataloader(dataset, batch_size=16, num_workers=8)
