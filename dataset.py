import torch
import torch.nn as nn
from torch.utils.data import Dataset

def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask == 0

class BilingualDataset(Dataset):

    def __init__(self,ds,tokenizer_src,tokenizer_target,src_lang,target_lang,seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")],dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        enc_ip_tokens = self.tokenizer_src.encode(src_text).ids
        dec_ip_tokens = self.tokenizer_src.encode(target_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_ip_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_ip_tokens) - 1

        # if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
        #     raise ValueError('sentence is too long')

        ## truncate if too long (safeguard)
        enc_ip_tokens = enc_ip_tokens[: self.seq_len - 2]
        dec_ip_tokens = dec_ip_tokens[: self.seq_len - 1]

        enc_num_padding_tokens = self.seq_len - len(enc_ip_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_ip_tokens) - 1


        ## add sos and eos
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_ip_tokens,dtype=torch.int64),
                self.eos_token,
                # torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.long)
            ]
        )

        ## add sos to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_ip_tokens,dtype=torch.int64),
                # torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.long)
            ]
        )

        ## add eos to the label 
        label = torch.cat(
            [
                torch.tensor(dec_ip_tokens,dtype=torch.int64),
                self.eos_token,
                # torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.long)

            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        pad_id = self.pad_token.item()
        encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0)
        decoder_mask = (
            (decoder_input != pad_id).unsqueeze(0).unsqueeze(0) & causal_mask(decoder_input.size(0))
        )

        return {
            "encoder_input":encoder_input, #(Seq_len)
            "decoder_input":decoder_input,
            # "encoder_mask":(encoder_input!=self.seq_len).unsqueeze(0).unsqueeze(0).int(), ## (1,1,seq_len)
            # "decoder_mask":(decoder_input!=self.seq_len).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "encoder_mask":encoder_mask,
            "decoder_mask":decoder_mask,
            "label":label,
            "src_text":src_text,
            "target_text":target_text
        }
    

