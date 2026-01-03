import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self,d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ## create positional encoding matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        ## create position tensor of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        
        ## apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)

        ## apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, seq_len, d_model)

        ## register as buffer to avoid being considered a model parameter
        self.register_buffer('pe', pe)
    
    def forward(self,x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)