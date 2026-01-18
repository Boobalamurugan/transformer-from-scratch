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
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None: # eps means epsilon: a small value to avoid division by zero
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones((1,)))  # multiplicative parameter - initialized to 1
        self.beta = nn.Parameter(torch.zeros((1,)))  # additive parameter - initialized to 0

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and B2
        self.relu = nn.ReLU()

    def forward(self,x):
        ## (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model:int , h:int,dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # h = number of attention heads

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model//h # dₖ = dimension of key (and query) vectors per head
        self.w_q = nn.Linear(d_model,d_model) # Wq
        self.w_k = nn.Linear(d_model,d_model) # Wk
        self.w_v = nn.Linear(d_model,d_model) # Wv

        self.wo = nn.Linear(d_model,d_model) # Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1]

        ## (batch, h, seq_len,d_k) --> (batch, h,seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask is not None:
            attention_scores.masked_fill_(mask ==0,float("-inf")) 
        attention_scores = attention_scores.softmax(dim = -1) ## (B,h,seq_len,seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value) , attention_scores

    def forward(self,q,k,v,mask):
        query = self.w_q(q) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        ## (Batch, Seq_len, d_model) --> (Batch,Seq_len,h,d_k) --> (Batch,h,Seq_len,d_k)
        batch_size = query.size(0)

        query_len = query.size(1)   # decoder length
        key_len   = key.size(1)     # encoder length

        query = query.view(batch_size, query_len, self.h, self.d_k).transpose(1, 2)
        key   = key.view(batch_size, key_len,   self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, key_len,  self.h, self.d_k).transpose(1, 2)


        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout) 

        ##  (Batch,h,Seq_len,d_k) --> (Batch,seq_len,h,d_k) --> (Batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)


        ## (Batch,seq_len,d_model) --> (Batch,seq_len,d_model)
        return self.wo(x)

class SkipConnection(nn.Module): ## ResidualConnection

    def __init__(self,dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block:MultiHeadAttentionBlock,
                feedforward_block:FeedForwardBlock,
                dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feedforward_block = feedforward_block
        self.skip_connections = nn.ModuleList([SkipConnection(dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.skip_connections[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.skip_connections[1](x,self.feedforward_block)
        return x

class Encoder(nn.Module):

    def __init__(self,layers:nn.ModuleList ):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self,self_attention_block:MultiHeadAttentionBlock,
                 cross_attention_block:MultiHeadAttentionBlock,
                 feedforward_block:FeedForwardBlock,
                 dropout:float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block
        self.skipconnections = nn.ModuleList([SkipConnection(dropout) for _ in range(3)])
    
    def forward(self,x,encoder_op,src_mask,target_mask):
        x = self.skipconnections[0](x,lambda x :self.self_attention_block(x,x,x,target_mask))
        x = self.skipconnections[1](x,lambda x: self.cross_attention_block(x,encoder_op,encoder_op,src_mask))
        x = self.skipconnections[2](x,self.feedforward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self,layers:nn.ModuleList ):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self,x,encoder_op,src_mask,target_mask):
        for layer in self.layers:
            x = layer(x,encoder_op,src_mask,target_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()

        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        ## (Batch,seq_len,d_model) --> (Batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x),dim=-1)


class Transformer(nn.Module):

    def __init__(self,encoder:Encoder,
                 decoder:Decoder,
                 src_embed:InputEmbedding,
                 target_embed:InputEmbedding,
                 src_pos:PositionalEncoding,
                 target_pos:PositionalEncoding,
                 projection_layer:ProjectionLayer
                ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder 
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer =projection_layer
    
    def encode(self,src,src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_op,src_mask,target,target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target,encoder_op,src_mask,target_mask)
    
    def projection(self,x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size:int,
                      target_vocab_size:int,
                      src_seq_len:int,
                      target_seq_len:int,
                      d_model:int = 512,
                      N:int = 4,
                      h:int = 4,
                      dropout:float=0.1,
                      d_ff:int=1048)-> Transformer:
    
    ## Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    target_embed = InputEmbedding(d_model, target_vocab_size)

    ## Create Postional encoding layers
    src_pos = PositionalEncoding(d_model,src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model,target_seq_len, dropout)
    
    ## Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feedforward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feedforward_block,dropout)
        encoder_blocks.append(encoder_block)

    ## create decoder block

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feedforward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feedforward_block,dropout)
        decoder_blocks.append(decoder_block)

    ## Create the encoder and decoder 
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    ## Create the projection layer
    projection_layer = ProjectionLayer(d_model,target_vocab_size)

    ## build the transformer
    transformer = Transformer(encoder,decoder,src_embed,target_embed,src_pos,target_pos,projection_layer)

    ## init the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer


