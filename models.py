# Imports
import torch
import torch.nn as nn
import math


# Define the Encoder Architecture using LSTM
class Encoder(nn.Module):
    def __init__(self, source_vectors,embedding_dim, hidden_dim,n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(source_vectors, freeze=False)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim, n_layers,bidirectional=True,dropout=dropout,batch_first=True)

    def forward(self, X):
        # X shape = [Batch_Size X Sequence_Len X 1]
        X = self.embedding(X)
        # X shape = [Batch_Size X Sequence_Len X Embedding_Dim]
        assert X.shape[0] > 0 and X.shape[1] > 0
        X, (hidden_state, cell_state) = self.lstm(X)
        # X shape = [Batch_Size X Seq_Len X Hidden_Dim] , Hidden_State_Shape = Cell_State_Shape = [Num_Layers X Batch_Size X Hidden_Dim]
        return hidden_state, cell_state


# Define the Decoder Architecture using LSTM
class Decoder(nn.Module):
    def __init__(self, source_vectors,target_vocab_size, embedding_dim,hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.target_vocab_size = target_vocab_size
        self.embedding = nn.Embedding.from_pretrained(source_vectors, freeze=False)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim, n_layers,bidirectional=True,dropout=dropout,batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2,target_vocab_size)  # bidrectional hence

    def forward(self, hidden_state, cell_state, Y,
                force_correction=0.5):
        # Hidden_State_Shape = Cell_State_Shape = [Num_Layers X Batch_Size X Hidden_Dim]
        # Y Shape = [Batch_Size X Sequence_Len]

        batch_size, seq_len = Y.shape[0], Y.shape[1]
        outputs = torch.zeros(seq_len, batch_size,self.target_vocab_size,requires_grad=True).to(device)  # [Batch_Size X Sequence_Len]

        X = Y[:, 1]
        # X shape = [Batch_Size X 1]
        for i in range(seq_len):
            X = X.unsqueeze(1)
            # X shape = [Batch_Size X 1 X 1]
            decoder_input = self.embedding(X)
            # decoder_input_shape = [Batch_Size X 1 X Embedding_Dim]
            assert decoder_input.shape[0] > 0 and decoder_input.shape[1] > 0
            decoder_output, (hidden_state, cell_state) = self.lstm(decoder_input,(hidden_state, cell_state))
            # Decoder_Output_Shape = [Batch_Size X 1 X Target_Vocab_Size]
            decoder_output = self.fc(decoder_output)
            # Store output
            outputs[i] = decoder_output.permute(1,0,2).squeeze(0)
            _, indexes = decoder_output.max(dim=2)
            # indexes shape = [Batch_Size X 1]
            indexes = indexes.squeeze(1)
            # use indexes as next input or correct it
            X = indexes if random.random() < 0.5 else Y[:,i]
            # indexes shape = X shape = [Batch_Size]

        # Output Shape = [Seq_Len X Batch_Size X Target_Vocab_Size]
        outputs = outputs.permute(1, 0, 2)
        outputs = outputs.reshape(-1,self.target_vocab_size)
        # Output Shape = [Batch_Size X Seq_Len X Target_Vocab_Size]
        return outputs

class EncDecLSTM(nn.Module):
    def __init__(self, enc, dec):
        super(EncDecLSTM, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, X, Y):
        hidden_state, cell_state = self.enc(X)
        output = self.dec(hidden_state,cell_state, Y)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention,self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q,K, V,mask=None):
        attn_scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length,self.num_heads,self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length,self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,
                 dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x,mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask,tgt_mask):
        attn_output = self.self_attn(x, x, x,tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x,enc_output,enc_output,src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

