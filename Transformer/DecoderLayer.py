import torch.nn as nn
from MHSA import MultiHeadAttention
from PWFeedforward import PositionwiseFeedForward
from SinusoidalPE import PositionalEncoding

"""
target_input
     ↓
Masked Multi-Head Self-Attention (只能看自己左边的词)
     ↓
+ Residual + LayerNorm
     ↓
Encoder-Decoder Attention（输入来自 Encoder 输出）
     ↓
+ Residual + LayerNorm
     ↓
Position-wise FeedForward
     ↓
+ Residual + LayerNorm

"""



class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention（不能看到未来的词）
        _x = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(_x)
        x = self.norm1(x)

        # Encoder-Decoder Attention（把 Encoder 的输出作为 KV）
        _x = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = x + self.dropout2(_x)
        x = self.norm2(x)

        # FeedForward
        _x = self.feed_forward(x)
        x = x + self.dropout3(_x)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)  # tgt_mask	自回归 mask，防止 Decoder 偷看后面的词,memory_mask	控制是否能访问 encoder 输出，一般不用特殊处理，设为 None 即可

        return self.norm(x)
