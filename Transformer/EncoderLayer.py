import torch.nn as nn
from MHSA import MultiHeadAttention
from PWFeedforward import PositionwiseFeedForward
from SinusoidalPE import PositionalEncoding

# EncoderLayer 将注意力和前馈组合在一起，还加了 LayerNorm 和残差连接

"""
Input
  ↓
LayerNorm
  ↓
Multi-Head Attention + 残差连接
  ↓
LayerNorm
  ↓
Feed Forward Network + 残差连接
  ↓
Output
"""


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)  # d_ff是前馈网络的隐藏层维度
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Multi-head attention + 残差 + Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)  # 残差连接

        # Feed Forward + 残差 + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)  # 残差连接

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None): # src是输入序列的 token 编号（也就是词的 index 序列），通常叫做 source sequence
        x = self.embedding(src)  # shape: (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


