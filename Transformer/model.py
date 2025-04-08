from torch import nn
from EncoderLayer import TransformerEncoder
from DecoderLayer import TransformerDecoder
import torch

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src, src_mask)  # 编码 source 序列
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)  # 解码目标序列
        output = self.output_projection(dec_output)  # 输出到词表维度（用于 softmax）
        return output

model = Transformer(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    max_len=100
)

src = torch.randint(0, 5000, (2, 20))  # batch_size=2, seq_len=20
tgt = torch.randint(0, 5000, (2, 20))  # batch_size=2, seq_len=20

out = model(src, tgt)
print(out.shape)  # 应该是 [2, 20, 5000]
