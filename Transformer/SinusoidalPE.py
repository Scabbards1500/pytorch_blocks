import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
# 正余弦位置编码 ≈ “在高维空间中的坐标”
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 你的代码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建一个形状为 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model) # pe= max_len，dim
        position = torch.arange(0, max_len).unsqueeze(1)  # 构造一个形状为 [max_len, 1] 的位置索引矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # d_model/2*常数 # 为不同维度准备好频率缩放因子,作用和 Python 原生的 range() 差不多，是用来生成一个连续数字的 一维 Tensor。

        # 奇偶交替使用 sin 和 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model) 这一步是在做维度对齐，为后续能和输入的 token embeddings 正确相加。
        self.register_buffer('pe', pe)  # 不参与训练

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # 添加前 seq_len 个位置编码
        x = x + self.pe[:, :seq_len]
        return x


pe = PositionalEncoding(d_model=32)
x = torch.zeros(1, 100, 32)
pe_out = pe(x)[0]  # 取第一个 batch
plt.figure(figsize=(10, 6))
plt.plot(pe_out[:, :4])  # 画前4个维度

plt.title("前4维的位置编码随位置变化")
plt.xlabel("Position")
plt.ylabel("Value")
plt.legend([f"dim {i}" for i in range(4)])
plt.show()