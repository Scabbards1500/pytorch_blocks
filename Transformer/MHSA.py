import torch
import torch.nn as nn
import math

#捕捉不同子空间的注意力特征

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 用于生成多个头的 Q, K, V 的线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 最后的线性层，用来合并多个头的输出
        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """ 将输入 x 切分成多个头 """
        batch_size, seq_len, d_model = x.shape  # d_model表示模型中每个输入、输出向量的 维度, 是 Transformer 中的 嵌入维度（embedding dimension） 或 特征维度（feature dimension）
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # 调整形状为 (batch_size, num_heads, seq_len, d_k)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        # 第一步：生成多个头的 Q, K, V
        Q = self.split_heads(self.W_q(query))  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(self.W_k(key))  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(self.W_v(value))  # (batch_size, num_heads, seq_len, d_k)

        # 第二步：计算每个头的 attention（注意力）权重
        # 计算点积注意力
        attention_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.nn.functional.softmax(attention_logits, dim=-1)

        # 第三步：加权求和 V，得到每个头的输出
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, d_k)

        # 第四步：拼接所有头的输出
        attention_output = attention_output.permute(0, 2, 1, 3)  # 调整为 (batch_size, seq_len, num_heads, d_k)
        attention_output = attention_output.contiguous().view(batch_size, -1, self.d_model) # contiguous() 是一个方法，用于确保张量在内存中是连续存储的。它返回一个连续的张量，如果原始张量已经是连续的，则直接返回原张量；如果不连续，则会创建一个新的连续副本，-1 告诉 PyTorch，"根据其他维度的大小，自动计算这个维度的大小"，使得张量的总元素数量保持不变。

        # 第五步：通过线性层生成最终的输出
        output = self.fc_out(attention_output)  # (batch_size, seq_len, d_model)
        return output, attention_weights
