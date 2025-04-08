import torch
import math

def scaled_dot_product_attention(Q, K, V):
    # input Q,K (batch_size, seq_len, d_k),d_k 是 Q 和 K 的 特征维度
    # 1. 计算 Q 和 K 的点积
    matmul_qk = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, seq_len, seq_len)

    # 2. 缩放
    dk = Q.size()[-1]  # dk是K/Q向量的维度
    scaled_attention_logits = matmul_qk / math.sqrt(dk)
    # 3. softmax
    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1) # softmax 操作是在最后一个维度（即每个 Query 对所有 Key 的相关性得分）上进行的,这样，每个 Query 都会根据所有 Key 的相似度得分，得到一组权重，这就是每个词在处理时需要关注的其他词。

    # 4. 加权 V
    output = torch.matmul(attention_weights, V)  # (batch_size, seq_len, d_model) torch.matmul 是一个用于执行矩阵乘法的函数

    return output, attention_weights
