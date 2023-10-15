import torch
import torch.nn as nn
def create_1d_absolute_trainable_embeddings(n_pos_vec,dim):
    # 传入索引
    # n_pos_vec: torch.aramge(n_pos, dtype=torch.float)
    # 因为可学习所以用nn.embedding来实现
    position_embedding = nn.Embedding(n_pos_vec.numel(), dim)
    # 初始化weight(parameter class)
    nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding  # 一维的，绝对的，可学习的embedding


