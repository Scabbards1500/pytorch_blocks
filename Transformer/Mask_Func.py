import torch
def create_padding_mask(seq, pad_token=0):
    # seq: [batch_size, seq_len]
    return (seq == pad_token).unsqueeze(1).unsqueeze(2)
    # 返回 shape: [batch_size, 1, 1, seq_len]，用于 broadcast

def create_look_ahead_mask(size): #Look-Ahead Mask（用于 decoder 自注意力）
    # 创建一个 shape: [size, size] 的上三角矩阵
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 1  # True 表示要 mask 掉

def combine_masks(pad_mask, look_ahead_mask):
    return pad_mask | look_ahead_mask  # shape 必须 broadcast-able
