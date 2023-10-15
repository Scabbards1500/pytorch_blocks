import torch

# n_pos_vec 是句子总长度，embedding dim维度
def create_1d_absolute_sincos_embeddings(n_pos_vec,dim):
    assert dim % 2 == 0, "wrong dimension" # dim must be even
    # 初始化position embedding
    position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float) #numel()返回数组元素个数
    # omega是对i进行遍历
    omega = torch.arange(dim//2, dtype=torch.float) #//是整除
    omega /= dim/2.
    omega = 1./(10000**omega)

    out = n_pos_vec[:, None]@omega[None, :] # 先把n_pos_vec变成列向量，一个维度加上None相当于扩了一维；接下来是把omega拓成一个行向量， @是矩阵乘法
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    # 接下来是偶数位用sin赋值，奇数位用cos去赋值
    position_embedding[:, 0::2] = emb_sin
    position_embedding[:, 1::2] = emb_cos

    return position_embedding

if __name__ == '__main__':
    n_pos = 4
    dim = 4
    n_pos_vec = torch.arange(n_pos, dtype=torch.float) # 生成一个长度为n_pos的向量
    print(n_pos_vec)
    pe = create_1d_absolute_sincos_embeddings(n_pos_vec,dim)
    print("pe", pe)




