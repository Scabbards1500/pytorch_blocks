import torch
import torch.nn as nn
def create_2d_relative_bias_trainable_embeddings(n_head,height,width,dim):
    # embeddings的行数就是bias的个数，列数就是num_heads
    # 横轴取值 width：5[0,1,2,3,4] bias ={-width+1, width-1 }{-4,4} 4-（-4）+1 = 9
    # 纵轴取值 height：5[0,1,2,3,4] bias ={-height+1, height-1} 1-（-1）+1 = 3
    position_embedding = nn.Embedding((2*width-1)*(2*height-1), n_head)
    # 初始化weight(parameter class)
    nn.init.constant_(position_embedding.weight, 0.)
    # 获取window中二维的，两两之间的位置偏差
    # step1：算出横轴和纵轴各自的位置偏差，用网格法把横轴的位置索引和纵轴的位置索引定义出来
    def get_2d_relative_position_index(height, width):
        m1, m2 = torch.meshgrid(torch.arange(height), torch.arange(width)) # m1行一样，m2列一样
        coords = torch.stack([m1, m2]) # 把m1和m2拼接起来，dim=-1表示最后一个维度 #2*height*width
        coords_flatten = torch.flatten(coords,1) # 把coords压缩成一维，dim=1表示第一个维度,得到2*【height*width】
        ralative_coords_bias = coords_flatten[:, :, :None]- coords_flatten[:, None, :]#得到网格里任意两点横轴纵轴坐标的差值,[2,height*width,height*width]
        # 把它们都变成正数
        ralative_coords_bias[0, :, :] += height-1 # 横轴坐标的差值,0代表高度维
        ralative_coords_bias[1, :, :] += width-1 # 纵轴坐标的差值 1代表宽度维
        # 把两个方向上的坐标转化成一个方向上的坐标，类似于把一个2dtensor赋值到1dtensor
        # A；2d，B:1d B[i*cols+j] = A[i,j]
        ralative_coords_bias[0,:,:] += ralative_coords_bias[1, :, :].max()+1 # 把横轴坐标的差值转化成一维坐标，即i*cols
        # 相对位置索引
        return ralative_coords_bias.sum(0) # [height*width,height*width] # 两个方向上的坐标相加，得到相对位置索引
    relative_position_bias = get_2d_relative_position_index(height, width) # [height*width,height*width]
    bias_embedding = position_embedding(torch.flatten(relative_position_bias)).reshape(height*width,height*width,n_head) # [height*width,height*width,n_head]
    bias_embedding.permute(2,0,1).unsqueeze(0) # [1, n_head,height*width,height*width]
    return bias_embedding # 二维的，相对的，可学习的embedding