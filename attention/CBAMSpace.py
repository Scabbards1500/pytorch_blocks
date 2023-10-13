# 是通道注意力和空间注意力的结合
import torch
from torch import nn


# 通道注意力
class channel_attention(nn.Module):
    def __init__(self,channel, ratio = 16):     # channel为输入通道数,ratio为压缩比
        super(channel_attention, self).__init__() # 初始化父类
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 全局最大池化.由于是在高和宽上进行的，所以输出的是一个1*1的特征图
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化.由于是在高和宽上进行的，所以输出的是一个1*1的特征图
        self.fc = nn.Sequential( # 2.两个全连接层, 用来学习通道之间的关系
            nn.Linear(channel, channel//ratio, False) # 第一次全连接神经元个数较少，输入通道数，输出通道数，是否偏置
            ,nn.ReLU(inplace=True) # 激活函数
            ,nn.Linear(channel//ratio, channel, False) # 第二次全连接神经元个数和输入特征层相同。
        )
        self.sigmoid = nn.Sigmoid() # 激活函数

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view([b,c]) # 全局最大池化后，将特征图拉伸成一维向量 b*c*h*w -> b*c*1*1, 再view reshape方便后面的处理, b*c*1*1 -> b*c
        avg_pool_out = self.avg_pool(x).view([b,c]) # 全局平均池化后，将特征图拉伸成一维向量 b*c*h*w -> b*c*1*1, 再view reshape方便后面的处理, b*c*1*1 -> b*c
        #再用全连接层进行处理
        max_fc_out = self.fc(max_pool_out).view([b,c,1,1]) # b*c -> b*c*1*1
        avg_fc_out = self.fc(avg_pool_out).view([b,c,1,1]) # b*c -> b*c*1*1
        #相加
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out) # 激活函数
        return x * out # 通道注意力机制的输出

# 空间注意力
class spatial_attention(nn.Module):
    def __init__(self,kernel_size = 7): # kernel_size为卷积核的大小
        super(spatial_attention, self).__init__() # 初始化父类
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding=kernel_size//2,bias=False) # 1.卷积层
        self.sigmoid = nn.Sigmoid() # 激活函数

    def forward(self, x):
        max_out,_ = torch.max(x, dim=1, keepdim=True) # 最大池化, _为最大值的索引,这里不需要;keepdim=True保持维度不变
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        pool_out = torch.cat([avg_out,max_out], dim=1) # 拼接
        out = self.conv(pool_out) # 卷积
        out = self.sigmoid(out) # 激活函数
        return x * out # 空间注意力机制的输出

# CBAM模块, 结合通道注意力机制和空间注意力机制
class CBAM(nn.Module):
    def __init__(self, channel, ratio = 16, kernel_size = 7):
        super(CBAM, self).__init__()
        # 定义通道注意力机制
        self.channel_attention = channel_attention(channel, ratio)
        # 定义空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self,x):
        x = self.channel_attention(x) # 通道注意力机制的输出
        x = self.spatial_attention(x) # 空间注意力机制的输出
        return x

#  测试一下
model = CBAM(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
output = model(inputs)





