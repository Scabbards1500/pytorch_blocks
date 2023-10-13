import math

import torch
import torch.nn as nn

class eca_block(nn.Module):
    def __init__(self,channel,gamma= 2,b = 1):
        super(eca_block,self).__init__() # 初始化父类
        kernel_size = int(abs((math.log(channel, 2)+b )/ gamma)) # 计算卷积核大小
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1 # 卷积核大小必须为奇数
        padding = kernel_size//2 # 填充大小

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 1.全局平均池化.由于是在高和宽上进行的，所以输出的是一个1*1的特征图,这里的1表示输出的通道数
        # 定义1D卷积
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False) # 2.一维卷积，这里的1，1是把它当作序列模型来看
        self.sigmoid = nn.Sigmoid() # 激活函数

    def forward(self, x):
        b, c, h, w = x.size() # b为batch_size,c为通道数,h为高,w为宽
        avg = self.avg_pool(x).view([b,1,c]) # 全局平均池化后，将特征图拉伸成一维向量 b*c*h*w -> b*1*c, 第一个维度是batchsize， 第二个维度是通道数，第二个维度是特征长度，第三个维度代表每个时序
        out = self.conv(avg) # 一维卷积
        out = self.sigmoid(out).view(b,c,1,1) # 激活函数，再reshape成b*c*1*1
        return out*x

#  测试一下
model = eca_block(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
output = model(inputs)

