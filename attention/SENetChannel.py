import torch
from torch import nn


class senet(nn.Module):
    def __init__(self, channel, ratio=16):  # channel为输入通道数,ratio为压缩比
        super(senet, self).__init__()  # 初始化父类
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 1.全局平均池化.由于是在高和宽上进行的，所以输出的是一个1*1的特征图
        self.fc = nn.Sequential(  # 2.两个全连接层
            nn.Linear(channel, channel // ratio, False),  # 第一次全连接神经元个数较少，输入通道数，输出通道数，是否偏置
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(channel // ratio, channel, False),  # 第二次全连接神经元个数和输入特征层相同。
            nn.Sigmoid(),  # 激活函数
        )

    def forward(self, x):
        b, c, h, w = x.size()  # b为batch_size,c为通道数,h为高,w为宽
        avg = self.avg_pool(x)  # 全局平均池化后，将特征图拉伸成一维向量 b*c*h*w -> b*c*1*1
        avg = avg.view(b, c)  # b*c*1*1 -> b*c
        fc = self.fc(avg).view([b, c, 1, 1])  # 对平均池化后的特征图进行两次全连接,再view reshape方便后面的处理 #b*c -> b*c*1*1
        print(fc) # 打印测试一下
        return x * fc


#  测试一下
model = senet(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
output = model(inputs)
