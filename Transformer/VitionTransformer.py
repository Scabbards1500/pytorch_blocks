# 此处是vision transformer模块的教学

import torch
import torch.nn as nn
import torch.nn.functional as F

#######step 1 #######
# conver image to embedding vector sequence

# 一般的实现,直接展开
def image2emb_naive(image, patch_size, weight):  # image是输入头像， patch_size是被切成的小图片的大小，weight是线性变换权重
    # image shape = bs*channel*h*w
    patch = F.unfold(image, kernel_size=patch_size,  # stride就说每一个图片块的大小，transformer中embedding大小和模型大小一样
                     stride=patch_size).transpose(-1, -2)  # transpose是为了让它能够相乘
    patch_embedding = patch @ weight
    return patch_embedding

    print(patch.shape)


# 用卷积来实现！
#首先对图像进行一个二维卷积，然后拉直，就可以得到一维的embedding序列
def image2emb_conv(image, kernel, stride):
    conv_output = F.conv2d(image, kernel, stride=stride)  # output = bs*oc*oh*ow
    # 输出会把output featuremap拉直
    bs, oc, oh, ow = conv_output.shape
    patch_embedding = conv_output.reshape((bs, oc, oh * ow)).transpose(-1, -2)
    return patch_embedding


# oc就是embeddingsize，oh*ow就是sequence的长度
# **定义kernel，跟weight相似


# test code for image2emb
bs, ic, image_h, image_w = 1, 3, 8, 8  # batch size, input channel
patch_size = 4
model_dim = 8
max_num_token = 16
num_classes = 10
label = torch.randint(10, (bs,))
patch_depth = patch_size * patch_size * ic  # 每个patch包含通道来着
# 生成一个图片用来测试
image = torch.randn(bs, ic, image_h, image_w)
# weight就是patch-》embedding的乘法矩阵
weight = torch.randn(patch_depth, model_dim)  # [48,8] #model dim是输出通道数，patch_depth是卷积核的面积*输入通道数
patch_embedding_naive = image2emb_naive(image, patch_size,
                                        weight)  # torch.Size([1, 48, 4])1:batchsize 48: input channel;4:（8*8图片/4*4小块大小）块数
print(patch_embedding_naive.shape)  # [1,4,8] 一个图片被分成了四块，每一块用一个长度为8的向量表示

# 测试卷积embedding
kernel = weight.transpose(0, 1).reshape((-1, ic, patch_size, patch_size))  #kernel形状应该是oc*ic*kh*kw 卷积核的大小就是上文中的patchsize
patch_embedding_conv = image2emb_conv(image, kernel, patch_size)
print(patch_embedding_conv.shape)  # [1,4,8]


# ####step2#######
# 加上classification token embedding
# 按照文章的意思此处embedding是随机初始化的
cls_token_embedding = torch.randn(bs, 1, model_dim, requires_grad=True) # requires_grad表示可训练
token_embedding = torch.cat([cls_token_embedding,patch_embedding_conv],dim=1)#在位置上去拼接，不是在batchsize或者通道，因此dim= 1,中间那个维度

# ####step3#######
# add position embedding 取出位置编码
position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
seq_len = token_embedding.shape[1]
position_embedding = torch.tile(position_embedding_table[:seq_len],[token_embedding.shape[0], 1, 1]) #通过重复 input 的元素构造一个张量。 dims 参数指定每个维度中的重复次数。
token_embedding += position_embedding  # 把position embedding 和 token embeddeding加起来
# 序列的长度
seq_len = token_embedding.shape[1]


# ####step4#######
# 把embedding 放入 transformer encoder， 这里pytorch自己就自带实现
encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)  # nhead是干嘛的？
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)  # num_layers是干嘛的？
encoder_output = transformer_encoder(token_embedding)

# ####step5#######
# 取出classification位置上的特征输出，映射到类别上算出一个概率分布
cls_token_output = encoder_output[:, 0, :]  # batch_size, 位置, 通道数 #得到第一个位置上encoder的输出
linear_layer = nn.Linear(model_dim, num_classes) #分类之前对encoder output做个映射
logits = linear_layer(cls_token_output)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, label)
print(loss)
