import torch.nn as nn
import torch
from Mask_Func import create_padding_mask, create_look_ahead_mask,combine_masks


# 交叉熵损失（忽略 padding 部分）
criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 学习率调度（线性预热 + 衰减）
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * (warmup_steps ** -1.5)))

def train(model, train_loader, optimizer, scheduler, criterion, num_epochs=10):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            optimizer.zero_grad()  # 清空梯度

            # 对目标序列进行移位，得到目标的输入和目标值
            tgt_input = tgt[:, :-1]  # 去掉最后一个 token，作为输入
            tgt_output = tgt[:, 1:]  # 去掉第一个 token，作为目标输出

            # 生成 mask
            src_mask = create_padding_mask(src).to(src.device)
            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(src.device)
            tgt_padding_mask = create_padding_mask(tgt_input).to(src.device)
            mask = combine_masks(tgt_mask, tgt_padding_mask)

            # 前向传播
            output = model(src, tgt_input, src_mask=src_mask, tgt_mask=mask)

            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            loss.backward()  # 反向传播

            # 更新参数
            optimizer.step()
            scheduler.step()  # 更新学习率

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

def evaluate(model, val_loader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    with torch.no_grad():  # 评估时不需要计算梯度
        for src, tgt in val_loader:
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_padding_mask(src).to(src.device)
            tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(src.device)
            tgt_padding_mask = create_padding_mask(tgt_input).to(src.device)
            mask = combine_masks(tgt_mask, tgt_padding_mask)

            output = model(src, tgt_input, src_mask=src_mask, tgt_mask=mask)

            loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            total_loss += loss.item()

    print(f'Validation Loss: {total_loss / len(val_loader)}')



# 训练数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 开始训练
train(model, train_loader, optimizer, scheduler, criterion)
