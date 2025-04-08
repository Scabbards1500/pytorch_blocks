from Mask_Func import create_padding_mask, create_look_ahead_mask,combine_masks
import torch
def predict(model, src, max_len=50, pad_token=0, start_token=1, end_token=2):
    batch_size = src.size(0)
    tgt = torch.full((batch_size, 1), start_token, dtype=torch.long).to(src.device)  # 初始 token
    memory_mask = None  # 可以选择是否给 Encoder 输出加 mask（通常不需要）

    # 对目标序列逐步生成
    for _ in range(max_len):
        tgt_mask = create_look_ahead_mask(tgt.size(1)).to(src.device)
        tgt_padding_mask = create_padding_mask(tgt, pad_token).to(src.device)
        mask = combine_masks(tgt_mask, tgt_padding_mask)  # 合并 masks

        # 模型输出
        output = model(src, tgt, tgt_mask=mask, memory_mask=memory_mask)

        # 选择最后一个位置的输出（batch_size, seq_len, vocab_size）
        logits = output[:, -1, :]  # shape: [batch_size, vocab_size]

        # 计算下一个 token 的概率
        probs = torch.softmax(logits, dim=-1)

        # 选择概率最大的位置作为下一个 token
        next_token = torch.argmax(probs, dim=-1).unsqueeze(1)  # shape: [batch_size, 1]

        # 将下一个 token 拼接到目标序列
        tgt = torch.cat([tgt, next_token], dim=1)

        # 如果生成了 `<end>` token，就提前结束
        if torch.all(next_token == end_token):
            break

    return tgt

# 示例：使用模型进行推理
src = torch.randint(0, 5000, (2, 20))  # 输入一个 batch 的源序列
predicted_tokens = predict(model, src)

# 输出预测的 token 序列
print(predicted_tokens)  # 输出形如 [[start_token, word1, word2, ..., end_token]]
